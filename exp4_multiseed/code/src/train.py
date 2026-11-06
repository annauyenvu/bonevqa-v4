import argparse
import math
import os
import random
import time
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .models.bonevqa import BoneVQAModel, normalize_batch
from .utils import (compute_metrics, count_parameters, format_params, get_device, gpu_memory_str, load_config,
                    load_json, normalize_answer, plot_training_curves, save_json, set_seed, setup_env_dirs, setup_logger)

setup_env_dirs()
logger = setup_logger()


class FakeVQADataset(Dataset):
    def __init__(self, n: int = 16, seed: int = 0):
        rng = random.Random(seed)
        self.samples = []
        regions = ["hand", "leg", "hip", "shoulder"]
        for i in range(n):
            closed = rng.random() < 0.5
            region = rng.choice(regions)
            if closed:
                q, a = "Is there a fracture in this image?", rng.choice(["yes", "no"])
            else:
                q, a = "Which body part is shown in this X-ray?", region
            self.samples.append({"question": q, "answer": a, "answer_type": "CLOSED" if closed else "OPEN",
                                 "region": region, "image_id": f"fake_{i}", "seed": rng.randint(0, 10 ** 6)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = dict(self.samples[idx])
        rng = np.random.default_rng(s.pop("seed"))
        img = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        mask = np.zeros((1, 256, 256), dtype=np.uint8)
        y0, x0 = rng.integers(20, 120, 2)
        mask[0, y0:y0 + 80, x0:x0 + 80] = 1
        s.update({"image": img, "masks": mask, "boxes": [[int(x0), int(y0), int(x0) + 80, int(y0) + 80]]})
        return s


def default_collate(samples: List[dict]) -> dict:
    keys = ["image", "question", "answer", "answer_type", "region", "masks", "boxes", "image_id"]
    return {f"{k}s" if not k.endswith("s") else k: [s.get(k) for s in samples] for k in keys}


def load_data_api():
    try:
        from .data import build_dataset
        try:
            from .data import collate_fn
        except Exception:
            try:
                from .data.collate import collate_fn
            except Exception:
                collate_fn = default_collate
        return build_dataset, collate_fn, True
    except Exception as exc:
        logger.warning("Chưa import được src.data (%s) → dùng dataset giả để smoke test", exc)
        return None, default_collate, False


def build_split(build_dataset, name: str, split: str, max_samples: Optional[int], real: bool,
                loader_kwargs: Optional[dict] = None):
    if not real:
        n = max_samples or 16
        return FakeVQADataset(n if split == "train" else max(4, n // 2), seed=0 if split == "train" else 1)
    candidates = [split] if split != "val" else ["val", "validation", "test"]
    if split not in ("train", "val"):
        candidates = [split]
    last_exc = None
    for sp in candidates:
        try:
            return build_dataset(name, sp, max_samples, **(loader_kwargs or {}))
        except TypeError:
            try:
                return build_dataset(name, sp, max_samples)
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Không tạo được split {split} cho {name}: {last_exc}")


def collect_answers(dataset, limit: int = 20000) -> List[str]:
    for attr in ("answers", "answer_list"):
        if hasattr(dataset, attr):
            return list(getattr(dataset, attr))
    for attr in ("samples", "records"):
        items = getattr(dataset, attr, None)
        if items and isinstance(items[0], dict) and "answer" in items[0]:
            return [s["answer"] for s in items]
    out = []
    for i in range(min(len(dataset), limit)):
        try:
            out.append(dataset[i]["answer"])
        except Exception:
            continue
    return out


def build_answer_vocab(answers: List[str], size: int) -> List[str]:
    from collections import Counter

    cnt = Counter(normalize_answer(a) for a in answers if a)
    vocab = ["yes", "no"]
    for a, _ in cnt.most_common():
        if a and a not in vocab:
            vocab.append(a)
        if len(vocab) >= size:
            break
    return vocab


def load_init_weights(model, path: str):
    if not path or not os.path.exists(path):
        if path:
            logger.warning("Không tìm thấy checkpoint khởi tạo %s → huấn luyện từ đầu", path)
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    state = {k: v for k, v in payload["state"].items() if k in own and own[k].shape == v.shape}
    skipped = len(payload["state"]) - len(state)
    model.load_state_dict(state, strict=False)
    logger.info("Khởi tạo từ %s: nạp %d tensor, bỏ qua %d (khác kích thước)", path, len(state), skipped)


@torch.no_grad()
def evaluate_split(model, loader, stage: int, max_samples: int, amp_dtype, device):
    model.eval()
    records, losses = [], []
    seen = 0
    for batch in loader:
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
        with amp_context:
            out = model(batch, stage=stage)
            losses.append(out["loss"].item())
            nb = normalize_batch(batch)
            if stage != 1:
                preds = model.predict_batch(batch)
                for p, a, t, q in zip(preds, nb["answers"], nb["answer_types"], nb["questions"]):
                    records.append({"pred": p["answer"], "gt": a, "answer_type": t, "question": q})
        seen += len(nb["questions"])
        if seen >= max_samples:
            break
    metrics = compute_metrics(records) if records else {}
    metrics["val_loss"] = float(np.mean(losses)) if losses else 0.0
    model.train()
    return metrics, records


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện BoneVQA-Prompt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--dataset", choices=["roco", "pmc_vqa", "vqa_rad", "fracatlas", "fracatlas_content"])
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--ablation", type=str, default="", help="danh sách prompt tắt, vd: visual_prompt,latent_prompt,lens")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--image_mode", choices=["normal", "gray"], default=None)
    parser.add_argument("--roi_control", choices=["real", "shuffled", "generic"], default=None)
    parser.add_argument("--disable_sam", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè seed, dùng khi chạy nhiều lần cùng cấu hình")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage:
        cfg["train"]["stage"] = args.stage
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.init_from is not None:
        cfg["init_from"] = args.init_from
    if args.no_llm:
        cfg["model"]["use_llm"] = False
    if args.disable_sam:
        cfg["model"].setdefault("sam", {})["enabled"] = False
    if args.image_mode or args.roi_control:
        loader = cfg["data"].setdefault("loader_kwargs", {})
        if args.image_mode:
            loader["image_mode"] = args.image_mode
        if args.roi_control:
            loader["roi_control"] = args.roi_control
    ablations = [a.strip() for a in args.ablation.split(",") if a.strip()]
    for a in ablations:
        key = f"use_{a}"
        cfg["model"][key] = False
    if args.seed is not None:
        cfg["seed"] = args.seed
    run_name = args.run_name or cfg.get("run_name", "run")
    if ablations:
        run_name += "_no_" + "_".join(ablations)
    cfg["run_name"] = run_name

    set_seed(cfg.get("seed", 42))
    device = get_device()
    if device.type == "mps" and cfg["model"].get("llm_dtype") in ("bfloat16", "bf16"):
        # Inference is stable in fp16 on MPS, but full fine-tuning can
        # overflow without loss scaling.  Train the small Qwen backbone in
        # fp32 to keep matched-control optimization numerically valid.
        cfg["model"]["llm_dtype"] = "float32"
    stage = cfg["train"]["stage"]
    tcfg = cfg["train"]
    dcfg = cfg["data"]
    amp_dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bfloat16") in ("bfloat16", "bf16") else torch.float16
    use_scaler = tcfg.get("amp", True) and amp_dtype == torch.float16 and device.type == "cuda"

    ckpt_dir = os.path.join(cfg["checkpoint_dir"], run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    fig_dir = cfg.get("figure_dir", "reports/figures")
    logger.info("Run %s | stage %d | dataset %s | thiết bị %s", run_name, stage, dcfg["dataset"], device)

    build_dataset, collate_fn, real = load_data_api()
    loader_kwargs = dcfg.get("loader_kwargs") or {}
    train_split = dcfg.get("train_split", "train")
    train_ds = build_split(build_dataset, dcfg["dataset"], train_split, dcfg.get("max_samples"), real, loader_kwargs)
    val_ds = build_split(build_dataset, dcfg["dataset"], "val", dcfg.get("val_max_samples"), real, loader_kwargs)
    logger.info("Số mẫu train %d, val %d", len(train_ds), len(val_ds))

    vocab = None
    if real:
        try:
            from .data import build_answer_vocab as data_answer_vocab

            vocab = list(data_answer_vocab(dcfg["dataset"], top_k=dcfg.get("answer_vocab_size", 500)))
        except Exception as exc:
            logger.warning("Không dùng được build_answer_vocab của src.data (%s), tự đếm từ dataset", exc)
    if not vocab:
        vocab = build_answer_vocab(collect_answers(train_ds), dcfg.get("answer_vocab_size", 500))
    logger.info("Kích thước answer vocab: %d", len(vocab))

    model = BoneVQAModel(cfg["model"], vocab, device=str(device))
    load_init_weights(model, cfg.get("init_from"))
    model.to(device)
    stats = count_parameters(model)
    logger.info("Tham số: %s", format_params(stats))

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate_fn,
                              num_workers=dcfg.get("num_workers", 0), drop_last=len(train_ds) > tcfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate_fn,
                            num_workers=dcfg.get("num_workers", 0))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.98))
    epochs = tcfg["epochs"]
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * epochs
    warmup = int(tcfg.get("warmup_ratio", 0.05) * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # torch 2.2 does not expose torch.amp.GradScaler; CUDA's legacy scaler
    # is compatible and is a disabled no-op on MPS/CPU.
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history = []
    start_epoch = 0
    best_metric = None
    select_metric = tcfg.get("select_metric", "overall_acc")
    lower_better = select_metric == "val_loss" or stage == 1
    if stage == 1:
        select_metric = "val_loss"
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        extra = model.load_trainable(args.resume)
        history = extra.get("history", [])
        start_epoch = extra.get("epoch", 0)
        best_metric = extra.get("best_metric")
        global_step = extra.get("global_step", start_epoch * steps_per_epoch)
        for _ in range(global_step):
            scheduler.step()
        duong_dan_optim = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "optimizer.pt")
        if os.path.exists(duong_dan_optim):
            trang_thai = torch.load(duong_dan_optim, map_location=device, weights_only=False)
            optimizer.load_state_dict(trang_thai["optimizer"])
            if trang_thai.get("scaler") is not None and use_scaler:
                scaler.load_state_dict(trang_thai["scaler"])
            logger.info("Đã nạp trạng thái optimizer từ %s", duong_dan_optim)
        else:
            logger.warning("Không có %s → moment của AdamW khởi tạo lại, vài chục bước đầu sẽ nhiễu hơn",
                           duong_dan_optim)
        logger.info("Tiếp tục từ epoch %d, bước %d/%d, lr %.2e", start_epoch, global_step, total_steps,
                    scheduler.get_last_lr()[0])

    save_json(cfg, os.path.join(ckpt_dir, "config.json"))
    model.train()
    luu_optimizer = tcfg.get("save_optimizer", False)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        running = {}
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if (tcfg.get("amp", True) and device.type == "cuda") else nullcontext()
            with amp_context:
                out = model(batch, stage=stage)
                loss = out["loss"] / accum
            scaler.scale(loss).backward()
            for k in ("loss", "loss_cls", "loss_lm", "loss_cs", "loss_con", "loss_type"):
                running[k] = running.get(k, 0.0) + float(out[k].detach())
            n_batches += 1
            if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % tcfg.get("log_every", 10) == 0:
                    msg = " ".join(f"{k}={v / n_batches:.4f}" for k, v in running.items())
                    logger.info("Epoch %d bước %d/%d | %s | lr %.2e | %s", epoch + 1, global_step, total_steps, msg,
                                scheduler.get_last_lr()[0], gpu_memory_str())
        epoch_log = {"epoch": epoch + 1, "train_loss": running.get("loss", 0.0) / max(1, n_batches),
                     "time_s": time.time() - t0}
        for k in ("loss_cls", "loss_lm", "loss_cs", "loss_con"):
            epoch_log[f"train_{k}"] = running.get(k, 0.0) / max(1, n_batches)
        if tcfg.get("eval_every_epoch", True):
            metrics, _ = evaluate_split(model, val_loader, stage, tcfg.get("eval_max_samples", 200), amp_dtype, device)
            for k, v in metrics.items():
                epoch_log[f"val_{k}" if not k.startswith("val_") else k] = v
        history.append(epoch_log)
        logger.info("Kết thúc epoch %d: %s | %s", epoch + 1,
                    ", ".join(f"{k}={v:.4f}" for k, v in epoch_log.items() if isinstance(v, float)), gpu_memory_str())
        key = select_metric if select_metric in epoch_log else f"val_{select_metric}"
        cur = epoch_log.get(key)
        improved = cur is not None and (best_metric is None or (cur < best_metric if lower_better else cur > best_metric))
        extra = {"history": history, "epoch": epoch + 1, "best_metric": best_metric, "run_name": run_name,
                 "stage": stage, "global_step": global_step}
        model.save(os.path.join(ckpt_dir, "last.pt"), extra)
        if luu_optimizer:
            torch.save({"optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict() if use_scaler else None,
                        "global_step": global_step, "epoch": epoch + 1},
                       os.path.join(ckpt_dir, "optimizer.pt"))
        if improved or cur is None:
            best_metric = cur
            extra["best_metric"] = best_metric
            model.save(os.path.join(ckpt_dir, "best.pt"), extra)
            logger.info("Checkpoint tốt nhất mới: %s=%s", key, cur)
        save_json(history, os.path.join(ckpt_dir, "history.json"))
        try:
            path = plot_training_curves(history, fig_dir, run_name)
            logger.info("Đã vẽ biểu đồ %s", path)
        except Exception as exc:
            logger.warning("Không vẽ được biểu đồ: %s", exc)
    logger.info("Hoàn tất huấn luyện. %s", gpu_memory_str())
    if torch.cuda.is_available():
        logger.info("VRAM đỉnh: %.2f GB", torch.cuda.max_memory_allocated() / 1024 ** 3)


if __name__ == "__main__":
    main()
