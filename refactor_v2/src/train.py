import argparse
import hashlib
import json
from pathlib import Path
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
from .checkpointing import compatible_state
from .context_controls import ComponentBatchSampler, load_component_lookup
from .endpoints import is_canonical_fracture_presence_question
from .protocol import (assert_disjoint_splits, group_subsample, manifest, records_of,
                       validate_prompt_protocol_config)
from .provenance import validate_detector_proposals
from .utils import (compute_metrics, count_parameters, format_params, get_device, gpu_memory_str, load_config,
                    normalize_answer, plot_training_curves, save_json, set_seed, setup_env_dirs, setup_logger)

setup_env_dirs()
logger = setup_logger()


def seed_training_epoch(base_seed: int, epoch: int):
    """Reproducible epoch-boundary resume, including torch 2.2 MPS.

    MPS get/set_rng_state does not replay dropout in our installed runtime.
    Resetting an explicitly recorded seed does; use only in the revised
    protocol. This does not promise mid-epoch resume or cross-device parity.
    """
    seed = (int(base_seed) + int(epoch)) % (2 ** 32)
    set_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed


def validate_training_runtime(model_cfg: dict, device_type: str, allow_shuffled_smoke=False):
    if model_cfg.get("latent_context_control", "conditioned") == "shuffled" and not allow_shuffled_smoke:
        raise RuntimeError("Shuffled LF context is diagnostic-only until image-group-aware donor batching is verified")
    # In torch 2.2.2/MPS the stochastic recomputation audit gives identical
    # forward losses but different adapter gradients vs no checkpointing.
    # Do not silently certify or run the unsafe optimization path in v3.
    if (device_type == "mps" and int(model_cfg.get("prompt_protocol_version", 0) or 0) >= 3
            and (model_cfg.get("grad_checkpoint", True) or model_cfg.get("llm_grad_checkpoint", True))):
        raise RuntimeError("MPS v3 training requires grad_checkpoint=false and llm_grad_checkpoint=false; "
                           "stochastic checkpoint recomputation failed the gradient parity audit")


def validate_component_smoke_config(cfg):
    seal_path = cfg["data"].get("loader_kwargs", {}).get("seal_path")
    if (not seal_path or json.loads(Path(seal_path).read_text()).get("scope") != "technical_smoke_train_only"
            or not cfg["train"].get("component_batching") or not 1 <= cfg["train"]["epochs"] <= 2):
        raise RuntimeError("Technical component smoke requires scoped seal, component batching and <=2 epochs")


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


def load_data_api(allow_fake: bool = False):
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
        if allow_fake:
            logger.warning("Không import được src.data (%s); --smoke_fake cho phép dataset giả.", exc)
            return None, default_collate, False
        raise RuntimeError("Không import được dataset thật; từ chối fallback im lặng sang dữ liệu giả.") from exc


def build_split(build_dataset, name: str, split: str, max_samples: Optional[int], real: bool,
                loader_kwargs: Optional[dict] = None):
    if not real:
        n = max_samples or 16
        return FakeVQADataset(n if split == "train" else max(4, n // 2), seed=0 if split == "train" else 1)
    candidates = [split]
    resolved_loader = dict(loader_kwargs or {})
    split_manifests = resolved_loader.pop("sealed_records_by_split", None)
    if split_manifests is not None:
        if split not in split_manifests or not split_manifests[split]:
            raise ValueError("Missing sealed manifest for split: " + split)
        resolved_loader["sealed_records_path"] = split_manifests[split]
    last_exc = None
    for sp in candidates:
        try:
            dataset = build_dataset(name, sp, None, **resolved_loader)
            return group_subsample(dataset, max_samples, seed=0 if split == "train" else 1)
        except TypeError:
            if resolved_loader.get("sealed_records_path") is not None:
                raise
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


def load_init_weights(model, path: str, allowed_mismatch_prefixes=(), allow_random_init=False):
    if not path or not os.path.exists(path):
        if allow_random_init:
            logger.warning("Checkpoint không có; cấu hình đã cho phép random initialization.")
            return
        raise FileNotFoundError(f"Checkpoint khởi tạo bắt buộc không tồn tại: {path!r}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    own = model.trainable_state_dict()
    state = compatible_state(own, payload["state"], allowed_mismatch_prefixes)
    model.load_state_dict(state, strict=False)
    logger.info("Khởi tạo strict từ %s: nạp %d tensor", path, len(state))


@torch.no_grad()
def evaluate_split(model, loader, stage: int, max_samples: int | None, amp_dtype, device, component_lookup=None):
    model.eval()
    records, losses = [], []
    loss_sample_counts = []
    seen = 0
    for batch in loader:
        if component_lookup is not None:
            batch["component_ids"] = [component_lookup[image] for image in normalize_batch(batch)["image_ids"]]
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
        with amp_context:
            out = model(batch, stage=stage)
            losses.append(out["loss"].item())
            nb = normalize_batch(batch)
            loss_sample_counts.append(len(nb["questions"]))
            if stage != 1:
                preds = model.predict_batch(batch)
                for p, a, t, q, iid in zip(
                    preds, nb["answers"], nb["answer_types"], nb["questions"],
                    nb.get("image_ids") or [None] * len(preds),
                ):
                    record = {"pred": p["answer"], "gt": a, "answer_type": t,
                              "question": q, "image_id": iid}
                    if "p_yes" in p:
                        record["p_yes"] = p["p_yes"]
                    records.append(record)
        seen += len(nb["questions"])
        if max_samples is not None and seen >= max_samples:
            break
    metrics = compute_metrics(records) if records else {}
    metrics["val_loss"] = float(np.average(losses, weights=loss_sample_counts)) if losses else 0.0
    model.train()
    return metrics, records


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện BoneVQA-Prompt")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                        help="Explicit device for bounded numerical reference audits; default auto")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3])
    parser.add_argument("--dataset", choices=["roco", "pmc_vqa", "vqa_rad", "fracatlas", "fracatlas_content",
                                                "fracatlas_component_clean"])
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--val_max_samples", type=int,
                        help="giới hạn validation cho smoke/debug; không dùng khi báo kết quả sealed")
    parser.add_argument("--eval_max_samples", type=int,
                        help="giới hạn số mẫu đánh giá mỗi epoch cho smoke/debug")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--technical_component_smoke", action="store_true",
                        help="Allow bounded shuffled-P diagnostic only with TRAIN-only technical seal, <=16 train rows and <=2 epochs")
    parser.add_argument("--stop_after_epoch", type=int, default=None,
                        help="Bounded resume audit: stop after saving this epoch without changing the LR horizon")
    parser.add_argument("--init_from", type=str, default=None)
    parser.add_argument("--ablation", type=str, default="", help="danh sách prompt tắt, vd: visual_prompt,latent_prompt,lens")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--no_llm", action="store_true")
    parser.add_argument("--image_mode", choices=["normal", "gray"], default=None)
    parser.add_argument("--roi_control", choices=["real", "shuffled", "generic", "matched", "matched_all"], default=None)
    parser.add_argument("--disable_sam", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè seed, dùng khi chạy nhiều lần cùng cấu hình")
    parser.add_argument("--smoke_fake", action="store_true", help="cho phép dữ liệu giả chỉ cho smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.stage:
        cfg["train"]["stage"] = args.stage
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.max_samples is not None:
        cfg["data"]["max_samples"] = args.max_samples
    if args.val_max_samples is not None:
        cfg["data"]["val_max_samples"] = args.val_max_samples
    if args.eval_max_samples is not None:
        cfg["train"]["eval_max_samples"] = args.eval_max_samples
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
    if int(cfg.get("model", {}).get("prompt_protocol_version", 0) or 0) >= 3:
        # Vary optional-module initialization across experimental seeds while
        # keeping it byte-identical across B/V/L/P/T variants of the same
        # seed.  A fixed module seed across every multi-seed run would
        # understate initialization uncertainty.
        offset = int(cfg["model"].get("module_init_seed_offset", 1729))
        cfg["model"]["module_init_seed"] = int(cfg.get("seed", 42)) + offset
    run_name = args.run_name or cfg.get("run_name", "run")
    if ablations:
        run_name += "_no_" + "_".join(ablations)
    cfg["run_name"] = run_name

    validate_prompt_protocol_config(cfg)
    if cfg.get("recipe_status") == "candidate_not_launch_sealed":
        raise RuntimeError("Candidate recipe is not launch-sealed; finish the protocol/config gate before training")
    if int(cfg.get("model", {}).get("prompt_protocol_version", 0) or 0) >= 3 and (
            cfg["model"].get("use_visual_prompt") or cfg["model"].get("use_lens")):
        proposal_path = cfg["data"].get("loader_kwargs", {}).get("boxes_from")
        proposal_provenance = validate_detector_proposals(proposal_path)
        logger.info("Detector proposal provenance hợp lệ: %s", proposal_provenance["proposal_sha256"])

    set_seed(cfg.get("seed", 42))
    device = torch.device(args.device) if args.device else get_device()
    if args.technical_component_smoke:
        validate_component_smoke_config(cfg)
    validate_training_runtime(cfg["model"], device.type, allow_shuffled_smoke=args.technical_component_smoke)
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

    build_dataset, collate_fn, real = load_data_api(allow_fake=args.smoke_fake)
    loader_kwargs = dcfg.get("loader_kwargs") or {}
    train_split = dcfg.get("train_split", "train")
    train_ds = build_split(build_dataset, dcfg["dataset"], train_split, dcfg.get("max_samples"), real, loader_kwargs)
    val_split = dcfg.get("val_split", "val")
    val_ds = build_split(build_dataset, dcfg["dataset"], val_split, dcfg.get("val_max_samples"), real, loader_kwargs)
    if args.technical_component_smoke and (len(train_ds) > 16 or len(val_ds) > 8):
        raise RuntimeError("Technical component smoke exceeds bounded train/validation row counts")
    if real:
        assert_disjoint_splits(train=train_ds, val=val_ds)
    logger.info("Số mẫu train %d, %s %d", len(train_ds), val_split, len(val_ds))

    vocab = None
    # Vocabulary provenance is exactly the selected clean training rows.
    selected_answers = collect_answers(train_ds)
    if dcfg.get("sealed_answer_vocab"):
        from .data.sealed_manifest import load_sealed_records
        vocab = load_sealed_records(
            dcfg["sealed_answer_vocab"], loader_kwargs["seal_path"], loader_kwargs["project_root"],
        )
        observed = {normalize_answer(answer) for answer in selected_answers}
        if not all(isinstance(answer, str) for answer in vocab) or len(vocab) != len(set(vocab)):
            raise ValueError("Invalid sealed answer vocabulary")
        if set(vocab) != observed:
            raise ValueError("Sealed vocabulary does not match selected training answers")
    else:
        vocab = build_answer_vocab(selected_answers, dcfg.get("answer_vocab_size", 500))
    logger.info("Kích thước answer vocab: %d", len(vocab))

    # Derive the fracture-presence class balance from the selected TRAIN rows
    # only. This prevents the 5:1 negative prior from producing a clinically
    # useless always-negative classifier while keeping val/test sealed.
    fracture_counts = {"yes": 0, "no": 0}
    for record in records_of(train_ds):
        question = str(record.get("question", ""))
        answer = normalize_answer(record.get("answer", ""))
        if (record.get("answer_type") == "CLOSED"
                and is_canonical_fracture_presence_question(question)
                and answer in fracture_counts):
            fracture_counts[answer] += 1
    if fracture_counts["yes"]:
        cfg["model"]["fracture_positive_weight"] = fracture_counts["no"] / fracture_counts["yes"]
    logger.info("Cân bằng fracture từ train: %s; positive weight %.4f", fracture_counts,
                cfg["model"].get("fracture_positive_weight", 1.0))

    model = BoneVQAModel(cfg["model"], vocab, device=str(device))
    load_init_weights(model, cfg.get("init_from"),
                      cfg.get("checkpoint_allow_mismatch_prefixes", ()),
                      cfg.get("allow_random_init", False) or args.smoke_fake)
    model.to(device)
    stats = count_parameters(model)
    logger.info("Tham số: %s", format_params(stats))

    component_sampler, component_lookup, batch_binding = None, None, None
    if tcfg.get("component_batching"):
        batching = tcfg["component_batching"]
        component_lookup, component_digest = load_component_lookup(
            batching["manifest"], batching["seal"], loader_kwargs["project_root"])
        selected_records = records_of(train_ds)
        component_sampler = ComponentBatchSampler(
            [component_lookup[row["image_id"]] for row in selected_records], tcfg["batch_size"], cfg["seed"])
        batch_binding = {**component_sampler.binding(), "component_manifest_sha256": component_digest,
            "selected_rows_sha256": hashlib.sha256(json.dumps(selected_records, sort_keys=True).encode()).hexdigest()}
        train_loader = DataLoader(train_ds, batch_sampler=component_sampler, collate_fn=collate_fn,
                                  num_workers=dcfg.get("num_workers", 0))
    else:
        train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate_fn,
                                  num_workers=dcfg.get("num_workers", 0), drop_last=len(train_ds) > tcfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate_fn,
                            num_workers=dcfg.get("num_workers", 0))

    params = [p for p in model.parameters() if p.requires_grad]
    visual_lora = {
        name: param for name, param in model.named_parameters()
        if name.startswith("visual.") and "lora_" in name.lower() and param.requires_grad
    }
    require_visual_lora_qc = bool(tcfg.get("require_visual_lora_qc", True))
    if require_visual_lora_qc and cfg["model"].get("visual_lora_rank", 0) > 0 and not visual_lora:
        raise RuntimeError("Visual-LoRA QC failed: no trainable visual LoRA parameters were found")
    visual_lora_initial = {name: param.detach().cpu().clone() for name, param in visual_lora.items()}
    visual_lora_gradient_checked = False
    visual_lora_update_checked = False

    # Module-specific causal wiring QC.  Visual-encoder LoRA activity alone
    # does not prove that V is used: the shared global image path also updates
    # those adapters.  Track parameters that are unique to each optional
    # semantic route, and fail closed when an enabled route receives no
    # gradient or no optimizer update.  T has no extra parameters by design;
    # its semantic controls are checked by protocol/unit tests and later by
    # paired causal evaluation.
    prompt_qc_parameters = {}
    if stage == 1 and cfg["model"].get("stage1_caption_loss_weight", 0.0) > 0:
        # Contrastive image loss bypasses these bridge components. Require
        # actual answer-loss gradients AND optimizer updates, not just a
        # decreasing contrastive objective.
        for label, prefix in (("bridge_qformer", "qformer."), ("bridge_prefix", "prefix_proj.")):
            prompt_qc_parameters[label] = {
                name: param for name, param in model.named_parameters()
                if name.startswith(prefix) and param.requires_grad
            }
    if cfg["model"].get("use_visual_prompt"):
        prompt_qc_parameters["V"] = {
            name: param for name, param in model.named_parameters()
            if name == "prompt_mixer.gamma_visual" and param.requires_grad
        }
    if cfg["model"].get("use_lens"):
        prompt_qc_parameters["L"] = {
            name: param for name, param in model.named_parameters()
            if name.startswith("lens_fusion.") and param.requires_grad
        }
    if cfg["model"].get("use_latent_prompt"):
        prompt_qc_parameters["P"] = {
            name: param for name, param in model.named_parameters()
            if (name.startswith("latent.") or name == "fusion.gamma_latent") and param.requires_grad
        }
    missing_prompt_qc = [key for key, values in prompt_qc_parameters.items() if not values]
    if missing_prompt_qc:
        raise RuntimeError(f"Prompt-module QC failed: no unique trainable parameters for {missing_prompt_qc}")
    prompt_qc_initial = {
        module: {name: param.detach().cpu().clone() for name, param in values.items()}
        for module, values in prompt_qc_parameters.items()
    }
    prompt_qc = {
        module: {"nonzero_gradient_tensors": set(), "updated_tensors": set()}
        for module in prompt_qc_parameters
    }
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.98))
    epochs = tcfg["epochs"]
    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * epochs
    epoch_seeded_rng = bool(tcfg.get("epoch_seeded_rng", False))
    rng_policy = "epoch_seed_v1" if epoch_seeded_rng else "stream_rng_v1"
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
        if extra.get("component_batch_binding") != batch_binding:
            raise RuntimeError("Resume component batch/selected-data binding mismatch")
        history = extra.get("history", [])
        start_epoch = extra.get("epoch", 0)
        best_metric = extra.get("best_metric")
        global_step = extra.get("global_step", start_epoch * steps_per_epoch)
        duong_dan_optim = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "optimizer.pt")
        if os.path.exists(duong_dan_optim):
            trang_thai = torch.load(duong_dan_optim, map_location=device, weights_only=False)
            if trang_thai.get("component_batch_binding") != batch_binding:
                raise RuntimeError("Optimizer component batch binding mismatch")
            if epoch_seeded_rng:
                expected_resume = {"rng_policy": rng_policy, "base_seed": cfg["seed"],
                                   "total_steps": total_steps, "steps_per_epoch": steps_per_epoch}
                if any(trang_thai.get(key) != value for key, value in expected_resume.items()):
                    raise RuntimeError("Resume seed policy/seed/LR horizon mismatch; preserve the old run")
            if loader_kwargs.get("sealed_records_by_split") or loader_kwargs.get("sealed_records_path"):
                if (trang_thai.get("global_step") != global_step
                        or trang_thai.get("epoch") != start_epoch):
                    raise RuntimeError("Optimizer epoch/global_step mismatch with resume checkpoint")
                if "scheduler" not in trang_thai:
                    raise RuntimeError("Sealed resume requires saved scheduler state")
            optimizer.load_state_dict(trang_thai["optimizer"])
            if "scheduler" in trang_thai:
                scheduler.load_state_dict(trang_thai["scheduler"])
            else:
                for _ in range(global_step):
                    scheduler.step()
            if "rng" in trang_thai:
                random.setstate(trang_thai["rng"]["python"])
                np.random.set_state(trang_thai["rng"]["numpy"])
                torch.set_rng_state(trang_thai["rng"]["torch"].cpu())
                if device.type == "mps":
                    if "mps" not in trang_thai["rng"]:
                        raise RuntimeError("MPS resume requires MPS RNG state")
                    torch.mps.set_rng_state(trang_thai["rng"]["mps"].cpu())
                if device.type == "cuda":
                    if "cuda" not in trang_thai["rng"]:
                        raise RuntimeError("CUDA resume requires CUDA RNG states")
                    torch.cuda.set_rng_state_all([state.cpu() for state in trang_thai["rng"]["cuda"]])
            elif loader_kwargs.get("sealed_records_by_split") or loader_kwargs.get("sealed_records_path"):
                raise RuntimeError("Sealed resume requires RNG state")
            if trang_thai.get("scaler") is not None and use_scaler:
                scaler.load_state_dict(trang_thai["scaler"])
            logger.info("Đã nạp trạng thái optimizer từ %s", duong_dan_optim)
        else:
            if loader_kwargs.get("sealed_records_by_split") or loader_kwargs.get("sealed_records_path"):
                raise RuntimeError("Sealed training resume requires optimizer.pt; refusing silent AdamW reset")
            logger.warning("Không có %s → moment của AdamW khởi tạo lại, vài chục bước đầu sẽ nhiễu hơn",
                           duong_dan_optim)
        logger.info("Tiếp tục từ epoch %d, bước %d/%d, lr %.2e", start_epoch, global_step, total_steps,
                    scheduler.get_last_lr()[0])

    if args.resume:
        visual_lora_initial = {name: param.detach().cpu().clone() for name, param in visual_lora.items()}
        prompt_qc_initial = {
            module: {name: param.detach().cpu().clone() for name, param in values.items()}
            for module, values in prompt_qc_parameters.items()
        }
    save_json(cfg, os.path.join(ckpt_dir, "config.json"))
    save_json(vocab, os.path.join(ckpt_dir, "answer_vocab.json"))
    save_json({"train": manifest(train_ds), "val": manifest(val_ds)},
              os.path.join(ckpt_dir, "data_manifest.json"))
    model.train()
    luu_optimizer = tcfg.get("save_optimizer", False)
    for epoch in range(start_epoch, epochs):
        # Defensive state reset: evaluate_split also restores train mode.
        model.train()
        if epoch_seeded_rng:
            epoch_seed = seed_training_epoch(cfg["seed"], epoch)
            logger.info("Epoch RNG policy %s: seed %d", rng_policy, epoch_seed)
        if component_sampler is not None:
            component_sampler.set_epoch(epoch)
        t0 = time.time()
        running = {}
        n_batches = 0
        skipped_no_supervision = 0
        optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_loader):
            if component_lookup is not None:
                ids = normalize_batch(batch).get("image_ids")
                if ids is None:
                    raise RuntimeError("Component batching requires image IDs in every batch")
                batch["component_ids"] = [component_lookup[image] for image in ids]
            amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if (tcfg.get("amp", True) and device.type == "cuda") else nullcontext()
            with amp_context:
                out = model(batch, stage=stage)
                loss = out["loss"] / accum
            # Lens alignment is defined only for rows with a detector-derived
            # proposal.  A shuffled image-grouped batch may contain only
            # detector-negative images and therefore no differentiable stage-1
            # objective.  Skip that batch explicitly instead of crashing or
            # inventing a surrogate signal from ROI absence.
            if not loss.requires_grad:
                skipped_no_supervision += 1
                logger.info("Bỏ qua batch %d: không có supervision khả vi cho stage %d", it + 1, stage)
                # Preserve accumulation/optimizer scheduling without creating
                # a semantic training signal.  The zero is connected only so
                # backward is well-defined; module QC below still sees zero
                # gradients and cannot mistake this for a passing batch.
                loss = sum(param.reshape(-1)[0] * 0.0 for param in params)
            scaler.scale(loss).backward()
            for module, values in prompt_qc_parameters.items():
                for name, param in values.items():
                    if (param.grad is not None and torch.isfinite(param.grad).all()
                            and torch.count_nonzero(param.grad).item() > 0):
                        prompt_qc[module]["nonzero_gradient_tensors"].add(name)
            if require_visual_lora_qc and visual_lora and not visual_lora_gradient_checked:
                missing = [name for name, param in visual_lora.items() if param.grad is None]
                non_finite = [
                    name for name, param in visual_lora.items()
                    if param.grad is not None and not torch.isfinite(param.grad).all()
                ]
                nonzero = [
                    name for name, param in visual_lora.items()
                    if param.grad is not None and torch.count_nonzero(param.grad).item() > 0
                ]
                if non_finite:
                    raise RuntimeError(
                        "Visual-LoRA gradient QC failed: "
                        f"missing={len(missing)}/{len(visual_lora)}, "
                        f"non_finite={len(non_finite)}, nonzero={len(nonzero)}"
                    )
                if nonzero:
                    logger.info("Visual-LoRA gradient QC passed: %d/%d tensors have non-zero gradients",
                                len(nonzero), len(visual_lora))
                    visual_lora_gradient_checked = True
            for k in ("loss", "loss_cls", "loss_lm", "loss_cs", "loss_con", "loss_lens_align",
                      "loss_type", "loss_fracture", "visual_contribution_norm",
                      "lens_contribution_norm", "latent_contribution_norm"):
                running[k] = running.get(k, 0.0) + float(out[k].detach())
            n_batches += 1
            if (it + 1) % accum == 0 or (it + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, tcfg.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                for module, values in prompt_qc_parameters.items():
                    for name, param in values.items():
                        if not torch.equal(param.detach().cpu(), prompt_qc_initial[module][name]):
                            prompt_qc[module]["updated_tensors"].add(name)
                if require_visual_lora_qc and visual_lora and not visual_lora_update_checked:
                    changed = [
                        name for name, param in visual_lora.items()
                        if not torch.equal(param.detach().cpu(), visual_lora_initial[name])
                    ]
                    if changed:
                        logger.info("Visual-LoRA update QC passed: optimizer changed %d/%d tensors",
                                    len(changed), len(visual_lora))
                        visual_lora_update_checked = True
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % tcfg.get("log_every", 10) == 0:
                    msg = " ".join(f"{k}={v / n_batches:.4f}" for k, v in running.items())
                    logger.info("Epoch %d bước %d/%d | %s | lr %.2e | %s", epoch + 1, global_step, total_steps, msg,
                                scheduler.get_last_lr()[0], gpu_memory_str())
        epoch_log = {"epoch": epoch + 1, "train_loss": running.get("loss", 0.0) / max(1, n_batches),
                     "time_s": time.time() - t0,
                     "skipped_no_supervision_batches": skipped_no_supervision}
        if n_batches == 0 or n_batches == skipped_no_supervision:
            raise RuntimeError(
                f"Không có batch nào chứa supervision khả vi ở stage {stage}; "
                "kiểm tra proposal coverage/sampler trước khi chạy dài."
            )
        for k in ("loss_cls", "loss_lm", "loss_cs", "loss_con", "loss_lens_align", "loss_fracture",
                  "visual_contribution_norm", "lens_contribution_norm", "latent_contribution_norm"):
            epoch_log[f"train_{k}"] = running.get(k, 0.0) / max(1, n_batches)
        if tcfg.get("eval_every_epoch", True):
            metrics, _ = evaluate_split(model, val_loader, stage, tcfg.get("eval_max_samples", 200), amp_dtype, device,
                                        component_lookup=component_lookup)
            for k, v in metrics.items():
                epoch_log[f"val_{k}" if not k.startswith("val_") else k] = v
        history.append(epoch_log)
        logger.info("Kết thúc epoch %d: %s | %s", epoch + 1,
                    ", ".join(f"{k}={v:.4f}" for k, v in epoch_log.items() if isinstance(v, float)), gpu_memory_str())
        key = select_metric if select_metric in epoch_log else f"val_{select_metric}"
        cur = epoch_log.get(key)
        if loader_kwargs.get("sealed_records_by_split") or loader_kwargs.get("sealed_records_path"):
            if cur is None or not math.isfinite(float(cur)):
                raise RuntimeError("Sealed checkpoint selection metric missing or non-finite: " + key)
        improved = cur is not None and (best_metric is None or (cur < best_metric if lower_better else cur > best_metric))
        if improved:
            best_metric = cur
        extra = {"history": history, "epoch": epoch + 1, "best_metric": best_metric, "run_name": run_name,
                 "stage": stage, "global_step": global_step, "rng_policy": rng_policy,
                 "base_seed": cfg["seed"], "total_steps": total_steps,
                 "component_batch_binding": batch_binding}
        model.save(os.path.join(ckpt_dir, "last.pt"), extra)
        if luu_optimizer:
            rng_state = {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state()}
            if device.type == "mps":
                rng_state["mps"] = torch.mps.get_rng_state()
            if device.type == "cuda":
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            torch.save({"optimizer": optimizer.state_dict(),
                        "component_batch_binding": batch_binding,
                        "rng_policy": rng_policy, "base_seed": cfg["seed"],
                        "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
                        "scheduler": scheduler.state_dict(),
                        "rng": rng_state,
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
        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            logger.info("Dừng có chủ đích sau epoch %d; giữ nguyên horizon %d epoch để kiểm thử resume",
                        epoch + 1, epochs)
            break

    prompt_qc_report = {}
    if require_visual_lora_qc and visual_lora and (
            not visual_lora_gradient_checked or not visual_lora_update_checked):
        raise RuntimeError(
            "Visual-LoRA QC failed at end of run: "
            f"gradient={visual_lora_gradient_checked}, update={visual_lora_update_checked}"
        )
    for module, values in prompt_qc_parameters.items():
        gradients = sorted(prompt_qc[module]["nonzero_gradient_tensors"])
        updates = sorted(prompt_qc[module]["updated_tensors"])
        module_passed = bool(gradients and updates)
        prompt_qc_report[module] = {
            "parameter_tensors": len(values),
            "nonzero_gradient_tensors": gradients,
            "updated_tensors": updates,
            "passed": module_passed,
        }
        if not module_passed:
            save_json(prompt_qc_report, os.path.join(ckpt_dir, "prompt_module_qc.json"))
            raise RuntimeError(
                f"Prompt-module QC failed for {module}: "
                f"nonzero_gradients={len(gradients)}, updates={len(updates)}"
            )
    save_json(prompt_qc_report, os.path.join(ckpt_dir, "prompt_module_qc.json"))
    logger.info("Hoàn tất huấn luyện. %s", gpu_memory_str())
    if torch.cuda.is_available():
        logger.info("VRAM đỉnh: %.2f GB", torch.cuda.max_memory_allocated() / 1024 ** 3)


if __name__ == "__main__":
    main()
