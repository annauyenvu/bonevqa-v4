import argparse
import hashlib
import json
from pathlib import Path
from contextlib import nullcontext
import glob
import os

import torch
from torch.utils.data import DataLoader

from .calibration import load_calibration_artifact
from .models.bonevqa import BoneVQAModel, normalize_batch
from .provenance import validate_detector_proposals
from .train import build_split, load_data_api
from .utils import (compute_metrics, get_device, load_json, plot_ablation_bar, plot_confusion_yes_no, save_json,
                    set_seed, setup_env_dirs, setup_logger)

setup_env_dirs()
logger = setup_logger()


def print_table(name: str, m: dict):
    rows = [("Độ chính xác câu đóng", m.get("closed_acc", 0.0), m.get("n_closed", 0)),
            ("Độ chính xác câu mở (khớp chính xác)", m.get("open_acc", 0.0), m.get("n_open", 0)),
            ("Recall câu mở (LLaVA-Med)", m.get("open_recall", 0.0), m.get("n_open", 0)),
            ("BLEU-1 câu mở", m.get("bleu1", 0.0), m.get("n_open", 0)),
            ("Độ chính xác chung", m.get("overall_acc", 0.0), m.get("n", 0))]
    print(f"\n=== Kết quả đánh giá: {name} ===")
    print(f"{'Chỉ số':<42}{'Giá trị':>10}{'Số mẫu':>10}")
    for label, val, n in rows:
        print(f"{label:<42}{val * 100:>9.2f}%{n:>10}")


def evaluate_model(model, loader, max_samples, device, amp_dtype=torch.bfloat16):
    records = []
    seen = 0
    for batch in loader:
        nb = normalize_batch(batch)
        # torch 2.2 accepts CUDA autocast only; MPS runs the model directly.
        amp_context = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.type == "cuda" else nullcontext()
        with amp_context:
            preds = model.predict_batch(batch, tra_chi_tiet=True)
        for p, a, t, q, iid in zip(preds, nb["answers"], nb["answer_types"], nb["questions"], nb.get("image_ids") or [None] * len(preds)):
            record = {"image_id": iid, "question": q, "gt": a, "pred": p["answer"], "answer_type": t,
                      "pred_type": p["answer_type"], "confidence": p["confidence"]}
            if "p_yes" in p:
                record["p_yes"] = p["p_yes"]
            if "p_yes_raw" in p:
                record["p_yes_raw"] = p["p_yes_raw"]
            records.append(record)
        seen += len(preds)
        if seen >= max_samples:
            break
    return compute_metrics(records, fracture_threshold=float(model.cfg.get("fracture_yes_threshold", 0.5))), records


TAG_LABELS = {
    "full": "Mô hình chính",
    "abl_du3": "Đủ 3 prompt",
    "abl3_du3": "Đủ 3 prompt",
    "abl3_khong_visual": "Không visual prompt",
    "abl3_khong_lens": "Không lens",
    "abl3_khong_latent": "Không latent prompt",
    "abl3_khong_prompt": "Không prompt nào",
    "closed_bang_llm": "Câu đóng bằng LLM",
    "abl_khong_visual": "Không visual prompt",
    "khong_visual_prompt": "Không visual prompt",
    "abl_khong_lens": "Không lens",
    "khong_lens": "Không lens",
    "abl_khong_latent": "Không latent prompt",
    "khong_latent_prompt": "Không latent prompt",
    "abl_khong_prompt": "Không prompt nào",
    "khong_prompt": "Không prompt nào",
}


def compare_results(report_dir: str, fig_dir: str, dataset: str):
    files = sorted(glob.glob(os.path.join(report_dir, f"results_{dataset}_*.json")))
    tags = {os.path.splitext(os.path.basename(f))[0].replace(f"results_{dataset}_", ""): f for f in files}
    tien_to = "abl3_" if any(t.startswith("abl3_") for t in tags) else "abl_"
    ablation_tags = {t: f for t, f in tags.items() if t.startswith(tien_to)}
    if len(ablation_tags) >= 2:
        tags = ablation_tags
    if len(tags) < 2:
        return None
    results = {}
    for tag, f in tags.items():
        data = load_json(f)
        results[TAG_LABELS.get(tag, tag)] = data["metrics"]
    order = list(TAG_LABELS.values())
    results = dict(sorted(results.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
    path = plot_ablation_bar(results, os.path.join(fig_dir, f"ablation_{dataset}.png"), title=f"Ablation prompt trên {dataset}: huấn luyện lại từng cấu hình, đánh giá trên cùng tập test")
    logger.info("Đã vẽ biểu đồ so sánh %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Đánh giá BoneVQA-Prompt trên tập test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--technical_smoke", action="store_true",
                        help="TRAIN-only derivative seal required; never scientific performance evidence")
    parser.add_argument("--dataset", choices=["roco", "pmc_vqa", "vqa_rad", "fracatlas", "fracatlas_content",
                                                "fracatlas_component_clean"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sealed_records_path", default=None)
    parser.add_argument("--data_seal", default=None)
    parser.add_argument("--project_root", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument("--figure_dir", default="reports/figures")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--ablation", default="",
        help=("gate-zero prompt lúc suy luận mà không đổi architecture/checkpoint: "
              "visual_prompt,lens,latent_prompt; textual_prompt đổi về raw question"),
    )
    parser.add_argument("--closed_via_llm", action="store_true", help="trả lời câu đóng bằng LLM thay vì head phân lớp")
    parser.add_argument("--tta_kinds", default="", help="TTA: danh sách kiểu visual prompt, vd: contour,box,circle,mask")
    parser.add_argument("--closed_llm_weight", type=float, default=None, help="trọng số phiếu của LLM cho câu yes/no (0..1)")
    parser.add_argument("--fracture_yes_threshold", type=float, default=None,
                        help="ngưỡng fracture đã chọn trên validation và khóa trước test")
    parser.add_argument("--fracture_calibration", type=str, default=None,
                        help="artifact v3 chứa Platt scaling + ngưỡng khóa trên calibration split")
    parser.add_argument("--box_bo_phat_hien", default=None,
                        help="dùng box do bộ phát hiện đề xuất, truyền đường dẫn JSON đã sinh sẵn")
    parser.add_argument("--che_do_ho_tro", action="store_true",
                        help="chế độ có bác sĩ khoanh vùng: bỏ mask thật nhưng GIỮ bounding box làm prompt cho SAM")
    parser.add_argument("--khong_dung_region", action="store_true",
                        help="bỏ tên vùng giải phẫu khỏi textual prompt (tránh rò rỉ đáp án câu hỏi về vùng)")
    parser.add_argument("--khong_dung_mask_that", action="store_true",
                        help="bỏ mask VÀ bounding box ground-truth, bắt SAM tự dò như khi triển khai thật")
    parser.add_argument("--image_mode", choices=["normal", "gray"], default="normal")
    parser.add_argument("--roi_control", choices=["real", "shuffled", "generic", "matched", "matched_all"], default="real")
    parser.add_argument("--disable_sam", action="store_true", help="không dùng SAM fallback khi ROI trống")
    args = parser.parse_args()
    if args.sealed_records_path and args.che_do_ho_tro:
        raise ValueError("Automatic sealed evaluation forbids annotation-assisted ROI")

    set_seed(42)
    device = torch.device(args.device) if args.device else get_device()
    if args.technical_smoke and (not args.data_seal or json.loads(Path(args.data_seal).read_text()).get("scope") != "technical_smoke_train_only"):
        raise ValueError("Technical evaluation requires an explicitly scoped TRAIN-only derivative seal")
    override = {}
    for a in [x.strip() for x in args.ablation.split(",") if x.strip()]:
        gate_zero = {
            "visual_prompt": "visual_force_zero",
            "lens": "lens_force_zero",
            "latent_prompt": "latent_force_zero",
        }
        if a in gate_zero:
            override[gate_zero[a]] = True
        elif a == "textual_prompt":
            # T has no conditional trainable architecture; raw question is its
            # correct negative control and remains checkpoint-compatible.
            override["use_textual_prompt"] = False
        else:
            raise ValueError(f"Ablation không được hỗ trợ: {a!r}")
    if args.closed_via_llm:
        override["closed_via_llm"] = True
    if args.tta_kinds:
        override["tta_kinds"] = [k.strip() for k in args.tta_kinds.split(",") if k.strip()]
    if args.closed_llm_weight is not None:
        override["closed_llm_weight"] = args.closed_llm_weight
    if args.fracture_yes_threshold is not None and args.fracture_calibration is not None:
        raise ValueError("Dùng --fracture_calibration hoặc --fracture_yes_threshold, không dùng đồng thời")
    calibration_provenance = None
    if args.fracture_calibration is not None:
        calibration_provenance = load_calibration_artifact(args.fracture_calibration,
            expected_checkpoint=args.checkpoint, expected_manifest=args.sealed_records_path,
            allow_technical_smoke=args.technical_smoke)
        override["fracture_platt_slope"] = calibration_provenance["platt_slope"]
        override["fracture_platt_intercept"] = calibration_provenance["platt_intercept"]
        override["fracture_yes_threshold"] = calibration_provenance["fracture_yes_threshold"]
    if args.fracture_yes_threshold is not None:
        if not 0.0 <= args.fracture_yes_threshold <= 1.0:
            raise ValueError("--fracture_yes_threshold phải nằm trong [0, 1]")
        override["fracture_yes_threshold"] = args.fracture_yes_threshold
    if args.khong_dung_region:
        override["use_region_in_prompt"] = False
    if args.disable_sam:
        override["sam"] = {"enabled": False}
    model = BoneVQAModel.load(args.checkpoint, device=str(device), cfg_override=override or None)
    build_dataset, collate_fn, real = load_data_api()
    loader_kwargs = {}
    if args.dataset == "fracatlas_component_clean":
        loader_kwargs = {"with_masks": False, "with_boxes": False}
        needs_roi = model.cfg.get("use_visual_prompt") or model.cfg.get("use_lens")
        if needs_roi and not args.box_bo_phat_hien:
            raise RuntimeError("Đánh giá V/L trên clean split bắt buộc có --box_bo_phat_hien; từ chối dùng ROI annotation.")
    if args.khong_dung_mask_that:
        loader_kwargs = {"with_masks": False, "with_boxes": False}
    elif args.che_do_ho_tro:
        loader_kwargs = {"with_masks": False, "with_boxes": True}
    if args.box_bo_phat_hien:
        detector_provenance = validate_detector_proposals(args.box_bo_phat_hien)
        # Production parity: detector proposals become prompt masks. Do not
        # silently mix them with lesion annotations from the benchmark.
        loader_kwargs = {"with_masks": False, "with_boxes": False,
                         "boxes_from": args.box_bo_phat_hien, "boxes_as_masks": True}
    if args.dataset in ("fracatlas", "fracatlas_component_clean"):
        loader_kwargs.update({"image_mode": args.image_mode, "roi_control": args.roi_control})
    if args.sealed_records_path:
        if not args.data_seal or not args.project_root:
            raise ValueError("Sealed evaluation requires --data_seal and --project_root")
        loader_kwargs.update({"sealed_records_path": args.sealed_records_path,
                              "seal_path": args.data_seal, "project_root": args.project_root})
    ds = build_split(build_dataset, args.dataset, args.split, args.max_samples, real, loader_kwargs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info("Đánh giá %s (%s) với %d mẫu", args.dataset, args.split, len(ds))
    metrics, records = evaluate_model(model, loader, args.max_samples or len(ds), device)
    run_name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    tag = args.tag or (run_name + ("_no_" + args.ablation.replace(",", "_") if args.ablation else ""))
    print_table(tag, metrics)
    out_path = os.path.join(args.report_dir, f"results_{args.dataset}_{tag}.json")
    save_json({"run_name": tag, "dataset": args.dataset, "split": args.split, "checkpoint": args.checkpoint,
               "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
               "evaluation_scope": "technical_smoke_train_only" if args.technical_smoke else "development_informed",
               "fracture_score_space": "fracture_head_sigmoid" if model.cfg.get("use_fracture_head") else "closed_head_yes_no",
               "sealed_data_provenance": ({
                   "manifest": str(Path(args.sealed_records_path).resolve()),
                   "manifest_sha256": hashlib.sha256(Path(args.sealed_records_path).read_bytes()).hexdigest(),
                   "seal": str(Path(args.data_seal).resolve()),
                   "seal_sha256": hashlib.sha256(Path(args.data_seal).read_bytes()).hexdigest(),
               } if args.sealed_records_path else None),
               "fracture_yes_threshold": model.cfg.get("fracture_yes_threshold", 0.5),
               "fracture_calibration_provenance": calibration_provenance,
               "detector_proposal_provenance": detector_provenance if args.box_bo_phat_hien else None,
               "metrics": metrics, "records": records}, out_path)
    logger.info("Đã lưu kết quả vào %s", out_path)
    cm_path = plot_confusion_yes_no(records, os.path.join(args.figure_dir, f"confusion_{args.dataset}_{tag}.png"))
    logger.info("Đã vẽ ma trận nhầm lẫn %s", cm_path)
    compare_results(args.report_dir, args.figure_dir, args.dataset)


if __name__ == "__main__":
    main()
