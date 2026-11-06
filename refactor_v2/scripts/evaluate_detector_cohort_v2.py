"""Evaluate a sealed detector on an explicit image-id cohort."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_detector_module():
    path = Path(__file__).with_name("train_detector_v2.py")
    spec = importlib.util.spec_from_file_location("detector_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return {str(row["image_id"]) for row in payload}
        return {str(value) for value in payload}
    if isinstance(payload, dict):
        values = payload.get("image_ids") or payload.get("ids")
        if values is not None:
            return {str(value) for value in values}
    raise ValueError("ids_from phải là list ID, list record có image_id, hoặc dict có image_ids/ids")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ids_from", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=6)
    # Dynamic module loading is not picklable through macOS spawn workers.
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=10)
    args = parser.parse_args()

    module = load_detector_module()
    torch.set_num_threads(args.torch_threads)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = module.tao_mo_hinh(payload["architecture"], weights=None)
    model.load_state_dict(payload["state"], strict=True)
    model.to("cpu")
    cohort_ids = read_ids(args.ids_from)
    dataset = module.BoPhatHienDataset("test", canh_dai=payload["canh_dai"], allowed_ids=cohort_ids)
    found = {item[0] for item in dataset.muc}
    if found != cohort_ids:
        missing = sorted(cohort_ids - found)
        raise RuntimeError(f"Thiếu {len(missing)} cohort image trong FracAtlas: {missing[:20]}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=module.collate,
                        num_workers=args.num_workers)
    metrics = module.danh_gia(model, loader, torch.device("cpu"), nguong=args.score)
    output = {
        "checkpoint": str(args.checkpoint),
        "ids_from": str(args.ids_from),
        "score_threshold": args.score,
        "n_requested": len(cohort_ids),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
