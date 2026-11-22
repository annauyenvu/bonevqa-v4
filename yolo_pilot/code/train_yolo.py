"""Train one YOLOv8 detector.  Detection only (no segmentation head).

    venv/bin/python code/train_yolo.py --name fa_only --data datasets/fa_fold0/data_fa.yaml --epochs 60 --device mps

Weights are selected on the val split by ultralytics (best.pt); test is never read here.
"""
import argparse
import os
from pathlib import Path

ROOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO, settings  # noqa: E402

settings.update({"sync": False, "runs_dir": str(ROOT / "runs"), "weights_dir": str(ROOT / "weights"),
                 "datasets_dir": str(ROOT / "datasets")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="none", choices=["none", "ram"])
    a = ap.parse_args()
    (ROOT / "weights").mkdir(exist_ok=True)
    os.chdir(ROOT / "weights")  # pretrained weights download here, not into the cwd of the caller
    model = YOLO(a.model)
    model.train(data=str(ROOT / a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                workers=a.workers, seed=a.seed, deterministic=True, amp=False, fraction=a.fraction,
                patience=a.patience, project=str(ROOT / "runs/yolo"), name=a.name, exist_ok=False,
                plots=False, val=True, verbose=True, cache=("ram" if a.cache == "ram" else False))


if __name__ == "__main__":
    main()
