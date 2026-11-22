"""Run a trained YOLOv8 detector on image folders and store every box (conf >= 0.001).

    venv/bin/python code/predict_yolo.py --weights runs/yolo/fa_only/weights/best.pt --out runs/pred/fa_only \
        --dirs datasets/fa_fold0/images/val datasets/fa_fold0/images/cal datasets/fa_fold0/images/test

Writes <out>_cls{k}.json = {"boxes": {image_stem: [[x1, y1, x2, y2, conf], ...]}} per class.
Images are read by ultralytics exactly as in training (letterbox to 640).
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO, settings  # noqa: E402

settings.update({"sync": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--lists", nargs="*", default=[], help="text files with one image path per line")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    model = YOLO(str(ROOT / a.weights))
    names = model.names
    per_class = {k: {} for k in names}
    start = time.time()
    assert a.dirs or a.lists
    sources = [(d, sorted(p for p in (ROOT / d).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
               for d in a.dirs]
    sources += [(f, sorted({Path(x) for x in (ROOT / f).read_text().split("\n") if x})) for f in a.lists]
    for d, files in sources:
        for i in range(0, len(files), 16):
            chunk = files[i:i + 16]
            results = model.predict([str(p) for p in chunk], imgsz=a.imgsz, conf=0.001, iou=0.7, max_det=100,
                                    device=a.device, verbose=False)
            for p, r in zip(chunk, results):
                for k in names:
                    per_class[k][p.stem] = []
                for box, conf, cls in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(),
                                          r.boxes.cls.cpu().tolist()):
                    per_class[int(cls)][p.stem].append([*box, float(conf)])
        print(d, len(files), round(time.time() - start), "s", flush=True)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, name in names.items():
        Path(f"{out}_cls{k}.json").write_text(json.dumps({"weights": a.weights, "class": name, "floor": 0.001,
                                                          "boxes": per_class[k]}))
    print("done", {names[k]: sum(len(v) for v in per_class[k].values()) for k in names})


if __name__ == "__main__":
    main()
