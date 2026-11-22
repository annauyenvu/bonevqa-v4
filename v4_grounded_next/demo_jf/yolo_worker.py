"""YOLOv8 worker for the JF demo, run with yolo_pilot/venv/bin/python (ultralytics is not in the sealed V4 venv).

Protocol: one JSON object per line on stdin {"path": "<canonical image file>"}; one JSON line back
{"boxes": {"0": [[x1, y1, x2, y2, score], ...], "1": [...]}, "ms": <int>} sorted by score, every box with score >= 0.001.
Same prediction call as yolo_pilot/code/predict_yolo.py (imgsz 640, conf 0.001, iou 0.7, max_det 100, CPU), so the
thresholds chosen on the stored predictions apply unchanged.
"""
import json
import os
import sys
import time
from pathlib import Path

PILOT = Path("/Users/uyen/Downloads/BONEVQA/yolo_pilot")
os.environ.setdefault("YOLO_CONFIG_DIR", str(PILOT / ".ultralytics"))
from ultralytics import YOLO, settings  # noqa: E402

settings.update({"sync": False})


def main(weights):
    model = YOLO(weights)
    print(json.dumps({"ready": True, "names": model.names}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        t0 = time.perf_counter()
        r = model.predict(req["path"], imgsz=640, conf=0.001, iou=0.7, max_det=100, device="cpu", verbose=False)[0]
        out = {"0": [], "1": []}
        for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            out[str(int(cls))].append([*box, float(conf)])
        for k in out:
            out[k].sort(key=lambda b: -b[4])
        print(json.dumps({"boxes": out, "ms": round((time.perf_counter() - t0) * 1000)}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
