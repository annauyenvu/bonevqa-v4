"""Detector scores for the non-TRAIN images of one v4 fold (boxes.json kept boxes only).  Each image must reproduce the saved boxes."""
import json, sys, time
from pathlib import Path
import torch
from PIL import Image, ImageFile
from experiments.detector640_checkpoint_v3 import load_bound_640
from src.detector_runtime import FractureDetectorRuntime
from src.data.fracatlas_vqa import FRACATLAS_ROOT
ImageFile.LOAD_TRUNCATED_IMAGES = False

R = Path("reports/local_noaux_refinement_v1"); OUT = Path("reports/v4_poststack_v1")
fold = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
f = R / f"fold_{fold}"; saved = json.load(open(f / "boxes.json"))
info = json.load(open(f / "proposal_receipt.json"))["images"]; meta = json.load(open(R / "metadata.json"))[str(fold)]["roles"]
rels = {}
for role, b in meta.items():
    if role != "train":
        for row in json.load(open(b["path"])): rels[row["image_id"]] = row["image"]
rt = object.__new__(FractureDetectorRuntime)
rt.model, _ = load_bound_640(json.load(open(f / "detector/receipt.json"))["checkpoint"])
rt.model.eval(); rt.device = torch.device("cpu"); rt.image_size = 640; rt.score_threshold = .05; rt.max_boxes = 3
out, start, bad = {}, time.time(), 0
for n, (iid, rel) in enumerate(sorted(rels.items())):
    with Image.open(FRACATLAS_ROOT / rel) as im:
        canonical, tensor, scale, px, py = rt._preprocess(im)
        boxes = rt.predict(im)
    with torch.no_grad():
        p = rt.model([tensor])[0]
    scores = [float(s) for s in p["scores"].tolist() if s >= .05][:3]
    if len(boxes) != len(saved.get(iid) or []) or any(abs(a - b) > 1e-3 for x, y in zip(boxes, saved.get(iid) or []) for a, b in zip(x, y)): bad += 1
    area = [(b[2] - b[0]) * (b[3] - b[1]) / (canonical.width * canonical.height) for b in boxes]
    out[iid] = dict(scores=scores, area=area)
    if n % 100 == 99: print(n + 1, len(rels), round(time.time() - start), "mismatch", bad, flush=True)
json.dump(dict(mismatch=bad, images=out), open(OUT / f"scores_fold{fold}.json", "w"))
print("done", len(out), "mismatch", bad, round(time.time() - start), flush=True)
