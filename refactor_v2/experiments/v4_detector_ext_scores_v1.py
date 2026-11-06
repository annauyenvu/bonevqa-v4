"""Score, box and image size of one extended detector checkpoint on the non-TRAIN images of its fold."""
import json, sys, time
from pathlib import Path
import torch
from PIL import Image
from experiments.detector640_checkpoint_v3 import load_bound_640
from src.detector_runtime import FractureDetectorRuntime
from src.data.fracatlas_vqa import FRACATLAS_ROOT

R = Path("reports/local_noaux_refinement_v1"); O = Path("reports/v4_detector_ext_v1")
fold, ep = int(sys.argv[1]), int(sys.argv[2]); torch.set_num_threads(int(sys.argv[3]))
if len(sys.argv) > 4: O = O / f"seed{sys.argv[4]}"
out_path = O / f"fold_{fold}/scores_ep{ep}.json"
if out_path.exists(): sys.exit(0)
ck = json.load(open(R / f"fold_{fold}/detector/receipt.json"))["checkpoint"] if ep == 2 else O / f"fold_{fold}/ep{ep}.pt"
rels = {}
for role, b in json.load(open(R / "metadata.json"))[str(fold)]["roles"].items():
    if role != "train":
        for row in json.load(open(b["path"])): rels[row["image_id"]] = row["image"]
rt = object.__new__(FractureDetectorRuntime); rt.model, _ = load_bound_640(ck); rt.model.eval(); rt.device = torch.device("cpu"); rt.image_size = 640; rt.score_threshold = .05; rt.max_boxes = 3
out = {}
for iid, rel in sorted(rels.items()):
    with Image.open(FRACATLAS_ROOT / rel) as im:
        canonical, tensor, scale, px, py = rt._preprocess(im)
    with torch.no_grad(): p = rt.model([tensor])[0]
    S, B = [], []
    for box, s in zip(p["boxes"].tolist(), p["scores"].tolist()):
        if s < .05 or len(S) == 3: continue
        S.append(float(s)); B.append([(box[0] - px) / scale, (box[1] - py) / scale, (box[2] - px) / scale, (box[3] - py) / scale])
    out[iid] = dict(scores=S, boxes=B, size=list(canonical.size))
json.dump(out, open(out_path, "w"))
print("done", fold, ep, len(out), flush=True)
