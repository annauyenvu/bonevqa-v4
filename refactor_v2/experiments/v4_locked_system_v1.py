"""The locked fracture scorer as one function, from saved parameters only.  main: fresh detector + stack_params must reproduce the exported scores."""
import json, math, random
from pathlib import Path
import torch
from PIL import Image
from experiments.detector640_checkpoint_v3 import load_bound_640
from src.detector_runtime import FractureDetectorRuntime
from src.data.fracatlas_vqa import FRACATLAS_ROOT
O = Path("reports/v4_detector_ext_v1"); V4 = Path("reports/local_noaux_refinement_v1"); LOW = math.log(.05 / .95) - 1


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))


def detect(fold):
    rt = object.__new__(FractureDetectorRuntime); rt.model, _ = load_bound_640(O / f"fold_{fold}/ep6.pt"); rt.model.eval(); rt.device = torch.device("cpu"); rt.image_size = 640; rt.score_threshold = .05; rt.max_boxes = 3
    def run(image):
        _, tensor, *_ = rt._preprocess(image)
        with torch.no_grad(): return [float(s) for s in rt.model([tensor])[0]["scores"].tolist() if s >= .05][:3]
    return run


def score(params, p_B, p_Bh, scores):
    f = dict(B=logit(p_B), Bh=logit(p_Bh), flag=float(bool(scores)), s1=logit(scores[0]) if scores else LOW)
    return params["weights"][0] + sum(w * (f[c] - params["mean_sd"][c][0]) / params["mean_sd"][c][1] for c, w in zip(params["columns"], params["weights"][1:]))


if __name__ == "__main__":
    P = json.load(open(O / "stack_params.json")); E = json.load(open(O / "final_test_answers.json")); CAN = "Is there a fracture in this X-ray?"; rnd = random.Random(0); worst = 0; n = 0
    meta = json.load(open(V4 / "metadata.json"))
    for k in range(5):
        run = detect(k); rel = {r["image_id"]: r["image"] for r in json.load(open(meta[str(k)]["roles"]["test"]["path"]))}
        rows = {a: {r["image_id"]: r["p_yes"] for r in json.load(open(V4 / f"fold_{k}/{d}/test_predictions.json"))["records"] if r["question"] == CAN} for a, d in (("B", "B"), ("Bh", "B/head_refit"))}
        saved = {r["image_id"]: r["s"] for r in E[str(k)] if r["s"] is not None}
        for i in rnd.sample(sorted(saved), 30):
            with Image.open(FRACATLAS_ROOT / rel[i]) as im: sc = run(im)
            worst = max(worst, abs(score(P[str(k)], rows["B"][i], rows["Bh"][i], sc) - saved[i])); n += 1
        print(k, n, worst, flush=True)
    assert worst < 1e-5, worst; print("replay passed", n, "images, worst score difference", worst)
