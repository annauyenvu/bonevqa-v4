"""Continue each fold's TRAIN-only detector for 4 more epochs (2 -> 6) with cosine LR from 0.002.  Saves every epoch in a new folder; sealed files stay untouched."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from experiments.detector640_checkpoint_v3 import load_bound_640
from scripts.train_detector_v2 import BoPhatHienDataset, PositiveAnchoredBatchSampler, collate
from src.detector_protocol import verify_detector_annotation_parity
from src.utils import set_seed

R = Path("reports/local_noaux_refinement_v1"); O = Path("reports/v4_detector_ext_v1")
fold = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2])); EPOCHS = 4
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42  # 42 is the first run, in reports/v4_detector_ext_v1/fold_k
BASE = SEED if SEED == 42 else SEED + 100
meta = json.load(open(R / "metadata.json"))[str(fold)]["roles"]["train"]["path"]
ids = {r["image_id"] for r in json.load(open(meta))}
model, payload = load_bound_640(json.load(open(R / f"fold_{fold}/detector/receipt.json"))["checkpoint"])
assert set(payload["training_image_ids"]) == ids
ds = BoPhatHienDataset("train", canh_dai=640, tang_cuong=True, allowed_ids=ids); verify_detector_annotation_parity(ds, meta)
set_seed(BASE + fold); sampler = PositiveAnchoredBatchSampler(ds, 4, BASE + fold)
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=0)
params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.SGD(params, lr=.002, momentum=.9, weight_decay=.0005)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS); dest = (O if SEED == 42 else O / f"seed{SEED}") / f"fold_{fold}"; dest.mkdir(exist_ok=True); start = time.time(); up = 0
for e in range(EPOCHS):
    sampler.epoch = 2 + e; model.train(); L = []
    for images, targets, _ in loader:
        loss = sum(model(list(images), list(targets)).values()); assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 5); opt.step(); up += 1; L.append(float(loss.detach()))
        if up % 100 == 0: print("epoch", 3 + e, "up", up, "loss", round(float(np.mean(L[-100:])), 4), "s", round(time.time() - start), flush=True)
    sched.step()
    torch.save({**{k: v for k, v in payload.items() if k != "state"}, "state": model.state_dict(), "epoch": 3 + e, "optimizer_steps": payload["optimizer_steps"] + up}, dest / f"ep{3 + e}.pt")
    print("saved", 3 + e, round(float(np.mean(L)), 4), round(time.time() - start), flush=True)
print("done", flush=True)
