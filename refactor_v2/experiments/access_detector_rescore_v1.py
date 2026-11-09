"""R3.3 support - re-run each fold's sealed detector on that fold's outer-test images to recover the proposal
scores (boxes.json stores coordinates only). Reference-only: imports the project's detector code read-only,
writes only to <out.json>. The kept boxes are checked against the stored boxes.json before any score is used.

Needs the fold detector checkpoints and the FracAtlas images (not in this repository); its output is stored in
reports/access_v1/review3/detector_scores.json.
Usage: python experiments/access_detector_rescore_v1.py <project root> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); R2 = ROOT / 'refactor_v2'
sys.path.insert(0, str(R2))
from PIL import Image  # noqa: E402
from src.detector_runtime import FractureDetectorRuntime  # noqa: E402

torch.set_num_threads(8)
S = R2 / 'reports/local_noaux_refinement_v1'
FRAC = ROOT / 'data'
PFX = 'BONE_VQA\\BaoYen_work\\data\\fracatlas\\FracAtlas\\'


def ckpt(f): return R2 / 'detector640_fold0_bounded_quality_v3_1/last.pt' if f == '0' else S / f'fold_{f}/detector/LAST.pt'


def main():
    out = {}
    for f in '01234':
        det = FractureDetectorRuntime(ckpt(f), 'cpu', score_threshold=0.0, max_boxes=100)
        payload = torch.load(ckpt(f), map_location='cpu', weights_only=False)
        tr = payload.get('detector_transform') or {}
        if tr:  # same transform the detector was trained with
            det.model.transform.min_size = tuple(tr['min_size']); det.model.transform.max_size = tr['max_size']
        stored = json.load(open(S / f'fold_{f}/boxes.json'))
        rec = json.load(open(S / f'fold_{f}/proposal_receipt.json'))['images']
        test = sorted(k for k, v in rec.items() if v['role'] == 'test')
        qa = {r['image_id']: r for r in json.load(open(ROOT / f'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_{f}/vqa_test.json'))}
        res = {}; match = 0
        for im in test:
            img = Image.open(FRAC / (PFX + qa[im]['image'].replace('/', '\\')))
            canonical, tensor, scale, px, py = det._preprocess(img)
            with torch.no_grad(): p = det.model([tensor])[0]
            sc = p['scores'].tolist(); bx = p['boxes'].tolist(); kept, ks = [], []
            for b, s in zip(bx, sc):
                if s < 0.05 or len(kept) >= 3: continue
                x1 = max(0., min(canonical.width, (b[0] - px) / scale)); y1 = max(0., min(canonical.height, (b[1] - py) / scale))
                x2 = max(0., min(canonical.width, (b[2] - px) / scale)); y2 = max(0., min(canonical.height, (b[3] - py) / scale))
                if x2 > x1 and y2 > y1: kept.append([x1, y1, x2, y2]); ks.append(s)
            ok = len(kept) == len(stored[im]) and (not kept or np.allclose(np.array(kept), np.array(stored[im]), atol=2.0))
            match += ok
            res[im] = dict(kept_scores=ks, top_score_any=(sc[0] if sc else 0.0), n_kept=len(kept), matches_stored=bool(ok))
        out[f] = dict(n_test=len(test), n_match_stored=match, transform=tr, images=res)
        print(f, len(test), 'match stored boxes:', match, flush=True)
    OUT.write_text(json.dumps(out))


if __name__ == '__main__':
    main()
