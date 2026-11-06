"""Kiểm định ngoài GRAZPEDWRI-DX trên checkpoint niêm phong (reports/access_v1/PROTOCOL.md, Phần D). Không huấn luyện lại.
Chạy: python -m experiments.access_external_v1 sample | run <fold> <arm> | analyze
"""
import csv, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image
from experiments.thesis_v4_checks_v1 import read, SEALED, ARMDIR, CAN, load_detector
from experiments.access_stats_v1_core import ap, auroc, ece
from src.preprocessing import canonicalize_raster
D = Path('/Users/uyen/Downloads/BONEVQA/.evaluation_runtime/data/external_grazpedwri'); OUT = SEALED.parent / 'access_v1/external'; SEED = 20261103


def rows(): return list(csv.DictReader(open(D / 'raw/dataset.csv', encoding='utf-8-sig')))
def paths(): return {p.stem: p for p in (D / 'images').rglob('*.png')}


def load(p):
    """Ảnh gốc có thể là PNG 16 bit; đưa về 8 bit theo toàn dải trước tiền xử lý thường lệ."""
    im = Image.open(p)
    if im.mode in ('I;16', 'I;16B', 'I'):
        a = np.asarray(im, dtype=np.float64); a = a / (65535.0 if a.max() > 255 else 255.0) * 255.0
        im = Image.fromarray(a.astype(np.uint8))
    return canonicalize_raster(im)


def sample():
    """Chỉ dùng filestem và patient_id, không đọc nhãn."""
    p = paths(); by = defaultdict(list)
    for r in rows():
        if r['filestem'] in p: by[r['patient_id']].append(r['filestem'])
    rng = random.Random(SEED); picked = sorted(rng.choice(sorted(v)) for v in (by[k] for k in sorted(by)))
    picked = sorted(rng.sample(picked, 2000)); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'sample.json').write_text(json.dumps(dict(seed=SEED, patients=len(by), images_in_csv=len(rows()), images_on_disk=len(p), ids=picked)))
    print(len(by), len(p), len(picked))


def run(fold, arm):
    from experiments.vnext_model_v1 import VNextModel
    from src.data.fracatlas_vqa import FracAtlasVQADataset as DS
    dest = OUT / f'fold_{fold}_{arm}.json'
    if dest.exists(): return
    torch.set_num_threads(2); ids = read(OUT / 'sample.json')['ids']; p = paths(); folder = SEALED / f'fold_{fold}'
    model = VNextModel.load(str(folder / ARMDIR[arm] / 'LAST.pt'), device='cpu').eval(); det = load_detector(fold) if arm == 'V4' else None; res = {}; nb = {}; start = time.time()
    for off in range(0, len(ids), 8):
        b = ids[off:off + 8]; ims = [load(p[i]) for i in b]; qs = [CAN] * len(b)
        with torch.no_grad():
            if det is None: out = model.predict_batch({'images': ims, 'questions': qs})
            else:
                bx = [det.predict(im) or [] for im in ims]; nb.update({i: len(x) for i, x in zip(b, bx)})
                out = model.predict_batch({'images': ims, 'questions': qs, 'boxes': [x or None for x in bx], 'masks': [DS._boxes_to_masks(x, im.size) if x else [] for x, im in zip(bx, ims)]})
        res.update({i: float(o['p_yes']) for i, o in zip(b, out)}); print(off, len(ids), round(time.time() - start), flush=True)
    t = dest.with_suffix('.tmp'); t.write_text(json.dumps(dict(fold=fold, arm=arm, seconds=time.time() - start, p_yes=res, n_boxes=nb))); t.rename(dest)


def analyze():
    ids = read(OUT / 'sample.json')['ids']; meta = {r['filestem']: r for r in rows()}; m = [meta[i] for i in ids]
    y = np.array([r['fracture_visible'].strip() in ('1', '1.0') for r in m]); metal = np.array([r['metal'].strip() in ('1', '1.0') for r in m]); F = '01234'
    P = {a: np.array([[read(OUT / f'fold_{f}_{a}.json')['p_yes'][i] for i in ids] for f in F]) for a in ('B', 'V4')}; N = np.array([[read(OUT / f'fold_{f}_V4.json')['n_boxes'][i] for i in ids] for f in F])
    rng = np.random.default_rng(SEED); idx = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
    def block(mask):
        yy = y[mask]; out = dict(n=int(mask.sum()), prevalence=float(yy.mean()))
        for a in P:
            pp = P[a][:, mask]; z = pp >= .5
            out[a] = dict(AP=float(np.mean([ap(yy, q) for q in pp])), AUROC=float(np.mean([auroc(yy, q) for q in pp])), per_fold_AP=[ap(yy, q) for q in pp], per_fold_AUROC=[auroc(yy, q) for q in pp], sens=float((z[:, yy]).mean()), spec=float((~z[:, ~yy]).mean()), Brier=float(((pp - yy) ** 2).mean()), ECE=float(np.mean([ece(yy, q) for q in pp])))
        A, U = [], []; kept = np.where(mask)[0]
        for ix in idx:
            s = kept[rng.integers(0, len(kept), len(kept))] if len(kept) != len(ids) else ix
            ys = y[s]; A.append(np.mean([ap(ys, P['V4'][k, s]) - ap(ys, P['B'][k, s]) for k in range(5)])); U.append(np.mean([auroc(ys, P['V4'][k, s]) - auroc(ys, P['B'][k, s]) for k in range(5)]))
        out['dAP'] = out['V4']['AP'] - out['B']['AP']; out['dAP_CI'] = [float(x) for x in np.percentile(A, [2.5, 97.5])]; out['dAUROC'] = out['V4']['AUROC'] - out['B']['AUROC']; out['dAUROC_CI'] = [float(x) for x in np.percentile(U, [2.5, 97.5])]
        out['proposal_pos'] = float((N[:, mask][:, yy] > 0).mean()); out['proposal_neg'] = float((N[:, mask][:, ~yy] > 0).mean()) if (~yy).any() else None
        return out
    res = dict(all=block(np.ones(len(ids), bool)), no_metal=block(~metal), n_metal=int(metal.sum()), sample=read(OUT / 'sample.json')['patients'])
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False, indent=0))


if __name__ == '__main__':
    c = sys.argv[1]
    sample() if c == 'sample' else run(sys.argv[2], sys.argv[3]) if c == 'run' else analyze()
