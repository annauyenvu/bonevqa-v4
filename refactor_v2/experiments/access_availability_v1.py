"""Post hoc analysis of the proposal-availability signal (advisor review, 2026-11-07).

Reads only sealed predictions and the fold detectors' stored boxes; trains nothing.
  A1  flag-only predictor m(I) = 1[detector returned >= 1 box]: confusion matrix, Sens, Spec, BA.
  A2  AP of every arm and the V4 - B, V4_flag - B, V4 - V4_flag contrasts inside m = 1 and m = 0.
  A3  sensitivity at 0.5 on fractured images with and without a proposal (detector-miss failure mode),
      specificity on non-fractured images with and without a proposal.
Seed 42: B and V4 from local_noaux_refinement_v1. Seed 43: all five retrained arms from v5_studies/access_v1.
Bootstrap: 2,000 component resamples within folds' pooled test sets, as in access_contrasts_v1.
Run: python -m experiments.access_availability_v1   -> reports/access_v1/availability.json
"""
import json
from pathlib import Path
import numpy as np
from experiments.access_stats_v1_core import ap

ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R = ROOT / 'refactor_v2/reports'
S = R / 'local_noaux_refinement_v1'; A = R / 'v5_studies/access_v1'; O = R / 'access_v1'
CAN = 'Is there a fracture in this X-ray?'; FOLDS = '01234'; SEED = 20261107; NBOOT = 2000
comp = {im: k for k, c in enumerate(json.load(open(ROOT / '.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json'))['components']) for im in c}
SOURCES = {
    's42': {'B': lambda f: S / f'fold_{f}/B', 'V4': lambda f: S / f'fold_{f}/local_real_noaux'},
    's43': {a: (lambda f, a=a: A / f'fold_{f}/{a}_seed43') for a in ('B', 'V4', 'V4_flag', 'V4_grid', 'V4_placebo')},
}
CONTRASTS = {'s42': [('V4', 'B')], 's43': [('V4', 'B'), ('V4_flag', 'B'), ('V4', 'V4_flag'), ('V4', 'V4_placebo')]}


def collect(src):
    ids, y, m, P = [], [], [], {a: [] for a in src}
    for f in FOLDS:
        boxes = json.load(open(S / f'fold_{f}/boxes.json'))
        rec = {a: [r for r in json.load(open(fn(f) / 'test_predictions.json'))['records'] if r['question'] == CAN] for a, fn in src.items()}
        ref = [r['image_id'] for r in rec['B']]
        for a in src: assert [r['image_id'] for r in rec[a]] == ref, (f, a)
        for k, im in enumerate(ref):
            ids.append(im); y.append(rec['B'][k]['gt'] == 'yes'); m.append(len(boxes[im]) > 0)
            for a in src: P[a].append(rec[a][k]['p_yes'])
    return ids, np.array(y), np.array(m), {a: np.array(v) for a, v in P.items()}


def resamples(ids):
    _, inv = np.unique([comp[i] for i in ids], return_inverse=True); rng = np.random.default_rng(SEED); k = inv.max() + 1
    return [np.repeat(np.arange(len(ids)), np.bincount(rng.integers(0, k, k), minlength=k)[inv]) for _ in range(NBOOT)]


def boot_ap(y, a, b, W, mask):
    d = []
    for s in W:
        s = s[mask[s]]
        if y[s].all() or not y[s].any(): continue
        d.append(ap(y[s], a[s]) - ap(y[s], b[s]))
    d = np.array(d); point = ap(y[mask], a[mask]) - ap(y[mask], b[mask])
    return dict(point=float(point), CI=[float(x) for x in np.percentile(d, [2.5, 97.5])], n_boot=len(d))


def rate(z, mask): return dict(value=float(z[mask].mean()), n=int(mask.sum()))


def main():
    out = {}
    for seed, src in SOURCES.items():
        ids, y, m, P = collect(src); W = resamples(ids)
        tp, fn_, fp, tn = int((m & y).sum()), int((~m & y).sum()), int((m & ~y).sum()), int((~m & ~y).sum())
        sens, spec = tp / (tp + fn_), tn / (tn + fp)
        res = dict(n=len(ids), n_pos=int(y.sum()),
                   flag_only=dict(TP=tp, FN=fn_, FP=fp, TN=tn, sens=sens, spec=spec, BA=(sens + spec) / 2, AP=ap(y, m.astype(float))),
                   strata={})
        for name, mask in (('all', np.ones_like(m)), ('m1', m), ('m0', ~m)):
            st = dict(n=int(mask.sum()), n_pos=int(y[mask].sum()), AP={a: ap(y[mask], P[a][mask]) for a in src},
                      sens={a: rate(P[a] >= .5, mask & y) for a in src}, spec={a: rate(P[a] < .5, mask & ~y) for a in src},
                      contrasts={f'{a} - {b}': boot_ap(y, P[a], P[b], W, mask) for a, b in CONTRASTS[seed]})
            res['strata'][name] = st
        out[seed] = res
    (O / 'availability.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
