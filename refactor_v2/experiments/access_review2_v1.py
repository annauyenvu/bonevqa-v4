"""Post hoc checks requested in the second advisor review (2026-11-07). Stored predictions only; trains nothing.

  R1  Exact-match QA accuracy over all 17,998 pairs (seed 42): component-bootstrap CI and two-sided bootstrap p,
      replacing the row-level McNemar test as the primary evidence (pairs from one image are not independent).
  R2  What V4_flag can see: proposal-count distribution, fracture prevalence and V4_flag's mean score by count,
      and whether count ranks fractures inside images with a proposal (seed 43).
  R3  Late fusion of B with detector signals: logistic regression on [1, logit p_B, m] ("flag", binary only) or
      [1, logit p_B, m, n] ("flag+count"), fitted per fold on the calibration + model-validation roles of that fold
      (never on outer test labels), applied once to the fold's test images, pooled over five folds.
      CIs and p by the same component bootstrap as the paper (2,000 resamples). Seeds 42 and 43.
Run: python -m experiments.access_review2_v1   -> reports/access_v1/review2.json
"""
import json, re
from pathlib import Path
import numpy as np
from experiments.access_stats_v1_core import ap, auroc

ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R = ROOT / 'refactor_v2/reports'
S = R / 'local_noaux_refinement_v1'; A = R / 'v5_studies/access_v1'; O = R / 'access_v1'
CAN = 'Is there a fracture in this X-ray?'; FOLDS = '01234'; SEED = 20261108; NBOOT = 2000
comp = {im: k for k, c in enumerate(json.load(open(ROOT / '.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json'))['components']) for im in c}
ARM = {('s42', 'B'): lambda f: S / f'fold_{f}/B', ('s42', 'V4'): lambda f: S / f'fold_{f}/local_real_noaux'}
for a in ('B', 'V4', 'V4_flag'): ARM[('s43', a)] = (lambda f, a=a: A / f'fold_{f}/{a}_seed43')


def norm(s): return ' '.join(re.sub(r'[^\w\s]', ' ', str(s).lower()).split())
def recs(p): return json.load(open(p))['records']
def can(p): return [r for r in recs(p) if r['question'] == CAN]
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def resamples(ids):
    _, inv = np.unique([comp[i] for i in ids], return_inverse=True); rng = np.random.default_rng(SEED); k = inv.max() + 1
    return [np.repeat(np.arange(len(ids)), np.bincount(rng.integers(0, k, k), minlength=k)[inv]) for _ in range(NBOOT)]


def pboot(d):  # two-sided bootstrap sign test, as in access_contrasts_v1
    return float(max(min(1., 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d)))


def contrast(stat, W, N):
    d = np.array([stat(s) for s in W])
    return dict(point=float(stat(np.arange(N))), CI=[float(x) for x in np.percentile(d, [2.5, 97.5])], p=pboot(d))


def fit(X, y, steps=60):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + 1e-4 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-4 * np.eye(len(w)); w -= np.linalg.solve(H, g)
    return w


def feats(rows, boxes, mode):
    n = np.array([len(boxes[r['image_id']]) for r in rows], float); cols = [np.ones(len(rows)), logit(np.array([r['p_yes'] for r in rows])), (n > 0).astype(float)]
    if mode == 'flag+count': cols.append(n)
    return np.stack(cols, 1)


def r1():
    ids, b, v = [], [], []
    for f in FOLDS:
        rb, rv = recs(ARM[('s42', 'B')](f) / 'test_predictions.json'), recs(ARM[('s42', 'V4')](f) / 'test_predictions.json')
        assert [(r['image_id'], r['question']) for r in rb] == [(r['image_id'], r['question']) for r in rv]
        ids += [r['image_id'] for r in rb]; b += [norm(r['pred']) == norm(r['gt']) for r in rb]; v += [norm(r['pred']) == norm(r['gt']) for r in rv]
    b, v = np.array(b, float), np.array(v, float); W = resamples(ids)
    return dict(n_pairs=len(ids), n_images=len(set(ids)), EM_B=float(b.mean()), EM_V4=float(v.mean()),
                V4_minus_B=contrast(lambda s: v[s].mean() - b[s].mean(), W, len(ids)),
                discordant=dict(V4_only=int(((v == 1) & (b == 0)).sum()), B_only=int(((v == 0) & (b == 1)).sum())))


def r2_r3():
    out = {}
    for seed in ('s42', 's43'):
        arms = [a for (s, a) in ARM if s == seed]; ids, y, n, P = [], [], [], {a: [] for a in arms}; F = {'flag': [], 'flag+count': []}; W8 = {}
        for f in FOLDS:
            boxes = json.load(open(S / f'fold_{f}/boxes.json'))
            fitrows = can(ARM[(seed, 'B')](f) / 'calibration_predictions.json') + can(ARM[(seed, 'B')](f) / 'model_val_predictions.json')
            test = {a: can(ARM[(seed, a)](f) / 'test_predictions.json') for a in arms}
            ref = [r['image_id'] for r in test['B']]
            for a in arms: assert [r['image_id'] for r in test[a]] == ref
            assert not set(ref) & {r['image_id'] for r in fitrows}, 'fit rows overlap the outer test fold'
            yf = np.array([r['gt'] == 'yes' for r in fitrows], float)
            for mode in F:
                w = fit(feats(fitrows, boxes, mode), yf); W8.setdefault(mode, []).append([float(x) for x in w])
                F[mode].append(1 / (1 + np.exp(-feats(test['B'], boxes, mode) @ w)))
            ids += ref; y += [r['gt'] == 'yes' for r in test['B']]; n += [len(boxes[i]) for i in ref]
            for a in arms: P[a] += [r['p_yes'] for r in test[a]]
        y, n = np.array(y), np.array(n); P = {a: np.array(v) for a, v in P.items()}; F = {k: np.concatenate(v) for k, v in F.items()}; W = resamples(ids)
        L = {'B': P['B'], 'V4': P['V4'], 'B+flag': F['flag'], 'B+flag+count': F['flag+count']}
        apd = lambda a, b: (lambda s: ap(y[s], L[a][s]) - ap(y[s], L[b][s]))
        res = dict(AP={k: ap(y, v) for k, v in L.items()}, weights=W8,
                   contrasts={f'{a} - {b}': contrast(apd(a, b), W, len(ids)) for a, b in
                              (('B+flag', 'B'), ('B+flag+count', 'B'), ('B+flag+count', 'B+flag'), ('V4', 'B+flag'), ('V4', 'B+flag+count'))})
        if seed == 's43':
            m1 = n > 0
            res['count'] = dict(
                images={int(k): int((n == k).sum()) for k in range(4)},
                prevalence={int(k): float(y[n == k].mean()) for k in range(4)},
                V4_flag_mean_p={int(k): float(P['V4_flag'][n == k].mean()) for k in range(4)},
                V4_mean_p={int(k): float(P['V4'][n == k].mean()) for k in range(4)},
                AUROC_count_within_m1=auroc(y[m1], n[m1].astype(float)),
                AUROC_V4_flag_within_m1=auroc(y[m1], P['V4_flag'][m1]),
                AUROC_B_within_m1=auroc(y[m1], P['B'][m1]))
        out[seed] = res
    return out


def main():
    out = dict(R1_exact_match_seed42=r1(), **{f'R2R3_{k}': v for k, v in r2_r3().items()})
    (O / 'review2.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
