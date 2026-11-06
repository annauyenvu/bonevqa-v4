"""Phân tích Phần A và Phần B.6 của reports/access_v1/PROTOCOL.md.
Chạy: python -m experiments.access_contrasts_v1 contrasts | baselines | dry
contrasts từ chối chạy khi chưa đủ 5 fold x 5 nhánh. dry dùng B fold 0 cho mọi nhánh để thử mã, mọi hiệu phải bằng 0.
p của Holm là p bootstrap hai phía: 2 min(P(d <= 0), P(d >= 0)), chặn dưới 1/2000.
"""
import json, re, sys
from pathlib import Path
import numpy as np
from experiments.access_stats_v1_core import ap, auroc
ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R = ROOT / 'refactor_v2/reports'; S = R / 'local_noaux_refinement_v1'; A = R / 'v5_studies/access_v1'; O = R / 'access_v1'
CAN = 'Is there a fracture in this X-ray?'; FOLDS = '01234'; ARMS = ('B', 'V4_flag', 'V4_grid', 'V4_placebo', 'V4'); SEED = 20261103
CONTRASTS = dict(C1=('V4', 'B'), C2=('V4_flag', 'B'), C3=('V4_grid', 'B'), C4=('V4', 'V4_flag'), C5=('V4', 'V4_placebo'))
comp = {im: k for k, c in enumerate(json.load(open(ROOT / '.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json'))['components']) for im in c}


def read(p): return json.load(open(p))
def norm(s): return ' '.join(re.sub(r'[^\w\s]', ' ', str(s).lower()).split())
def weights(ids, n=2000):
    _, inv = np.unique([comp[i] for i in ids], return_inverse=True); rng = np.random.default_rng(SEED); k = inv.max() + 1
    return [np.repeat(np.arange(len(ids)), np.bincount(rng.integers(0, k, k), minlength=k)[inv]) for _ in range(n)]
def summary(y, p):
    z = p >= .5; se, sp = float(z[y].mean()), float((~z[~y]).mean())
    return dict(AP=ap(y, p), AUROC=auroc(y, p), sens=se, spec=sp, BA=(se + sp) / 2, Brier=float(((p - y) ** 2).mean()))
def boot(y, a, b, W):
    d = np.array([ap(y[s], a[s]) - ap(y[s], b[s]) for s in W]); p = max(min(1., 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(W))
    return dict(point=ap(y, a) - ap(y, b), CI=[float(x) for x in np.percentile(d, [2.5, 97.5])], p=float(p))
def holm(ps):
    order = sorted(ps, key=ps.get); out = {}; run = 0.
    for i, k in enumerate(order): run = max(run, min(1., ps[k] * (len(order) - i))); out[k] = run
    return out


def collect(folder_of, folds):
    """Trả về y, id, p_yes theo nhánh và độ đúng QA theo nhánh. Mọi nhánh phải có cùng tập khóa."""
    ids, y = [], []; P = {a: [] for a in ARMS}; Q = {a: [] for a in ARMS}; qa_meta = []
    for f in folds:
        rec = {a: read(folder_of(f, a) / 'test_predictions.json')['records'] for a in ARMS}; keys = [(r['image_id'], r['question']) for r in rec['B']]
        for a in ARMS: assert [(r['image_id'], r['question']) for r in rec[a]] == keys, (f, a)
        for k, r in enumerate(rec['B']):
            qa_meta.append(dict(fold=f, closed=norm(r['gt']) in ('yes', 'no')))
            for a in ARMS: Q[a].append(norm(rec[a][k]['pred']) == norm(r['gt']))
            if r['question'] == CAN:
                ids.append(r['image_id']); y.append(r['gt'] == 'yes')
                for a in ARMS: P[a].append(rec[a][k]['p_yes'])
    return ids, np.array(y), {a: np.array(v) for a, v in P.items()}, {a: np.array(v) for a, v in Q.items()}, qa_meta


def analyze(folder_of, folds):
    ids, y, P, Q, qm = collect(folder_of, folds); W = weights(ids); closed = np.array([m['closed'] for m in qm]); fold = np.array([m['fold'] for m in qm])
    res = dict(n=len(ids), prevalence=float(y.mean()), arms={a: summary(y, P[a]) for a in ARMS}, qa={a: dict(all=float(Q[a].mean()), closed=float(Q[a][closed].mean()), open=float(Q[a][~closed].mean())) for a in ARMS}, contrasts={k: boot(y, P[a], P[b], W) for k, (a, b) in CONTRASTS.items()})
    for k, v in holm({k: c['p'] for k, c in res['contrasts'].items()}).items(): res['contrasts'][k]['p_holm'] = v
    return res


def contrasts():
    missing = [f'{f}/{a}' for f in FOLDS for a in ARMS if not (A / f'fold_{f}/{a}_seed43/completed.json').exists() or not read(A / f'fold_{f}/{a}_seed43/replay.json').get('passed')]
    if missing: sys.exit('Chưa đủ, không phân tích: ' + ' '.join(missing))
    res = analyze(lambda f, a: A / f'fold_{f}/{a}_seed43', FOLDS); res['per_fold'] = {}
    for f in FOLDS:
        ids, y, P, _, _ = collect(lambda g, a: A / f'fold_{g}/{a}_seed43', f); res['per_fold'][f] = {k: ap(y, P[a]) - ap(y, P[b]) for k, (a, b) in CONTRASTS.items()}
    (O / 'contrasts.json').write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False, indent=0))


def dry():
    res = analyze(lambda f, a: A / f'fold_{f}/B_seed43', '0'); assert all(c['point'] == 0 and c['CI'] == [0, 0] for c in res['contrasts'].values()); print('dry ok: mọi hiệu bằng 0', res['n'], {k: c['p_holm'] for k, c in res['contrasts'].items()})


def baselines():
    B = O / 'image_baselines'; clip = read(B / 'biomedclip_lr.json'); out = {}
    for name, get, folds in (('BiomedCLIP_LR', lambda f: clip[f], FOLDS), ('MobileNetV3', lambda f: read(B / f'mobilenet_fold{f}.json')['p_yes'], [f for f in FOLDS if (B / f'mobilenet_fold{f}.json').exists()])):
        if not folds: continue
        ids, y, base, pb, pv = [], [], [], [], []
        for f in folds:
            g = get(f); pr = {a: {r['image_id']: r for r in read(S / f'fold_{f}/{d}/test_predictions.json')['records'] if r['question'] == CAN} for a, d in (('B', 'B'), ('V4', 'local_real_noaux'))}
            for im, r in pr['B'].items():
                ids.append(im); y.append(r['gt'] == 'yes'); base.append(g[im]); pb.append(r['p_yes']); pv.append(pr['V4'][im]['p_yes'])
        y = np.array(y); base, pb, pv = map(np.array, (base, pb, pv)); W = weights(ids)
        out[name] = dict(folds=list(folds), n=len(ids), baseline=summary(y, base), B=summary(y, pb), V4=summary(y, pv), V4_minus_baseline=boot(y, pv, base, W), B_minus_baseline=boot(y, pb, base, W))
    (B / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=0))


def methods(folds=FOLDS, dest='methods.json'):
    """Phần E: V, P, L seed 42 so với B, V4 niêm phong seed 42 và, khi đủ, B, V4 seed 43 của Phần A. Chỉ mô tả, không Holm."""
    M = R / 'v5_studies/access_v1_methods'; missing = [f'{f}/{a}' for f in folds for a in 'VPL' if not (M / f'fold_{f}/{a}_seed42/completed.json').exists() or not read(M / f'fold_{f}/{a}_seed42/replay.json').get('passed')]
    if missing: sys.exit('Chưa đủ, không phân tích: ' + ' '.join(missing))
    def can(path): return {r['image_id']: r for r in read(path)['records'] if r['question'] == CAN}
    def qa(path): return np.mean([norm(r['pred']) == norm(r['gt']) for r in read(path)['records']])
    ids, y = [], []; P = {k: [] for k in ('V', 'P', 'L', 'B', 'V4', 'B43', 'V443')}; refs = {f: dict(B=S / f'fold_{f}/B/test_predictions.json', V4=S / f'fold_{f}/local_real_noaux/test_predictions.json', **{a: M / f'fold_{f}/{a}_seed42/test_predictions.json' for a in 'VPL'}) for f in folds}
    seed43 = all((A / f'fold_{f}/{a}_seed43/completed.json').exists() for f in folds for a in ('B', 'V4'))
    for f in folds:
        d = {k: can(p) for k, p in refs[f].items()}
        if seed43: d.update(B43=can(A / f'fold_{f}/B_seed43/test_predictions.json'), V443=can(A / f'fold_{f}/V4_seed43/test_predictions.json'))
        for im, r in d['B'].items():
            ids.append(im); y.append(r['gt'] == 'yes')
            for k in d: P[k].append(d[k][im]['p_yes'])
    y = np.array(y); P = {k: np.array(v) for k, v in P.items() if v}; W = weights(ids)
    out = dict(folds=list(folds), n=len(ids), prevalence=float(y.mean()), arms={k: summary(y, v) for k, v in P.items()}, qa={k: float(np.average([qa(refs[f][k]) for f in folds])) for k in ('V', 'P', 'L', 'B', 'V4')}, diffs={})
    for a in 'VPL':
        for ref in [r for r in ('B', 'V4', 'B43', 'V443') if r in P]: out['diffs'][f'{a}-{ref}'] = boot(y, P[a], P[ref], W)
    (O / dest).write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=0))


if __name__ == '__main__': dict(contrasts=contrasts, dry=dry, baselines=baselines, methods=methods)[sys.argv[1]]()
