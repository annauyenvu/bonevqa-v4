"""Phân tích Phần B.5 của reports/access_v1/PROTOCOL.md: gộp 5 fold, ghi reports/access_v1/paraphrase/results.json.
Câu phủ định có đáp án đúng đảo chiều, nên điểm gãy xương của câu đó là 1 - p_yes.
"""
import json
from pathlib import Path
import numpy as np
from experiments.access_stats_v1_core import ap, auroc
ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R = ROOT / 'refactor_v2/reports'; S = R / 'local_noaux_refinement_v1'; O = R / 'access_v1/paraphrase'
CAN = 'Is there a fracture in this X-ray?'; FOLDS = '01234'; NEG = 5; ARM = dict(B='B', V4='local_real_noaux'); meta = json.load(open(S / 'metadata.json'))
comp = {im: k for k, c in enumerate(json.load(open(ROOT / '.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json'))['components']) for im in c}
ids, y, can = [], [], dict(B=[], V4=[])
for f in FOLDS:
    pr = {a: {r['image_id']: r for r in json.load(open(S / f'fold_{f}/{d}/test_predictions.json'))['records'] if r['question'] == CAN} for a, d in ARM.items()}
    for r in json.load(open(meta[f]['roles']['test']['path'])):
        if r['question'] == CAN:
            ids.append((f, r['image_id'])); y.append(r['answer'] == 'yes'); [can[a].append(pr[a][r['image_id']]['p_yes']) for a in ARM]
y = np.array(y); can = {a: np.array(v) for a, v in can.items()}
_, inv = np.unique([comp[i] for _, i in ids], return_inverse=True); rng = np.random.default_rng(20261103); W = [np.bincount(rng.integers(0, inv.max() + 1, inv.max() + 1), minlength=inv.max() + 1)[inv] for _ in range(2000)]
def wid(w): return np.repeat(np.arange(len(y)), w)
def block(p, c):
    z = p >= .5; cz = c >= .5
    return dict(AP=ap(y, p), AUROC=auroc(y, p), sens=float(z[y].mean()), spec=float((~z[~y]).mean()), agree_canonical=float((z == cz).mean()), mean_abs_diff_canonical=float(np.abs(p - c).mean()))
res = dict(n=len(y), canonical={a: block(can[a], can[a]) for a in ARM}, questions=[])
for i in range(6):
    q = None; P = {}; closed = {}
    for a in ARM:
        v = []; ok = 0
        rows = {f: json.load(open(O / f'fold_{f}/{a}/q{i}.json')) for f in FOLDS}; q = rows['0']['question']
        for f, im in ids:
            r = rows[f]['rows'][im]; assert r['p_yes'] is not None; v.append(1 - r['p_yes'] if i == NEG else r['p_yes']); ok += str(r['answer']).strip().lower() in ('yes', 'no')
        P[a] = np.array(v); closed[a] = ok / len(ids)
    d = []
    for w in W:
        s = wid(w); d.append(ap(y[s], P['V4'][s]) - ap(y[s], P['B'][s]))
    res['questions'].append(dict(question=q, negated=i == NEG, B=block(P['B'], can['B']), V4=block(P['V4'], can['V4']), answer_yes_no=closed, dAP=ap(y, P['V4']) - ap(y, P['B']), dAP_CI=[float(x) for x in np.percentile(d, [2.5, 97.5])]))
(O / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print('n', res['n'], 'canonical AP', {a: round(res['canonical'][a]['AP'], 3) for a in ARM})
for e in res['questions']:
    print(e['question'], '| B', {k: round(e['B'][k], 3) for k in ('AP', 'sens', 'spec', 'agree_canonical')}, '| V4', {k: round(e['V4'][k], 3) for k in ('AP', 'sens', 'spec', 'agree_canonical')}, '| dAP', round(e['dAP'], 3), [round(x, 3) for x in e['dAP_CI']], '| yes/no', e['answer_yes_no'])
