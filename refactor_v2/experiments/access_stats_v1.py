"""Phần B.1-4 của reports/access_v1/PROTOCOL.md: thống kê, nhóm con, câu mở, detector. Chỉ đọc dự đoán niêm phong. Ghi reports/access_v1/stats.json."""
import json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R = ROOT / 'refactor_v2/reports'; S = R / 'local_noaux_refinement_v1'; O = R / 'access_v1'
CAN = 'Is there a fracture in this X-ray?'; FOLDS = list('01234'); meta = json.load(open(S / 'metadata.json'))
comp = {im: k for k, c in enumerate(json.load(open(ROOT / '.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json'))['components']) for im in c}


def norm(s): return ' '.join(re.sub(r'[^\w\s]', ' ', str(s).lower()).split())
def ap(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    ix = np.argsort(-p, kind='stable'); y, p, w = y[ix], p[ix], w[ix]
    e = np.r_[np.where(p[:-1] != p[1:])[0], len(p) - 1]; tp = np.cumsum(y * w)[e]; tot = np.cumsum(w)[e]
    return float(np.sum(np.diff(np.r_[0, tp]) * np.divide(tp, tot, out=np.zeros_like(tp, float), where=tot > 0)) / np.sum(y * w))
def auroc(y, p):
    from scipy.stats import rankdata
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1; return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1); return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))
def slope(y, p):
    from sklearn.linear_model import LogisticRegression
    x = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1); m = LogisticRegression(C=1e6, max_iter=1000).fit(x, y); return float(m.coef_[0, 0]), float(m.intercept_[0])
def summary(y, p):
    z = p >= .5; se = float((y & z).sum() / max(y.sum(), 1)); sp = float((~y & ~z).sum() / max((~y).sum(), 1))
    return dict(n=int(len(y)), pos=int(y.sum()), AP=ap(y, p) if y.any() else None, AUROC=auroc(y, p) if y.any() and (~y).any() else None, sens=se, spec=sp, BA=(se + sp) / 2, Brier=float(((p - y) ** 2).mean()))
def mcnemar(a, b):
    """Đáp án đúng của hai mô hình trên cùng hàng; kiểm định chính xác."""
    n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); return dict(a_only=n10, b_only=n01, p=float(binomtest(n10, n10 + n01, .5).pvalue) if n10 + n01 else 1.0)
def bleu1(pred, ref):
    p, r = norm(pred).split(), norm(ref).split()
    if not p: return 0.
    c = Counter(r); m = sum(min(v, c[k]) for k, v in Counter(p).items()); bp = 1. if len(p) >= len(r) else math.exp(1 - len(r) / len(p)); return bp * m / len(p)
def f1(pred, ref):
    p, r = Counter(norm(pred).split()), Counter(norm(ref).split()); m = sum((p & r).values())
    return 2 * m / (sum(p.values()) + sum(r.values())) if m else 0.
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)


rows = []; qa = defaultdict(list)
for f in FOLDS:
    recs = json.load(open(meta[f]['roles']['test']['path'])); bx = json.load(open(S / f'fold_{f}/boxes.json'))
    pb = {(r['image_id'], r['question']): r for r in json.load(open(S / f'fold_{f}/B/test_predictions.json'))['records']}
    pv = {(r['image_id'], r['question']): r for r in json.load(open(S / f'fold_{f}/local_real_noaux/test_predictions.json'))['records']}
    for r in recs:
        k = (r['image_id'], r['question']); assert pb[k]['gt'] == r['answer'] == pv[k]['gt']
        qa[r['question']].append(dict(fold=f, type=r['answer_type'], gt=r['answer'], B=pb[k]['pred'], V4=pv[k]['pred']))
        if r['question'] == CAN:
            g = [b[:4] for b in (r.get('boxes') or [])]; b = bx[r['image_id']]
            rows.append(dict(id=r['image_id'], fold=f, y=r['answer'] == 'yes', B=pb[k]['p_yes'], V4=pv[k]['p_yes'], region=r.get('region'), hardware=bool(r.get('hardware')), views=str(r.get('views')), count=int(r.get('fracture_count') or 0), nb=len(b), best=max((iou(x, q) for x in b for q in g), default=0.), ngt=len(g)))
y = np.array([r['y'] for r in rows]); pb, pv = np.array([r['B'] for r in rows]), np.array([r['V4'] for r in rows]); fold = np.array([r['fold'] for r in rows]); nb = np.array([r['nb'] for r in rows]); ids = [r['id'] for r in rows]
_, inv = np.unique([comp[i] for i in ids], return_inverse=True); rng = np.random.default_rng(20261103); W = np.stack([np.bincount(rng.integers(0, inv.max() + 1, inv.max() + 1), minlength=inv.max() + 1)[inv].astype(float) for _ in range(2000)])
res = dict(overall={a: {**summary(y, p), 'ECE': ece(y, p), 'cal_slope_intercept': slope(y, p)} for a, p in (('B', pb), ('V4', pv))})
d = [ap(y, pv, w) - ap(y, pb, w) for w in W]; res['dAP'] = dict(point=ap(y, pv) - ap(y, pb), CI=[float(x) for x in np.percentile(d, [2.5, 97.5])])
res['mcnemar_canonical'] = mcnemar((pv >= .5) == y, (pb >= .5) == y)
allq = [x for v in qa.values() for x in v]; ca = np.array([norm(x['V4']) == norm(x['gt']) for x in allq]); cb = np.array([norm(x['B']) == norm(x['gt']) for x in allq]); res['mcnemar_QA'] = {**mcnemar(ca, cb), 'V4': float(ca.mean()), 'B': float(cb.mean()), 'n': len(allq)}
res['per_fold'] = {f: dict(B=summary(y[fold == f], pb[fold == f]), V4=summary(y[fold == f], pv[fold == f])) for f in FOLDS}
def group(key, name):
    out = {}
    vals = np.array([str(r[key]) if key != 'count' else ('0' if r[key] == 0 else '1' if r[key] == 1 else '2+') for r in rows])
    for v in sorted(set(vals)):
        m = vals == v
        if m.sum() < 30: out[v] = dict(n=int(m.sum()), note='quá ít ảnh'); continue
        e = dict(B=summary(y[m], pb[m]), V4=summary(y[m], pv[m]), proposal_pos=float((nb[m & y] > 0).mean()) if (m & y).any() else None, proposal_neg=float((nb[m & ~y] > 0).mean()) if (m & ~y).any() else None)
        if y[m].sum() >= 10 and (~y[m]).sum() >= 10:
            dd = [ap(y[m], pv[m], w[m]) - ap(y[m], pb[m], w[m]) for w in W if (w[m] * y[m]).sum() > 0]; e['dAP'] = ap(y[m], pv[m]) - ap(y[m], pb[m]); e['dAP_CI'] = [float(x) for x in np.percentile(dd, [2.5, 97.5])]
        out[v] = e
    res[name] = out
group('region', 'by_region'); group('hardware', 'by_hardware'); group('views', 'by_view'); group('count', 'by_fracture_count')
# Câu hỏi theo họ
fam = {}
for q, v in qa.items():
    b = np.array([norm(x['B']) == norm(x['gt']) for x in v]); a = np.array([norm(x['V4']) == norm(x['gt']) for x in v])
    fam[q] = dict(n=len(v), type=v[0]['type'], B=float(b.mean()), V4=float(a.mean()), mcnemar=mcnemar(a, b))
    if v[0]['type'] != 'CLOSED':
        fam[q].update(B_f1=float(np.mean([f1(x['B'], x['gt']) for x in v])), V4_f1=float(np.mean([f1(x['V4'], x['gt']) for x in v])), B_bleu1=float(np.mean([bleu1(x['B'], x['gt']) for x in v])), V4_bleu1=float(np.mean([bleu1(x['V4'], x['gt']) for x in v])))
res['by_question'] = fam
op = [x for x in allq if x['type'] != 'CLOSED']; res['open_overall'] = {a: dict(exact=float(np.mean([norm(x[a]) == norm(x['gt']) for x in op])), f1=float(np.mean([f1(x[a], x['gt']) for x in op])), bleu1=float(np.mean([bleu1(x[a], x['gt']) for x in op]))) for a in ('B', 'V4')}; res['open_overall']['n'] = len(op)
# Detector
def det(m):
    pos = m & y & np.array([r['ngt'] > 0 for r in rows]); best = np.array([r['best'] for r in rows])
    return dict(n_pos=int(pos.sum()), recall_iou={str(t): float((best[pos] >= t).mean()) for t in (.1, .3, .5)} if pos.any() else {}, proposal_rate_pos=float((nb[m & y] > 0).mean()) if (m & y).any() else None, proposal_rate_neg=float((nb[m & ~y] > 0).mean()) if (m & ~y).any() else None, boxes_per_neg=float(nb[m & ~y].mean()) if (m & ~y).any() else None)
res['detector'] = dict(overall=det(np.ones(len(y), bool)), per_fold={f: det(fold == f) for f in FOLDS}, by_region={v: det(np.array([str(r['region']) == v for r in rows])) for v in sorted({str(r['region']) for r in rows})})
(O / 'stats.json').write_text(json.dumps(res, ensure_ascii=False, indent=1)); print('ok')
