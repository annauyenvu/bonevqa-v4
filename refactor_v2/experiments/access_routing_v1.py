"""Phần F của reports/access_v1/PROTOCOL.md: định tuyến câu hỏi theo ý nghĩa và ngưỡng chọn trên calibration. Không huấn luyện lại.
Chạy: python -m experiments.access_routing_v1 check | verify | run <fold> <arm> <set> <index> <route> | threshold | analyze
set: old (6 câu Phần B.5) hoặc new (6 câu mới khóa ở Phần F); route: router hoặc current.
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
from experiments.thesis_v4_checks_v1 import read, SEALED, ARMDIR, CAN
from experiments.access_paraphrase_v1 import QUESTIONS as OLD
from experiments.access_stats_v1_core import ap, auroc, ece
NEW = ['Do you see a fracture on this radiograph?', 'Is there a broken bone in this image?', 'Does this radiograph demonstrate a fracture?', 'Is a bone fracture visible here?', 'Is this X-ray free of fractures?', 'Is there not any fracture on this film?']
SETS = dict(old=OLD, new=NEW); OUT = SEALED.parent / 'access_v1/routing'; FOLDS = '01234'; SEED = 20261103
_FRAC = re.compile(r"\b(fractures?|fractured|broken|break)\b")
_EXCL = re.compile(r"\b(leg|legs|hand|hands|hip|hips|shoulder|shoulders|arm|arms|wrist|foot|feet|knee|ankle|elbow|spine|skull|pelvis|rib|ribs|finger|fingers|toe|toes|multiple|more than one|how many|several|sites?|where|located|location|which|what)\b")
_NEG = re.compile(r"\b(no|not|without|absent|free of)\b|n't\b")


def route(q):
    """(có vào fracture head không, có phủ định không). Quy tắc khóa ở Phần F, viết trước khi chạy."""
    from src.utils import is_polar_question
    s = str(q or '').strip().lower()
    hit = is_polar_question(s) and bool(_FRAC.search(s)) and not _EXCL.search(s)
    return hit, bool(hit and _NEG.search(s))


def check():
    """Router phải: nhận câu canonical, không nhận mọi mẫu câu khác của dữ liệu, nhận đủ 12 câu diễn đạt lại; câu mới không có trong TRAIN."""
    meta = read(SEALED / 'metadata.json'); qs = set(); train = set()
    for f in FOLDS:
        for role, b in meta[f]['roles'].items():
            for r in read(b['path']): qs.add(r['question']); train.add(r['question']) if role == 'train' else None
    bad = [q for q in qs if route(q)[0] != (q == CAN)]
    print('dataset templates', len(qs), 'misrouted', bad)
    for q in OLD + NEW: print(route(q), q in train, q)
    assert not bad and all(route(q)[0] for q in OLD + NEW) and not any(q in train for q in NEW)
    assert [route(q)[1] for q in OLD] == [0, 0, 0, 0, 0, 1] and [route(q)[1] for q in NEW] == [0, 0, 0, 0, 1, 1]
    print('check ok')


def verify():
    """Router không được đổi câu canonical: 80 ảnh đầu fold 0, mỗi nhánh, phải khớp dự đoán niêm phong (CPU) trong 1e-4."""
    for arm in ('B', 'V4'):
        dest = OUT / f'verify_{arm}.json'; OLD.append(CAN); run('0', arm, 'old', str(len(OLD) - 1), 'router', limit=80, dest=dest); OLD.pop()
        got = read(dest)['p_yes']; ref = sealed('0', arm, 'test'); diff = max(abs(v - ref[k]['p_yes']) for k, v in got.items())
        print(arm, len(got), 'max abs diff', diff); assert diff < 1e-4, arm
    print('verify ok')


def run(fold, arm, which, i, how, limit=None, dest=None):
    import torch
    import src.models.bonevqa as bv
    from experiments.vnext_model_v1 import VNextModel
    from src.data.fracatlas_vqa import FracAtlasVQADataset as DS
    q = SETS[which][int(i)]; dest = dest or OUT / f'fold_{fold}/{arm}/{which}{i}_{how}.json'
    if dest.exists(): return
    if how == 'router': bv.is_canonical_fracture_presence_question = lambda s: route(s)[0]
    torch.set_num_threads(2); dest.parent.mkdir(parents=True, exist_ok=True); folder = SEALED / f'fold_{fold}'
    ds = DS(records_path=read(SEALED / 'metadata.json')[fold]['roles']['test']['path'], with_masks=False, with_boxes=False)
    rows = [k for k, r in enumerate(ds.records) if r['question'] == CAN][:limit]; boxes = read(folder / 'boxes.json')
    model = VNextModel.load(str(folder / ARMDIR[arm] / 'LAST.pt'), device='cpu').eval(); res = {}; start = time.time()
    for off in range(0, len(rows), 8):
        idx = rows[off:off + 8]; ims = [ds[k]['image'] for k in idx]; ids = [ds.records[k]['image_id'] for k in idx]; qs = [q] * len(idx)
        with torch.no_grad():
            if arm == 'B': out = model.predict_batch({'images': ims, 'questions': qs})
            else:
                bx = [boxes[a] for a in ids]
                out = model.predict_batch({'images': ims, 'questions': qs, 'boxes': [b or None for b in bx], 'masks': [DS._boxes_to_masks(b, im.size) if b else [] for b, im in zip(bx, ims)]})
        for a, p in zip(ids, out): res[a] = float(p['p_yes'])
        print(off, len(rows), round(time.time() - start), flush=True)
    t = dest.with_suffix('.tmp'); t.write_text(json.dumps(dict(question=q, route=how, routed=route(q), fold=fold, arm=arm, p_yes=res))); t.rename(dest)


def sealed(f, arm, role):
    d = 'B' if arm == 'B' else 'local_real_noaux'
    return {r['image_id']: r for r in read(SEALED / f'fold_{f}/{d}/{role}_predictions.json')['records'] if r['question'] == CAN}


def youden(y, p):
    t = np.unique(p); j = [(p[y] >= x).mean() + (p[~y] < x).mean() - 1 for x in t]; return float(t[int(np.argmax(j))])


def threshold():
    """Ngưỡng Youden và Platt chọn trên vai trò calibration của từng fold, áp lên test và lên kiểm định ngoài."""
    from sklearn.linear_model import LogisticRegression
    ext = SEALED.parent / 'access_v1/external'; import csv
    meta = {r['filestem']: r for r in csv.DictReader(open('/Users/uyen/Downloads/BONEVQA/.evaluation_runtime/data/external_grazpedwri/raw/dataset.csv', encoding='utf-8-sig'))}
    ids = read(ext / 'sample.json')['ids']; ye = np.array([meta[i]['fracture_visible'].strip() == '1' for i in ids])
    out = {}
    for arm in ('B', 'V4'):
        Y, P, T, PP, E, EP = [], [], [], [], [], []; th = {}
        for f in FOLDS:
            c = sealed(f, arm, 'calibration'); t = sealed(f, arm, 'test')
            yc = np.array([r['gt'] == 'yes' for r in c.values()]); pc = np.array([r['p_yes'] for r in c.values()]); x = youden(yc, pc); th[f] = x
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1)).reshape(-1, 1); m = LogisticRegression(C=1e6, max_iter=1000).fit(lg(pc), yc)
            yt = np.array([r['gt'] == 'yes' for r in t.values()]); pt = np.array([r['p_yes'] for r in t.values()])
            Y.append(yt); P.append(pt); T.append(np.full(len(pt), x)); PP.append(m.predict_proba(lg(pt))[:, 1])
            pe = np.array([read(ext / f'fold_{f}_{arm}.json')['p_yes'][i] for i in ids]); E.append(pe >= x); EP.append(m.predict_proba(lg(pe))[:, 1])
        y, p, t, pp = map(np.concatenate, (Y, P, T, PP))
        def ops(y, z): se, sp = float(z[y].mean()), float((~z[~y]).mean()); return dict(sens=se, spec=sp, BA=(se + sp) / 2)
        out[arm] = dict(thresholds=th, test_at_05=ops(y, p >= .5), test_at_youden=ops(y, p >= t), test_platt=dict(ECE_before=ece(y, p), ECE_after=ece(y, pp), Brier_before=float(((p - y) ** 2).mean()), Brier_after=float(((pp - y) ** 2).mean())),
                        external_at_05=[ops(ye, np.array([read(ext / f'fold_{f}_{arm}.json')['p_yes'][i] for i in ids]) >= .5) for f in FOLDS], external_at_youden=[ops(ye, z) for z in E], external_platt_ECE=[ece(ye, q) for q in EP])
        for k in ('external_at_05', 'external_at_youden'): out[arm][k + '_mean'] = {m: float(np.mean([d[m] for d in out[arm][k]])) for m in ('sens', 'spec', 'BA')}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'threshold.json').write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(json.dumps(out, ensure_ascii=False, indent=0))


def analyze():
    from experiments.access_contrasts_v1 import weights
    ids, y, can = [], [], {a: [] for a in ('B', 'V4')}
    for f in FOLDS:
        s = {a: sealed(f, a, 'test') for a in can}
        for im, r in s['B'].items(): ids.append((f, im)); y.append(r['gt'] == 'yes'); [can[a].append(s[a][im]['p_yes']) for a in can]
    y = np.array(y); can = {a: np.array(v) for a, v in can.items()}; W = weights([i for _, i in ids]); res = []
    for which, qs in SETS.items():
        for i, q in enumerate(qs):
            neg = route(q)[1]; e = dict(set=which, question=q, negated=neg)
            for how in ('current', 'router'):
                for a in ('B', 'V4'):
                    src = OUT / f'fold_{{}}/{a}/{which}{i}_{how}.json'
                    if which == 'old' and how == 'current':
                        p = np.array([read(SEALED.parent / f'access_v1/paraphrase/fold_{f}/{a}/q{i}.json')['rows'][im]['p_yes'] for f, im in ids])
                    else:
                        d = {f: read(Path(str(src).format(f)))['p_yes'] for f in FOLDS}; p = np.array([d[f][im] for f, im in ids])
                    # router: p_yes là p_fracture của fracture head, đã là điểm gãy; head đóng: p_yes là xác suất "yes" của câu phủ định
                    s = 1 - p if neg and how == 'current' else p; z = s >= .5
                    e[f'{a}_{how}'] = dict(AP=ap(y, s), AUROC=auroc(y, s), sens=float(z[y].mean()), spec=float((~z[~y]).mean()), agree_canonical=float((z == (can[a] >= .5)).mean())); e[f'_{a}_{how}'] = s
            for a in ('B', 'V4'):
                d = np.array([ap(y[w], e[f'_{a}_router'][w]) - ap(y[w], e[f'_{a}_current'][w]) for w in W]); e[f'{a}_router_minus_current'] = dict(point=e[f'{a}_router']['AP'] - e[f'{a}_current']['AP'], CI=[float(x) for x in np.percentile(d, [2.5, 97.5])])
            d = np.array([ap(y[w], e['_V4_router'][w]) - ap(y[w], e['_B_router'][w]) for w in W]); e['router_V4_minus_B'] = dict(point=e['V4_router']['AP'] - e['B_router']['AP'], CI=[float(x) for x in np.percentile(d, [2.5, 97.5])])
            res.append({k: v for k, v in e.items() if not k.startswith('_')})
    (OUT / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for e in res: print(e['set'], e['negated'], e['question'], {k: round(e[k]['AP'], 3) for k in ('B_current', 'B_router', 'V4_current', 'V4_router')}, 'V4 router-current', round(e['V4_router_minus_current']['point'], 3), [round(x, 3) for x in e['V4_router_minus_current']['CI']], 'router V4-B', round(e['router_V4_minus_B']['point'], 3), [round(x, 3) for x in e['router_V4_minus_B']['CI']])


if __name__ == '__main__':
    c = sys.argv[1]
    dict(check=check, verify=verify, threshold=threshold, analyze=analyze)[c]() if c != 'run' else run(*sys.argv[2:7])
