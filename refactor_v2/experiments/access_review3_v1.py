"""Post hoc checks added in the third revision, 2026-11-07 (labels R2.2-R3.4 = reviewer comment codes, kept as JSON keys).
Reads the stored predictions of the release repository READ-ONLY and writes one JSON to <out.json>.
Trains nothing; touches no file in the repository.

  R2.2  Proposal distribution by fold role (TRAIN / model-validation / calibration / outer test) and fracture
        label: proposal rate, mean count, and lesion hit rate (>= 1 proposal with IoU >= 0.5 to an annotation box)
        on fractured images. TRAIN proposals are in-sample for the detector, the other roles are not.
  R2.3  Fold-level summaries: mean of the five per-fold APs and the mean of paired per-fold differences, with a
        fold-stratified component bootstrap; plus a scale-free pooled AP in which each fold's scores are replaced
        by within-fold percentile ranks before pooling (removes any between-fold score scale).
  R2.4  Paired component-level permutation (randomization) test: under H0 the two arms are exchangeable, so the
        arm labels are swapped for all images of a randomly chosen half of the duplicate components; statistic =
        pooled AP difference; 10,000 permutations, two-sided, Holm over the five prespecified seed-43 contrasts.
        Also: the original bootstrap re-run with resampling stratified by outer fold.
  R3.4  Exact-match correctness for the three negated wordings: answer of the learned closed head and of the
        rule-based router, scored against the negation-inverted reference answer.
  R3.3  (if r33_detector_scores.json is given) sensitivity of B and V4 on fractured images by top proposal score.

Usage (from refactor_v2/): python experiments/access_review3_v1.py .. reports/access_v1/review3/posthoc.json reports/access_v1/review3/detector_scores.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SC = Path(sys.argv[3]) if len(sys.argv) > 3 else None
R = REPO / 'refactor_v2/reports'; S = R / 'local_noaux_refinement_v1'; A = R / 'v5_studies/access_v1'
CV = REPO / '.evaluation_runtime/data/fracatlas_component_cv_v3'
CAN = 'Is there a fracture in this X-ray?'; FOLDS = '01234'; NB = 2000; NP = 10000; SEED = 20261109
comp = {im: k for k, c in enumerate(json.load(open(CV / 'manifest.json'))['components']) for im in c}
ARMS = {'s42': {'B': lambda f: S / f'fold_{f}/B', 'V4': lambda f: S / f'fold_{f}/local_real_noaux'},
        's43': {a: (lambda f, a=a: A / f'fold_{f}/{a}_seed43') for a in ('B', 'V4', 'V4_flag', 'V4_grid', 'V4_placebo')}}
CON = {'s42': [('V4', 'B')], 's43': [('V4', 'B'), ('V4_flag', 'B'), ('V4_grid', 'B'), ('V4', 'V4_flag'), ('V4', 'V4_placebo')]}


# ---------- metrics (same definitions as experiments/access_stats_v1_core.py)
def ap(y, p):
    ix = np.argsort(-p, kind='stable'); y, p = y[ix], p[ix]
    e = np.r_[np.where(p[:-1] != p[1:])[0], len(p) - 1]; tp = np.cumsum(y)[e]; tot = e + 1
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / tot) / y.sum())


def ci(d): return [float(x) for x in np.percentile(d, [2.5, 97.5])]
def pb(d): return float(max(min(1., 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d)))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1., (m - r) * ps[i])); adj[i] = run
    return [float(x) for x in adj]


def can(p): return [r for r in json.load(open(p))['records'] if r['question'] == CAN]


def collect(seed):
    ids, fold, y, P = [], [], [], {a: [] for a in ARMS[seed]}
    for f in FOLDS:
        rec = {a: can(fn(f) / 'test_predictions.json') for a, fn in ARMS[seed].items()}
        ref = [r['image_id'] for r in rec['B']]
        for a in rec: assert [r['image_id'] for r in rec[a]] == ref
        ids += ref; fold += [f] * len(ref); y += [r['gt'] == 'yes' for r in rec['B']]
        for a in rec: P[a] += [r['p_yes'] for r in rec[a]]
    return np.array(ids), np.array(fold), np.array(y), {a: np.array(v) for a, v in P.items()}


def strat_resamples(ids, fold, rng):
    """Component bootstrap stratified by outer fold: within each fold, draw that fold's components with replacement."""
    groups = []
    for f in FOLDS:
        ix = np.flatnonzero(fold == f); c = np.array([comp[i] for i in ids[ix]]); u, inv = np.unique(c, return_inverse=True)
        groups.append((ix, inv, len(u)))
    out = []
    for _ in range(NB):
        parts = []
        for ix, inv, k in groups:
            cnt = np.bincount(rng.integers(0, k, k), minlength=k); parts.append(np.repeat(ix, cnt[inv]))
        out.append(np.concatenate(parts))
    return out


def fold_rank(p, fold):
    r = np.empty_like(p)
    for f in FOLDS:
        ix = fold == f; r[ix] = rankdata(p[ix]) / ix.sum()
    return r


def r23_r24():
    out = {}; rng = np.random.default_rng(SEED)
    for seed in ('s42', 's43'):
        ids, fold, y, P = collect(seed); W = strat_resamples(ids, fold, rng); res = {}
        per_fold = {a: {f: ap(y[fold == f], P[a][fold == f]) for f in FOLDS} for a in P}
        res['per_fold_AP'] = per_fold
        res['mean_fold_AP'] = {a: float(np.mean(list(v.values()))) for a, v in per_fold.items()}
        res['pooled_AP'] = {a: ap(y, P[a]) for a in P}
        R_ = {a: fold_rank(P[a], fold) for a in P}
        res['pooled_AP_within_fold_ranks'] = {a: ap(y, R_[a]) for a in P}
        # component -> indices, for the permutation test
        cidx = np.array([comp[i] for i in ids]); u, inv = np.unique(cidx, return_inverse=True)
        res['contrasts'] = {}
        for a, b in CON[seed]:
            def mf(s, a=a, b=b):
                return np.mean([ap(y[s][fold[s] == f], P[a][s][fold[s] == f]) - ap(y[s][fold[s] == f], P[b][s][fold[s] == f]) for f in FOLDS])
            dmf = np.array([mf(s) for s in W])
            dpool = np.array([ap(y[s], P[a][s]) - ap(y[s], P[b][s]) for s in W])
            drank = np.array([ap(y[s], R_[a][s]) - ap(y[s], R_[b][s]) for s in W])
            obs = ap(y, P[a]) - ap(y, P[b]); prng = np.random.default_rng(SEED + 1); cnt = 0
            for _ in range(NP):
                sw = prng.random(len(u)) < 0.5; m = sw[inv]
                pa = np.where(m, P[b], P[a]); pbb = np.where(m, P[a], P[b])
                cnt += abs(ap(y, pa) - ap(y, pbb)) >= abs(obs) - 1e-12
            res['contrasts'][f'{a} - {b}'] = dict(
                pooled=dict(point=obs, CI_fold_stratified=ci(dpool), p_boot_fold_stratified=pb(dpool)),
                mean_fold=dict(point=float(mf(np.arange(len(y)))), CI=ci(dmf), p_boot=pb(dmf),
                               per_fold={f: per_fold[a][f] - per_fold[b][f] for f in FOLDS},
                               folds_positive=int(sum(per_fold[a][f] > per_fold[b][f] for f in FOLDS))),
                pooled_within_fold_ranks=dict(point=ap(y, R_[a]) - ap(y, R_[b]), CI=ci(drank)),
                permutation=dict(p=float((cnt + 1) / (NP + 1)), n_perm=NP))
        if seed == 's43':
            names = list(res['contrasts'])
            for key, get in (('p_perm', lambda c: c['permutation']['p']), ('p_boot_fold_stratified', lambda c: c['pooled']['p_boot_fold_stratified'])):
                h = holm([get(res['contrasts'][n]) for n in names])
                for n, v in zip(names, h): res['contrasts'][n][f'holm_{key}'] = v
        out[seed] = res
    return out


# ---------- R2.2
def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0])); iy = max(0., min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i; return i / u if u > 0 else 0.


def gt_boxes(rows):
    """Annotation boxes per image; the QA files store them as [x1, y1, x2, y2] in original pixels."""
    g = {}
    for r in rows:
        if r.get('boxes') and r['image_id'] not in g:
            g[r['image_id']] = [list(map(float, b[:4])) for b in r['boxes']]
    return g


def r22():
    roles = {'train': 'vqa_train.json', 'model_val': 'vqa_model_val.json', 'calibration': 'vqa_calibration.json', 'test': 'vqa_test.json'}
    acc = {r: {'frac': dict(n=0, m1=0, cnt=0, hit=0, n_gt=0), 'nonfrac': dict(n=0, m1=0, cnt=0)} for r in roles}
    per_fold = {}
    box_fmt_check = []
    for f in FOLDS:
        boxes = json.load(open(S / f'fold_{f}/boxes.json')); per_fold[f] = {}
        for role, fn in roles.items():
            rows = json.load(open(CV / f'fold_{f}/{fn}')); lab = {}; g = gt_boxes(rows)
            for r in rows: lab[r['image_id']] = int(r['fractured'])
            pf = dict(n=0, m1_frac=0, n_frac=0, m1_non=0, n_non=0)
            for im, yy in lab.items():
                b = boxes[im]; k = 'frac' if yy else 'nonfrac'; a = acc[role][k]; a['n'] += 1; a['m1'] += bool(b); a['cnt'] += len(b)
                if yy:
                    pf['n_frac'] += 1; pf['m1_frac'] += bool(b)
                    if im in g:
                        a['n_gt'] += 1; a['hit'] += any(iou(p, q) >= .5 for p in b for q in g[im])
                        if len(box_fmt_check) < 200 and b: box_fmt_check.append(max(iou(p, q) for p in b for q in g[im]))
                else:
                    pf['n_non'] += 1; pf['m1_non'] += bool(b)
            per_fold[f][role] = dict(proposal_rate_fractured=pf['m1_frac'] / pf['n_frac'], proposal_rate_nonfractured=pf['m1_non'] / pf['n_non'])
    out = {}
    for role, d in acc.items():
        fr, nf = d['frac'], d['nonfrac']
        out[role] = dict(image_role_instances=fr['n'] + nf['n'], fractured=fr['n'],
                         proposal_rate_fractured=fr['m1'] / fr['n'], proposal_rate_nonfractured=nf['m1'] / nf['n'],
                         mean_count_fractured=fr['cnt'] / fr['n'], mean_count_nonfractured=nf['cnt'] / nf['n'],
                         lesion_hit_rate_iou50_fractured=(fr['hit'] / fr['n_gt']) if fr['n_gt'] else None, n_with_annotation=fr['n_gt'],
                         flag_BA=0.5 * (fr['m1'] / fr['n'] + 1 - nf['m1'] / nf['n']))
    return dict(pooled_over_folds=out, per_fold=per_fold, note='counts are image-role instances pooled over 5 folds (each image is TRAIN in ~3-4 folds, test in exactly 1)',
                median_best_iou_sanity=float(np.median(box_fmt_check)) if box_fmt_check else None)


# ---------- R3.4
def r34():
    rout = R / 'access_v1/routing'; out = {}
    labels = {}
    for f in FOLDS:
        for r in can(S / f'fold_{f}/B/test_predictions.json'): labels[r['image_id']] = r['gt'] == 'yes'
    for i, q in ((4, 'Is this X-ray free of fractures?'), (5, 'Is there not any fracture on this film?')):
        e = dict(question=q)
        for arm in ('B', 'V4'):
            for how in ('current', 'router'):
                p = {}
                for f in FOLDS:
                    d = json.load(open(rout / f'fold_{f}/{arm}/new{i}_{how}.json')); assert d['question'] == q; p.update(d['p_yes'])
                ims = sorted(p); y = np.array([labels[k] for k in ims]); pp = np.array([p[k] for k in ims])
                # closed head: p is P(answer "yes"); router: p is P(fracture) and the answer is flipped
                ans_yes = pp >= .5 if how == 'current' else pp < .5
                ref_yes = ~y  # correct answer to a negated question is "yes" when there is no fracture
                e[f'{arm}_{how}'] = dict(n=len(ims), exact_match=float((ans_yes == ref_yes).mean()),
                                         answered_yes_rate=float(ans_yes.mean()), reference_yes_rate=float(ref_yes.mean()))
        out[f'new{i}'] = e
    # old negated wording ("Is there no fracture in this X-ray?") lives in the paraphrase folder for the closed head
    try:
        from_par = {}
        q = 'Is there no fracture in this X-ray?'; e = dict(question=q)
        for arm in ('B', 'V4'):
            idx = None
            for k in range(6):
                d = json.load(open(R / f'access_v1/paraphrase/fold_0/{arm}/q{k}.json'))
                if d.get('question') == q: idx = k
            for how in ('current', 'router'):
                p = {}
                for f in FOLDS:
                    src = (R / f'access_v1/paraphrase/fold_{f}/{arm}/q{idx}.json') if how == 'current' else (rout / f'fold_{f}/{arm}/old{idx}_router.json')
                    d = json.load(open(src)); pv = d['rows'] if 'rows' in d else d['p_yes']
                    p.update({k: (v['p_yes'] if isinstance(v, dict) else v) for k, v in pv.items()})
                ims = sorted(p); y = np.array([labels[k] for k in ims]); pp = np.array([p[k] for k in ims])
                ans_yes = pp >= .5 if how == 'current' else pp < .5
                e[f'{arm}_{how}'] = dict(n=len(ims), exact_match=float((ans_yes == ~y).mean()), answered_yes_rate=float(ans_yes.mean()), reference_yes_rate=float((~y).mean()))
        out['old_negated'] = e
    except Exception as ex:  # report instead of hiding
        out['old_negated'] = dict(error=repr(ex))
    # canonical exact match for comparison
    for arm, d in (('B', 'B'), ('V4', 'local_real_noaux')):
        rr = [r for f in FOLDS for r in can(S / f'fold_{f}/{d}/test_predictions.json')]
        out[f'canonical_exact_match_{arm}'] = float(np.mean([(r['p_yes'] >= .5) == (r['gt'] == 'yes') for r in rr]))
    return out


# ---------- R3.3
def r33():
    if not SC or not SC.exists(): return None
    sc = json.load(open(SC)); out = {}
    match = {f: (v['n_match_stored'], v['n_test']) for f, v in sc.items()}
    top = {im: (max(d['kept_scores']) if d['kept_scores'] else 0.0) for v in sc.values() for im, d in v['images'].items()}
    for seed in ('s42', 's43'):
        ids, fold, y, P = collect(seed); t = np.array([top[i] for i in ids]); bins = [(0, 0.05, 'no proposal'), (0.05, 0.25, '0.05-0.25'), (0.25, 0.5, '0.25-0.50'), (0.5, 0.75, '0.50-0.75'), (0.75, 1.01, '>=0.75')]
        res = {}
        for lo, hi, name in bins:
            m = (t >= lo) & (t < hi) if lo > 0 else (t < 0.05)
            res[name] = dict(n=int(m.sum()), n_fractured=int((m & y).sum()), prevalence=float(y[m].mean()) if m.any() else None,
                             **{f'sens_{a}': (float((P[a][m & y] >= .5).mean()) if (m & y).any() else None) for a in ('B', 'V4')},
                             **{f'spec_{a}': (float((P[a][m & ~y] < .5).mean()) if (m & ~y).any() else None) for a in ('B', 'V4')},
                             **{f'mean_p_{a}': float(P[a][m].mean()) if m.any() else None for a in ('B', 'V4')})
        out[seed] = res
    # note: torchvision Faster R-CNN drops boxes below box_score_thresh=0.05 internally, so no sub-threshold score exists for m = 0
    out['replay_match_stored_boxes'] = match
    return out


def main():
    out = dict(R2_2_proposal_distribution=r22(), R2_3_R2_4_fold_and_permutation=r23_r24(), R3_4_negation_exact_match=r34(), R3_3_by_top_score=r33())
    OUT.write_text(json.dumps(out, indent=1)); print('wrote', OUT)


if __name__ == '__main__':
    main()
