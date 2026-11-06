"""Post-hoc diagnostic audit; fixed calibrator refit on calibration only, no tuning."""
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
REPORTS = ROOT / 'refactor_v2/reports'
OUT = REPORTS / 'sensitivity_calibration_audit_v3_1.json'
Q = 'Is there a fracture in this X-ray?'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def scores(rows):
    y = np.array([r['gt'] == 'yes' for r in rows])
    p = np.array([r['p_yes'] for r in rows])
    d = p >= .5
    tp, fn, tn, fp = [int(x.sum()) for x in (y & d, y & ~d, ~y & ~d, ~y & d)]
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return dict(n=len(y), positives=int(y.sum()), TP=tp, FN=fn, TN=tn, FP=fp,
                sensitivity=sens, specificity=spec,
                BA=(sens+spec)/2 if sens is not None and spec is not None else None,
                AUROC=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                AP=float(average_precision_score(y, p)) if y.any() else None,
                Brier=float(np.mean((p-y)**2)),
                positive_probability_quantiles=np.quantile(p[y], [0,.1,.25,.5,.75,.9,1]).tolist() if y.any() else [],
                positives_below_025=int((y & (p < .25)).sum()),
                positives_025_to_05=int((y & (p >= .25) & (p < .5)).sum()))

def main():
    assert not OUT.exists(), 'Preserve previous analysis'
    start = time.perf_counter()
    seal = read(REPORTS / 'GROUPED_EXPERIMENT_CLOSURE_SEAL_V3_1.json')
    assert all(sha(p) == h for p,h in seal['artifacts'].items())
    b = read(REPORTS / 'grouped_evaluation_bindings_v3_1/bindings.json')
    manifest = read(ROOT / '.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json')
    owners = {image:f'component_{i:05d}' for i,c in enumerate(manifest['components']) for image in c}
    assert owners == b['image_to_known_component']
    image_hashes = {}
    folds, pooled, adjusted_all, metadata = {}, [], [], {}
    parameters = dict(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=42,fit_intercept=True,class_weight=None)
    for fold, binding in b['folds'].items():
        assert sha(binding['checkpoint']) == binding['checkpoint_sha256']
        cfg = read(binding['config'])
        roles, groups, hashes, prevalence = {}, {}, {}, {}
        for role, info in binding['roles'].items():
            assert sha(info['path']) == info['sha256'] == manifest['folds'][fold][role]['sha256']
            records = read(info['path'])
            rows = [r for r in records if r['question'] == Q]
            assert len(rows) == len({r['image_id'] for r in rows}) == info['images']
            assert all(r['answer'] in ('yes','no') and (r['answer']=='yes') == bool(r['fractured']) for r in rows)
            roles[role] = {r['image_id'] for r in rows}
            groups[role] = {owners[i] for i in roles[role]}
            for r in rows:
                image = r['image_id']
                if image not in image_hashes:
                    image_hashes[image] = sha(Path(cfg['data']['loader_kwargs']['root']) / r['image'])
            hashes[role] = {image_hashes[i] for i in roles[role]}
            prevalence[role] = dict(n=len(rows), positive=sum(r['fractured'] for r in rows))
            if role == 'test':
                for r in rows:
                    assert r['image_id'] not in metadata
                    metadata[r['image_id']] = r
        for a,c in itertools.combinations(roles,2):
            assert not roles[a] & roles[c]
            assert not groups[a] & groups[c]
            assert not hashes[a] & hashes[c]
        d = read(REPORTS / f'grouped_reserved_evaluation_v3_1/fold_{fold}.json')
        cal = d['calibration_records']
        assert {r['image_id'] for r in cal} == roles['calibration']
        expected_cal = {r['image_id']:r for r in read(binding['roles']['calibration']['path']) if r['question']==Q}
        assert all(r['gt']==expected_cal[r['image_id']]['answer'] and r['question']==Q for r in cal)
        p = np.clip([r['p_yes'] for r in cal],1e-6,1-1e-6)
        model = LogisticRegression(**parameters).fit(np.log(p/(1-p)).reshape(-1,1), [int(r['gt']=='yes') for r in cal])
        a, c = float(model.coef_[0,0]), float(model.intercept_[0])
        assert a > 0 and model.n_iter_[0] < parameters['max_iter']
        assert abs(a-d['calibration']['slope'])<1e-12 and abs(c-d['calibration']['intercept'])<1e-12
        raw = [r for r in d['raw_records'] if r['question']==Q]
        adjusted = [r for r in d['calibrated_records'] if r['question']==Q]
        assert len(raw)==len(roles['test']) and {r['image_id'] for r in raw}==roles['test']
        assert all(r['gt']==metadata[r['image_id']]['answer'] for r in raw)
        x = np.clip([r['p_yes'] for r in raw],1e-6,1-1e-6)
        transformed = model.predict_proba(np.log(x/(1-x)).reshape(-1,1))[:,1]
        assert np.allclose(transformed,[r['p_yes'] for r in adjusted],rtol=0,atol=1e-12)
        assert all(r['image_id']==s['image_id'] and r['gt']==s['gt'] and s['pred']==('yes' if p>=.5 else 'no') for r,s,p in zip(raw,adjusted,transformed))
        raw_metric, cal_metric = scores(raw), scores(adjusted)
        for mode,m in [('raw',raw_metric),('calibrated',cal_metric)]:
            old = d[mode+'_metrics']['canonical']
            for new,key in [('sensitivity','fracture_sensitivity'),('specificity','fracture_specificity'),('BA','fracture_balanced_accuracy'),('AUROC','fracture_auroc'),('AP','fracture_auprc'),('Brier','fracture_brier')]:
                assert abs(m[new]-old[key])<1e-12
        # Paired known-component uncertainty, conditional on fixed models, no refitting.
        keys = sorted({owners[r['image_id']] for r in raw})
        counts = np.zeros((len(keys),3),dtype=int)
        index = {k:i for i,k in enumerate(keys)}
        for r,s in zip(raw,adjusted):
            if r['gt']=='yes':
                counts[index[owners[r['image_id']]]] += [1,int(r['p_yes']>=.5),int(s['p_yes']>=.5)]
        rng = np.random.default_rng(20261022+int(fold))
        sums = counts[rng.integers(0,len(keys),(2000,len(keys)))].sum(axis=1)
        assert (sums[:,0]>0).all()
        intervals = {name:np.quantile(sums[:,j]/sums[:,0],[.025,.975]).tolist() for name,j in [('raw',1),('calibrated',2)]}
        folds[fold] = dict(role_counts=prevalence,raw=raw_metric,calibrated=cal_metric,
            calibration_only_refit=dict(slope=a,intercept=c,raw_probability_equivalent_to_calibrated_05=float(1/(1+np.exp(c/a))),n_iter=int(model.n_iter_[0])),
            calibration_raw=scores(cal),sensitivity_component_CI95=intervals,
            region_positive_counts={region:sum(r['gt']=='yes' and metadata[r['image_id']]['region']==region for r in raw) for region in sorted({metadata[r['image_id']]['region'] for r in raw})})
        pooled.extend(raw); adjusted_all.extend(adjusted)
    assert len(pooled)==len({r['image_id'] for r in pooled})==4013
    assert len({owners[r['image_id']] for r in pooled})==3920
    subgroup = {}
    for field in ('region','hardware','fracture_count','views'):
        value = lambda r:json.dumps(metadata[r['image_id']][field],sort_keys=True)
        subgroup[field] = {v:scores([r for r in pooled if value(r)==v]) for v in sorted({value(r) for r in pooled})}
    # Descriptive case-mix standardization; overlapping training sets preclude causal fold inference.
    regions = sorted({metadata[r['image_id']]['region'] for r in pooled if r['gt']=='yes'})
    weights = {g:sum(r['gt']=='yes' and metadata[r['image_id']]['region']==g for r in pooled)/715 for g in regions}
    for fold in folds:
        rows = [r for r in read(REPORTS/f'grouped_reserved_evaluation_v3_1/fold_{fold}.json')['raw_records'] if r['question']==Q]
        rates = {}
        for g in regions:
            pos = [r for r in rows if r['gt']=='yes' and metadata[r['image_id']]['region']==g]
            rates[g] = sum(r['p_yes']>=.5 for r in pos)/len(pos) if pos else None
        folds[fold]['region_positive_sensitivity'] = rates
        folds[fold]['pooled_region_standardized_sensitivity'] = sum(weights[g]*rates[g] for g in regions) if all(rates[g] is not None for g in regions) else None
    output = dict(passed=True,scope='Post-hoc descriptive analysis; original frozen thresholds and model unchanged; no outer fitting or threshold selection',
        source_sha256=sha(__file__),closure_artifacts_verified=len(seal['artifacts']),image_bytes_rehashed=len(image_hashes),
        partition_audit='Zero image-ID, known-component and exact-byte intersections among four roles within each fold; outer IDs exactly once',
        folds=folds,pooled_raw=scores(pooled),pooled_calibrated=scores(adjusted_all),subgroups_existing_ground_truth=subgroup,
        region_standardization_weights=weights,
        false_negative_records=[dict(image_id=r['image_id'],p_yes=r['p_yes'],known_component=owners[r['image_id']],region=metadata[r['image_id']]['region'],hardware=metadata[r['image_id']]['hardware'],fracture_count=metadata[r['image_id']]['fracture_count'],views=metadata[r['image_id']]['views']) for r in pooled if r['gt']=='yes' and r['p_yes']<.5],
        limitations=['Known components are not verified patients','Previously inspected development-informed outer folds','Subgroups are exploratory, sparse and unadjusted for multiple comparisons','One training seed; folds differ in training and test data, so model variance and case-mix cannot be separated','Fresh byte checks cover sealed FracAtlas images, not a new exhaustive upstream perceptual audit'],
        elapsed_seconds=time.perf_counter()-start)
    with OUT.open('x') as f:
        json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:output[k] for k in ('passed','closure_artifacts_verified','image_bytes_rehashed','elapsed_seconds')}))
    print(json.dumps(folds,indent=2))

if __name__=='__main__':
    main()
