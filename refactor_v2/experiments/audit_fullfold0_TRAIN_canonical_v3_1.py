"""Read-only artifact and metric recomputation, no model instantiation."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'
QUALITY=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_artifact_audit.json'
    assert not destination.exists()
    started=time.perf_counter()
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    from src.utils import compute_metrics,normalize_yes_no
    report=read(OUT/'report.json');plan=read(OUT/'protocol.json')
    state=read(OUT/'progress.json')
    assert state['status']=='replayed_pending_independent_artifact_audit'
    assert state['report_sha256']==sha(OUT/'report.json')
    assert report['protocol_sha256']==sha(OUT/'protocol.json')
    assert report['replay_sha256']==sha(OUT/'fresh_process_replay.json')
    replay=read(OUT/'fresh_process_replay.json')
    assert replay['passed'] and replay['all2568_records_probabilities_metrics_exact_each']
    assert replay['constructor_seed']==90211 and replay['optimizer_updates']==0
    assert not replay['calibration_test_used']
    for path,digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest,path
    preflight=read(OUT/'independent_preflight.json')
    assert preflight['passed'] and preflight['protocol_sha256']==sha(OUT/'protocol.json')
    training=read(plan['TRAIN_manifest'])
    selected=[r for r in training if r['question']==plan['canonical_question']]
    assert len(selected)==len({r['image_id'] for r in selected})==2568
    expected=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in selected]
    identity=lambda rows:[(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]
    quality=read(QUALITY/'report.json')
    assert read(QUALITY/'independent_artifact_audit.json')['report_sha256']==sha(QUALITY/'report.json')
    counts={};fn_ids={}
    for mode,info in plan['targets'].items():
        path=OUT/(mode+'.json');assert sha(path)==report['prediction_sha256'][mode]
        result=read(path);rows=result['records']
        assert identity(rows)==expected
        assert all(math.isfinite(r['p_yes']) and 0<=r['p_yes']<=1 for r in rows)
        assert compute_metrics(rows,fracture_threshold=.5)==result['metrics']==report['TRAIN_metrics'][mode]
        assert result['weights_unchanged'] and result['checkpoint_sha256']==sha(info['path'])==info['sha256']
        assert report['saved_modelval_metrics'][mode]==quality['results'][mode]['canonical_metrics']
        labels=np.array([normalize_yes_no(r['gt'])=='yes' for r in rows])
        scores=np.array([r['p_yes'] for r in rows]);pred=scores>=.5
        assert int(labels.sum())==458 and int((~labels).sum())==2110
        assert all(normalize_yes_no(r['pred'])==('yes' if p else 'no') for r,p in zip(rows,pred))
        tp=int((pred&labels).sum());tn=int((~pred&~labels).sum())
        m=result['metrics']
        assert m['fracture_sensitivity']==tp/458 and m['fracture_specificity']==tn/2110
        assert m['fracture_balanced_accuracy']==.5*(tp/458+tn/2110)
        assert abs(m['overall_acc']-(tp+tn)/2568)<1e-15
        assert m['fracture_auroc']==float(roc_auc_score(labels,scores))
        assert m['fracture_auprc']==float(average_precision_score(labels,scores))
        assert m['fracture_brier']==float(brier_score_loss(labels,scores))
        bins=np.minimum((scores*10).astype(int),9)
        ece=sum(float(np.mean(bins==b))*abs(float(scores[bins==b].mean())-float(labels[bins==b].mean()))
            for b in range(10) if np.any(bins==b))
        assert abs(ece-m['fracture_ece10'])<1e-15
        counts[mode]={'TP':tp,'FN':458-tp,'TN':tn,'FP':2110-tn}
        fn_ids[mode]={r['image_id'] for r,p,y in zip(rows,pred,labels) if y and not p}
    assert not report['modelval_new_inference'] and not report['calibration_test_used']
    assert report['optimizer_updates']==report['threshold_searches']==0
    result={'passed':True,'report_sha256':sha(OUT/'report.json'),'protocol_sha256':sha(OUT/'protocol.json'),
        'auditor_sha256':sha(__file__),'all2568identities_order_labels_perarm_exact':True,
        'bothphysicalLASThashes_and_sources_checked':True,
        'metrics_and_manual_confusion_sklearn_scores_ECE_recomputed':True,
        'fresh_replay_report_binding_NOT_additional_neural_replay':True,
        'counts':counts,'TRAIN_FN_overlap':{'common':len(set.intersection(*fn_ids.values())),
            'native_only':len(fn_ids['token_mean']-fn_ids['sequence_uniform']),
            'uniform_only':len(fn_ids['sequence_uniform']-fn_ids['token_mean'])},
        'elapsed_seconds':time.perf_counter()-started,'neural_models':0,'optimizer_updates':0,
        'threshold_searches':0,'modelval_new_inference':False,'calibration_test_used':False,
        'TRAIN_fit_diagnostic_NOT_independent_efficacy_or_causal_proof':True}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
