"""Independent artifact/metric/component-bootstrap recomputation; no models."""
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
PARENT=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
MODES=('token_mean','sequence_uniform')
QUESTION='Is there a fracture in this X-ray?'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def audit():
    destination=OUT/'independent_artifact_audit.json'
    assert not destination.exists(), 'Preserve existing audit'
    started=time.perf_counter()
    import numpy as np
    from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
    from src.utils import compute_metrics,normalize_answer,normalize_yes_no
    report=read(OUT/'report.json')
    plan=read(OUT/'evaluation_plan.json')
    progress=read(OUT/'progress.json')
    assert progress['status']=='replayed_pending_independent_artifact_audit'
    assert progress['report_sha256']==sha(OUT/'report.json')
    assert report['plan_sha256']==sha(OUT/'evaluation_plan.json')
    assert report['replay_sha256']==sha(OUT/'fresh_process_replay.json')
    replay=read(OUT/'fresh_process_replay.json')
    assert replay['passed'] and replay['all1432records_probabilities_metrics_exact']
    assert replay['constructor_seed']==90211 and replay['training_updates']==0
    assert not replay['calibration_test_used'] and not report['calibration_test_used']
    assert not report['step3_complete'] and not report['production_released']
    for path,digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest,path
    check=read(OUT/'checkpoint_preflight.json')
    assert check['passed'] and check['plan_sha256']==sha(OUT/'evaluation_plan.json')
    assert check['evaluator_sha256']==sha(ROOT/'refactor_v2/experiments/evaluate_fullfold0_fixed_LAST_v3_1.py')
    protocol=read(PARENT/'protocol.json')
    cfg=read(protocol['configs']['token_mean']['path'])
    paths=cfg['data']['loader_kwargs']['sealed_records_by_split']
    truth=read(paths['model_val'])
    assert len(truth)==1432
    batching=cfg['train']['component_batching']
    component_path=Path(batching['manifest'])
    assert read(batching['seal'])['artifacts'][component_path.relative_to(ROOT).as_posix()]==sha(component_path)
    groups=read(component_path)['components']
    lookup={image:index for index,images in enumerate(groups) for image in images}
    assert len(lookup)==sum(len(images) for images in groups)
    training=read(paths['train'])
    assert not {lookup[r['image_id']] for r in truth}&{lookup[r['image_id']] for r in training}
    identity=lambda rows: [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]
    expected=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in truth]
    results={}
    manual={}
    for mode in MODES:
        path=OUT/(mode+'.json')
        assert sha(path)==report['prediction_sha256'][mode]
        result=read(path)
        rows=result['records']
        assert identity(rows)==expected and len(rows)==1432
        metrics=compute_metrics(rows,fracture_threshold=.5)
        assert metrics==result['metrics']==report['results'][mode]['metrics']
        canonical=[r for r in rows if r['question']==QUESTION]
        assert len(canonical)==len({r['image_id'] for r in canonical})==321
        assert all('p_yes' in r and math.isfinite(r['p_yes']) and 0<=r['p_yes']<=1 for r in canonical)
        assert compute_metrics(canonical,fracture_threshold=.5)==result['canonical_metrics']
        assert result['canonical_metrics']==report['results'][mode]['canonical_metrics']
        opened=[r for r in rows if r['answer_type']=='OPEN']
        closed=[r for r in rows if r['answer_type']=='CLOSED']
        assert len(opened)==628 and len(closed)==804
        op=sum(normalize_answer(r['pred'])==normalize_answer(r['gt']) for r in opened)
        cl=sum(normalize_yes_no(r['pred'])==normalize_yes_no(r['gt']) for r in closed)
        assert metrics['open_acc']==op/628 and metrics['closed_acc']==cl/804
        assert abs(metrics['overall_acc']-(op+cl)/1432)<1e-15
        labels=np.array([int(normalize_yes_no(r['gt'])=='yes') for r in canonical])
        scores=np.array([r['p_yes'] for r in canonical])
        assert int(labels.sum())==57 and len(labels)-int(labels.sum())==264
        predicted=scores>=.5
        assert all(normalize_yes_no(r['pred'])==('yes' if p else 'no') for r,p in zip(canonical,predicted))
        tp=int((predicted & (labels==1)).sum());tn=int((~predicted & (labels==0)).sum())
        assert metrics['fracture_sensitivity']==tp/57 and metrics['fracture_specificity']==tn/264
        assert metrics['fracture_balanced_accuracy']==.5*(tp/57+tn/264)
        assert metrics['fracture_auroc']==float(roc_auc_score(labels,scores))
        assert metrics['fracture_auprc']==float(average_precision_score(labels,scores))
        assert metrics['fracture_brier']==float(brier_score_loss(labels,scores))
        bins=np.minimum((scores*10).astype(int),9)
        ece=sum(float(np.mean(bins==b))*abs(float(scores[bins==b].mean())-float(labels[bins==b].mean()))
                for b in range(10) if np.any(bins==b))
        assert abs(ece-metrics['fracture_ece10'])<1e-15
        c=read(protocol['configs'][mode]['path'])
        target=ROOT/c['checkpoint_dir']/c['run_name']
        completion=read(PARENT/(mode+'_completion.json'))
        assert sha(target/'last.pt')==completion['last_sha256']==result['checkpoint_sha256']
        assert sha(target/'optimizer.pt')==completion['optimizer_sha256']
        history=read(target/'history.json')[-1]
        differences={k[4:]:metrics[k[4:]]-v for k,v in history.items() if k.startswith('val_') and k!='val_loss'}
        assert differences==result['trainer_metric_differences']
        assert all(v==0 for v in differences.values())
        assert result['weights_unchanged']
        manual[mode]={'QA_correct':op+cl,'OPEN_correct':op,'CLOSED_correct':cl,
                      'TP':tp,'FN':57-tp,'TN':tn,'FP':264-tn}
        results[mode]=rows
    assert identity(results[MODES[0]])==identity(results[MODES[1]])
    group_ids=sorted({lookup[r['image_id']] for r in truth})
    assert len(group_ids)==314
    group_rows=[[i for i,r in enumerate(truth) if lookup[r['image_id']]==group] for group in group_ids]
    keys=list(report['paired_bootstrap']['paired_intervals'])
    assert keys==['open_acc','overall_acc','closed_acc','fracture_auroc','fracture_auprc',
                  'fracture_sensitivity','fracture_specificity','fracture_balanced_accuracy',
                  'fracture_brier','fracture_ece10']
    assert report['paired_bootstrap']['seed']==plan['bootstrap_seed']==20261021
    assert report['paired_bootstrap']['resamples']==plan['bootstrap_resamples']==500
    rng=np.random.default_rng(20261021)
    draws={k:[] for k in keys}
    for _ in range(500):
        selected=rng.integers(0,len(group_ids),size=len(group_ids))
        indices=[i for position in selected for i in group_rows[position]]
        metrics={m:compute_metrics([rows[i] for i in indices],fracture_threshold=.5) for m,rows in results.items()}
        for key in keys:
            if all(key in metric for metric in metrics.values()):
                difference=metrics['sequence_uniform'][key]-metrics['token_mean'][key]
                if math.isfinite(difference):
                    draws[key].append(difference)
    for key,values in draws.items():
        expected_interval=np.quantile(values,[.025,.975]).tolist()
        observed=report['paired_bootstrap']['paired_intervals'][key]
        assert observed['valid_resamples']==len(values)==500
        assert observed['interval95']==expected_interval,(key,observed,expected_interval)
    checked={'passed':True,'report_sha256':sha(OUT/'report.json'),'plan_sha256':sha(OUT/'evaluation_plan.json'),
        'auditor_sha256':sha(__file__),'both1432identity_order_and_metrics_recomputed':True,
        'canonical321_manual_counts_sklearn_scores_and_ECE_recomputed':True,
        'both_physical_LAST_optimizer_hashes_checked':True,'all_trainer_metric_differences_exact_zero':True,
        'component_bootstrap_seed20261021_500resamples_all10intervals_exact':True,
        'neural_model_instantiations':0,'model_predictions':0,'optimizer_updates':0,
        'replay_report_binding_checked_NOT_additional_neural_replay':True,'manual_counts':manual,
        'elapsed_seconds':time.perf_counter()-started,'calibration_test_used':False,
        'development_informed_single_training_seed':True,'step3_complete':False,'production_released':False}
    destination.write_text(json.dumps(checked,indent=2)+'\n')
    print(json.dumps(checked))


if __name__=='__main__':
    audit()
