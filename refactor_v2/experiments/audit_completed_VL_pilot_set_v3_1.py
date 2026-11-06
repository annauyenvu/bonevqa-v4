"""Seal verified bounded-pilot evidence; technical completion is not efficacy."""
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    folder=Path.cwd()/'refactor_v2/reports/vl_bounded_quality_v3_1'
    destination=folder/'completed_pilot_set_audit.json'
    assert not destination.exists() and not (folder/'ACTIVE.lock').exists()
    release=json.loads((folder/'execution_release.json').read_text())
    for path,digest in release['dependency_sha256'].items():assert sha(path)==digest
    results={}
    for arm in release['arms']:
        run=folder/arm
        report=json.loads((run/'independent_LAST296_verification.json').read_text())
        progress=json.loads((run/'progress.json').read_text())
        assert report['passed'] and report['global_step']==progress['global_step']==296
        assert report['answers_exact']==142 and report['max_p_yes_error']==report['max_metric_error']==0.
        assert report['all_trainable_weights_exact_and_unchanged'] and report['module_gradient_and_update_QC_passed']
        assert not report['calibration_or_test_used'] and not progress['calibration_or_test_used']
        assert sha(run/'last.pt')==report['checkpoint_sha256']==progress['last_sha256']
        assert sha(run/'optimizer.pt')==report['optimizer_sha256']==progress['optimizer_sha256']
        init=release['fresh_initializations'][arm]
        assert sha(init['path'])==init['sha256']
        history=json.loads((run/'history.json').read_text())
        assert [h['global_step'] for h in history]==[74,148,222,296]
        assert [h['stage'] for h in history]==([3]*4 if arm.startswith('V_') else [1,1,3,3])
        results[arm]={'independent_report_sha256':sha(run/'independent_LAST296_verification.json'),
                      'checkpoint_sha256':report['checkpoint_sha256'],'optimizer_sha256':report['optimizer_sha256'],
                      'train_seconds':report['train_wall_seconds'],'metrics':report['metrics'],
                      'verified_public_input_keys':report['public_input_keys']}
    contrasts={}
    for group in ('V','L'):
        path=folder/(group+'_LAST296_matched_comparison.json')
        pair=json.loads(path.read_text())
        for arm,digest in pair['reports_sha256'].items():assert digest==results[arm]['independent_report_sha256']
        contrasts[group]={'report_sha256':sha(path),'contrasts':pair['contrasts']}
    report={'passed_artifact_and_replay_audit':True,'scope':'five named seed42 bounded development pilots ONLY',
            'source_script_sha256':sha(__file__),'execution_release_sha256':sha(folder/'execution_release.json'),
            'results':results,'verified_arm_count':5,'optimizer_updates_per_arm':296,'total_pilot_updates':1480,
            'total_actual_train_wall_seconds':sum(r['train_seconds'] for r in results.values()),
            'matched_contrasts':contrasts,
            'scientific_gate_decisions':{
                'V':'local OPEN signal only; overall accuracy interval includes zero; no fracture AUROC/AP gain on32images; no expansion or efficacy release',
                'L':'real teacher below both same-phase controls on QA/OPEN; no positive semantic contribution established; no expansion or efficacy release',
                'P':'previous separate matched pilot did not establish semantic benefit; do not expand',
                'B':'provisional technical reference only; not Final Solution or clinical production model'},
            'limitations':['One downstream seed42, reused142QA32images5positives, not independent confirmation',
                           'V/B versus L not phase/capacity/detector-GT-exposure matched',
                           'Parameter/view/update matching does not certify equal FLOPs or walltime',
                           'Full321clinical development evaluation and honest independent/nested confirmation remain separate requirements'],
            'step3_complete':False,'long_training_authorized':False,'Final_Solution_selected':False,
            'production_released':False,'calibration_or_test_used':False}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'verified_arms':5,'total_updates':1480,'long_training_authorized':False}))


if __name__=='__main__':main()
