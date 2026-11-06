"""No-model independent arithmetic and binding audit; never fit or infer."""
import json
import math
from pathlib import Path
from experiments import evaluate_B_sequence_loss_pilots_v3_2 as wrapper

evaluation=wrapper.evaluation
core=evaluation.core
OUT=evaluation.OUT


def main():
    output=OUT/'independent_artifact_audit.json';assert not output.exists()
    plan=json.loads((OUT/'evaluation_plan.json').read_text())
    progress=json.loads((OUT/'progress.json').read_text())
    report=json.loads((OUT/'report.json').read_text())
    replay=json.loads((OUT/'fresh_process_replay.json').read_text())
    assert progress['status']=='complete_verified_development_only_NOTfull_step3'
    assert evaluation.sha(OUT/'report.json')==progress['report_sha256']
    assert evaluation.sha(OUT/'evaluation_plan.json')==report['evaluation_plan_sha256']
    for path,digest in plan['dependencies_sha256'].items():assert evaluation.sha(path)==digest,path
    assert evaluation.sha(OUT/'fresh_process_replay.json')==report['fresh_process_replay_sha256']
    assert replay['passed'] and replay['all_records_and_metrics_exact']
    assert replay['checkpoint_sha256']==report['checkpoint_sha256']
    assert report['training_updates']==0 and not report['calibration_or_test_used']
    qa,full,indices,lookup,_=core.cohorts()
    assert core.manifest(qa)==plan['QA_manifest'] and core.manifest(full)==plan['full_modelval_manifest']
    baseline=json.loads((core.BASE/'independent_evaluation.json').read_text())
    execution=json.loads((evaluation.FOLDER/'plan.json').read_text())
    results={};confusions={}
    for mode in evaluation.MODES:
        path=OUT/(mode+'.json');assert evaluation.sha(path)==report['prediction_sha256'][mode]
        results[mode]=json.loads(path.read_text())
        completion=json.loads((evaluation.FOLDER/(mode+'_completion.json')).read_text())
        cfg=json.loads(Path(execution['configs'][mode]['path']).read_text())
        checkpoint=evaluation.ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
        assert evaluation.sha(checkpoint)==completion['last_sha256']==report['checkpoint_sha256'][mode]
        for endpoint,value in results[mode].items():
            assert value['weights_unchanged']
            assert core.compute_metrics(value['records'],fracture_threshold=.5)==value['metrics']==report['results'][mode][endpoint]
            expected=[core.records_of(full)[i] for i in indices] if endpoint=='fracture321' else core.records_of(qa)
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in value['records']]==[
                (r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
        evaluation.shared.compare_QA_metrics(results[mode]['QA142']['metrics'],completion['history'][-1])
        counts={k:0 for k in ('TP','FN','TN','FP')}
        for row in results[mode]['fracture321']['records']:
            assert math.isfinite(row['p_yes']) and 0<=row['p_yes']<=1
            predicted=row['p_yes']>=.5;truth=row['gt'].strip().lower()=='yes'
            assert row['pred'].strip().lower()==('yes' if predicted else 'no')
            counts[('TP' if predicted else 'FN') if truth else ('FP' if predicted else 'TN')]+=1
        assert counts['TP']+counts['FN']==57 and counts['TN']+counts['FP']==264
        confusions[mode]=counts
    reference={'QA142':baseline['pilot142_QA']['fracatlas_B_roco_scale_pilot_v3/correct_image'],
        'fracture321':baseline['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']}
    endpoints={name:{mode:result[name]['records'] for mode,result in results.items()} for name in reference}
    for name in endpoints:endpoints[name]['raw_B_seed42']=reference[name]['records']
    assert core.bootstrap(endpoints['QA142'],'sequence_rarity',lookup,plan['QA_metrics'])==report['QA_component_bootstrap']
    assert core.bootstrap(endpoints['fracture321'],'sequence_rarity',lookup,plan['clinical_metrics'])==report['clinical_component_bootstrap']
    audit={'passed':True,'scope':'No-model saved artifact and arithmetic audit, inspected development only',
        'source_script_sha256':evaluation.sha(Path(__file__).resolve()),
        'report_sha256':evaluation.sha(OUT/'report.json'),
        'all_source_manifest_checkpoint_prediction_replay_hashes_passed':True,
        'all_record_identities_metrics_and_native_history_fields_recomputed':True,
        'all500_seed20261020_QA32_and_clinical314_component_bootstraps_exact':True,
        'canonical321_confusions':confusions,'model_instantiations':0,'training_updates':0,
        'calibration_or_test_used':False,'long_training_authorized':False,
        'step3_complete':False,'Final_Solution_selected':False,'production_released':False}
    evaluation.dump(output,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
