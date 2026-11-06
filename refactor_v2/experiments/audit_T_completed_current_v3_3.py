"""Read-only independent artifact/arithmetic audit of completed T pilots.

Recompute saved-record metrics and grouped bootstrap; no model or fitting.
"""
import json
import math
from pathlib import Path
from experiments.evaluate_T_bounded_current_v3_3 import core,compare_QA_metrics


def main():
    output = core.OUT/'independent_artifact_audit.json'
    assert not output.exists(), 'Preserve prior audit'
    plan = json.loads((core.OUT/'evaluation_plan.json').read_text())
    progress = json.loads((core.OUT/'progress.json').read_text())
    report = json.loads((core.OUT/'report.json').read_text())
    replay = json.loads((core.OUT/'fresh_process_replay.json').read_text())
    assert progress['status']=='complete_verified_development_only_NOTfull_step3'
    assert core.sha(core.OUT/'report.json')==progress['report_sha256']
    assert core.sha(core.OUT/'evaluation_plan.json')==report['evaluation_plan_sha256']
    for path,digest in plan['dependencies_sha256'].items():
        assert core.sha(path)==digest, path
    assert core.sha(core.OUT/'fresh_process_replay.json')==report['fresh_process_replay_sha256']
    assert replay['passed'] and replay['all_records_and_metrics_exact']
    assert replay['checkpoint_sha256']==report['checkpoint_sha256']
    assert all(v==0 for v in replay['max_probability_error'].values())
    assert report['training_updates']==0 and not report['calibration_or_test_used']
    qa,full,indices,lookup,_ = core.cohorts()
    assert core.manifest(qa)==plan['QA_manifest'] and core.manifest(full)==plan['full_modelval_manifest']
    baseline = json.loads((core.BASE/'independent_evaluation.json').read_text())
    results = {};confusions = {}
    for style in core.STYLES:
        path = core.OUT/(style+'.json')
        assert core.sha(path)==report['prediction_sha256'][style]
        result = json.loads(path.read_text());results[style]=result
        completion = json.loads((core.FOLDER/(style+'_completion.json')).read_text())
        assert completion['last_sha256']==report['checkpoint_sha256'][style]
        checkpoint = core.ROOT/'refactor_v2/checkpoints'/('T_'+style+'_seed42_bounded_current_v3_1')/'last.pt'
        assert core.sha(checkpoint)==completion['last_sha256']
        for endpoint,value in result.items():
            assert value['weights_unchanged']
            metrics = core.compute_metrics(value['records'],fracture_threshold=.5)
            assert metrics==value['metrics']==report['results'][style][endpoint]
            expected = [core.records_of(full)[i] for i in indices] if endpoint=='fracture321' else core.records_of(qa)
            assert [(r['image_id'],r['question'],r['gt']) for r in value['records']]==[(r['image_id'],r['question'],r['answer']) for r in expected]
        compare_QA_metrics(result['QA142']['metrics'],completion['history'][-1])
        counts = {k:0 for k in ('TP','FN','TN','FP')}
        for row in result['fracture321']['records']:
            assert math.isfinite(row['p_yes']) and 0<=row['p_yes']<=1
            pred = row['p_yes']>=.5;truth=row['gt'].strip().lower()=='yes'
            assert row['pred'].strip().lower()==('yes' if pred else 'no')
            counts[('TP' if pred else 'FN') if truth else ('FP' if pred else 'TN')]+=1
        assert counts['TP']+counts['FN']==57 and counts['TN']+counts['FP']==264
        confusions[style]=counts
    qa_endpoints = {s:r['QA142']['records'] for s,r in results.items()}
    qa_endpoints['raw_B_seed42']=baseline['pilot142_QA']['fracatlas_B_roco_scale_pilot_v3/correct_image']['records']
    clinical_endpoints = {s:r['fracture321']['records'] for s,r in results.items()}
    clinical_endpoints['raw_B_seed42']=baseline['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']['records']
    assert core.bootstrap(qa_endpoints,'clinical',lookup,['overall_acc','open_acc'])==report['QA_component_bootstrap']
    assert core.bootstrap(clinical_endpoints,'clinical',lookup,['fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity'])==report['clinical_component_bootstrap']
    # Descriptive error overlap on already evaluated data, no extra hypothesis
    # test or inference. Never interpret it as an untouched holdout.
    from src.utils import normalize_answer,normalize_yes_no
    def correct(row):
        norm = normalize_yes_no if row['answer_type']=='CLOSED' else normalize_answer
        return norm(row['pred'])==norm(row['gt'])
    primary = results['clinical']['QA142']['records']
    secondary = results['clinical']['secondary_inference_paraphrase142']['records']
    transitions = {'both_correct':0,'correct_to_wrong':0,'wrong_to_correct':0,'both_wrong':0}
    for original,changed in zip(primary,secondary):
        a,b=correct(original),correct(changed)
        transitions['both_correct' if a and b else 'correct_to_wrong' if a else 'wrong_to_correct' if b else 'both_wrong']+=1
    audit = {'passed':True,'scope':'No-model saved-artifact development audit, not independent clinical confirmation',
        'source_script_sha256':core.sha(__file__),'report_sha256':core.sha(core.OUT/'report.json'),
        'all_source_manifest_checkpoint_prediction_and_replay_hashes_passed':True,
        'all_saved_record_metrics_and_history_keysets_recomputed':True,
        'all500_seed20261020_QA32_and_clinical314_component_bootstraps_exact':True,
        'canonical321_confusions':confusions,'secondary_predeclared_robustness_transitions':transitions,
        'model_instantiations':0,'training_updates':0,'calibration_or_test_used':False,
        'T_semantic_benefit_established':False,'long_training_authorized':False,'step3_complete':False}
    core.dump(output,audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
