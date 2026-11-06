"""No-model verification of full1432 fixed-checkpoint coverage results."""
import json
from pathlib import Path
from experiments import evaluate_full_modelval_coverage_v3_1 as experiment


def main():
    output=experiment.OUT/'independent_artifact_audit.json';assert not output.exists()
    protocol,full,indices,lookup=experiment.validate()
    report=json.loads((experiment.OUT/'report.json').read_text())
    progress=json.loads((experiment.OUT/'progress.json').read_text())
    replay=json.loads((experiment.OUT/'fresh_process_replay.json').read_text())
    assert progress['status']=='complete_verified_development_coverage_not_confirmation'
    assert experiment.sha(experiment.OUT/'report.json')==progress['report_sha256']
    assert experiment.sha(experiment.OUT/'protocol.json')==report['protocol_sha256']
    assert experiment.sha(experiment.OUT/'fresh_process_replay.json')==report['fresh_process_replay_sha256']
    assert replay['passed'] and replay['all1432_records_probabilities_metrics_and_coverage_exact_each_mode']
    assert replay['checkpoint_sha256']==protocol['checkpoint_sha256']
    assert report['training_updates']==0 and not report['calibration_or_test_used']
    assert not report['original142_primary_screen_redefined']
    expected=experiment.core.records_of(full)
    identities=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
    results={};correct={}
    for mode in experiment.MODES:
        path=experiment.OUT/(mode+'.json')
        assert experiment.sha(path)==report['prediction_sha256'][mode]
        result=json.loads(path.read_text());results[mode]=result
        assert result['weights_unchanged'] and len(result['full1432']['records'])==1432
        rows=result['full1432']['records']
        assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==identities
        assert result['canonical321']['records']==[rows[i] for i in indices]
        for endpoint in ('full1432','canonical321'):
            assert experiment.core.compute_metrics(result[endpoint]['records'],fracture_threshold=.5)==result[endpoint]['metrics']==report['results'][mode][endpoint]
        assert experiment.coverage(rows)==result['coverage']==report['results'][mode]['coverage']
        assert sum(v['n'] for v in result['coverage']['per_question'].values())==1432
        assert sum(v['n'] for v in result['coverage']['TRAIN_answer_frequency_strata'].values())==1432
        correct[mode]={'full1432_correct':sum(experiment.correct(r) for r in rows),
            'OPEN_correct':sum(experiment.correct(r) for r in rows if r['answer_type']=='OPEN'),
            'OPEN_total':sum(r['answer_type']=='OPEN' for r in rows)}
    qa={m:r['full1432']['records'] for m,r in results.items()}
    clinical={m:r['canonical321']['records'] for m,r in results.items()}
    assert experiment.core.bootstrap(qa,'sequence_rarity',lookup,protocol['full_QA_metrics'])==report['full_QA_component_bootstrap']
    assert experiment.core.bootstrap(clinical,'sequence_rarity',lookup,protocol['clinical_metrics'])==report['clinical_component_bootstrap']
    audit={'passed':True,'source_script_sha256':experiment.sha(Path(__file__).resolve()),
        'report_sha256':experiment.sha(experiment.OUT/'report.json'),
        'all_protocol_source_checkpoint_prediction_replay_hashes_verified':True,
        'all1432_identities_321_subset_metrics_coverage_and500_component_bootstraps_recomputed':True,
        'correct_counts':correct,'model_instantiations':0,'training_updates':0,
        'calibration_or_test_used':False,'original142_primary_screen_redefined':False,
        'independent_confirmation_claimed':False,'long_training_authorized':False,
        'step3_complete':False,'Final_Solution_selected':False,'production_released':False}
    experiment.dump(output,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
