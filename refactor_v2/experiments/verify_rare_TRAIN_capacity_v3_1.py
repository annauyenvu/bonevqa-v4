"""No-model artifact audit; does not claim a second training replay."""
import json
import math
from pathlib import Path
from experiments import gate_rare_TRAIN_capacity_v3_1 as gate


def main():
    output=gate.OUT/'independent_artifact_audit.json';assert not output.exists()
    protocol,dataset=gate.validate()
    progress=json.loads((gate.OUT/'progress.json').read_text())
    report=json.loads((gate.OUT/'report.json').read_text())
    assert progress['status']=='complete_capacity_diagnostic_not_quality'
    assert progress['report_sha256']==gate.sha(gate.OUT/'report.json')
    assert report['protocol_sha256']==gate.sha(gate.OUT/'protocol.json')
    assert report['all263_initial_tensors_and_initial_predictions_equal']
    assert report['total_disposable_updates']==128 and report['quality_training_updates']==0
    assert not report['weights_saved'] and not report['reserved_role_inference']
    assert not list(gate.OUT.rglob('*.pt')),'Do not save diagnostic weights as a quality candidate'
    expected=[gate.records_of(dataset)[i] for i in protocol['TRAIN_indices']]
    expected_ids=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
    comparison={}
    for mode in gate.MODES:
        result=json.loads((gate.OUT/(mode+'.json')).read_text())
        assert result==report['results'][mode]
        assert result['disposable_updates']==64 and result['smoke_weights_discarded'] and not result['weights_saved']
        assert set(result['snapshots'])=={'0','16','32','64'}
        for step,snapshot in result['snapshots'].items():
            rows=snapshot['records'];assert len(rows)==snapshot['n']==16
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==expected_ids
            assert sum(gate.correct(row) for row in rows)==snapshot['correct']
            assert math.isfinite(snapshot['teacher_forced_token_CE_mean']) and snapshot['teacher_forced_token_CE_mean']>=0
        assert result['capacity_rule_passed_at_LAST64']==(result['snapshots']['64']['correct']==16)
        assert [g['step'] for g in result['tracked_answer_gradients']]==[1,32,64]
        for record in result['tracked_answer_gradients']:
            assert len(record['answer_gradient_maxima'])==4
            assert all(math.isfinite(v) and v>0 for v in record['answer_gradient_maxima'].values())
        assert len(result['actual_selected_updates'])==4
        assert all(math.isfinite(v) and v>0 for v in result['actual_selected_updates'].values())
        comparison[mode]={'correct_at_snapshots':{s:v['correct'] for s,v in result['snapshots'].items()},
            'CE_at_snapshots':{s:v['teacher_forced_token_CE_mean'] for s,v in result['snapshots'].items()},
            'capacity_rule_passed_at_LAST64':result['capacity_rule_passed_at_LAST64']}
    assert report['results'][gate.MODES[0]]['snapshots']['0']==report['results'][gate.MODES[1]]['snapshots']['0']
    audit={'passed':True,'source_script_sha256':gate.sha(Path(__file__).resolve()),
        'report_sha256':gate.sha(gate.OUT/'report.json'),'protocol_and_dependency_hashes_verified':True,
        'all16_fixture_identities_snapshots_metrics_gradient_update_arithmetic_verified':True,
        'comparison':comparison,'all_smoke_weights_discarded_no_checkpoint_saved':True,
        'independent_training_or_trained_model_replay_claimed':False,
        'model_instantiations_for_artifact_audit':0,'new_training_updates_for_artifact_audit':0,
        'reserved_role_inference':False,'clinical_or_generalization_gain_claimed':False,
        'long_training_authorized':False,'step3_complete':False}
    gate.dump(output,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
