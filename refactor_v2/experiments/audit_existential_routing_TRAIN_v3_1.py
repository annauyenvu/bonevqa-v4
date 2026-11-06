"""No-model verification of the isolated routing correction diagnostic."""
import json
from src.utils import compute_metrics,is_closed_question
from experiments import check_existential_routing_TRAIN_v3_1 as task
from experiments.existential_polar_routing_adapter_v3_1 import candidate_closed_question,test_contract


def main():
    out=task.OUT;output=out/'independent_artifact_audit.json';assert not output.exists()
    protocol=json.loads((out/'protocol.json').read_text());report=json.loads((out/'report.json').read_text())
    progress=json.loads((out/'progress.json').read_text())
    assert progress['status']=='complete_TRAIN_diagnostic_only'
    assert task.sha(out/'report.json')==progress['report_sha256']
    assert task.sha(out/'protocol.json')==report['protocol_sha256']
    for path,digest in protocol['dependency_sha256'].items():assert task.sha(path)==digest
    assert {k:task.sha(v) for k,v in task.CHECKPOINTS.items()}==protocol['checkpoint_sha256']
    completions={'B_ROCO':task.ROOT/'refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_completion.json',
                 'T_clinical':task.selection.prior.FOLDER/'clinical_completion.json'}
    assert all(json.loads(p.read_text())['last_sha256']==protocol['checkpoint_sha256'][k] for k,p in completions.items())
    assert test_contract()==protocol['grammar_tests']
    ds,manifests,batches,_=task.selection.select_rows()
    records=[task.selection.prior.records_of(ds)[i] for b in batches for i in b]
    assert manifests==protocol['actual_manifests'] and records==protocol['selected_TRAIN_records']
    changed={}
    for name,value in report['results'].items():
        assert json.loads((out/(name+'.json')).read_text())==value
        assert value['head_tensors_exact_all_conditions'] and value['weights_config_unchanged']
        assert value['records']['original']==value['records']['restored']
        assert value['original_runtime_predictions_restored_exact']
        for condition,rows in value['records'].items():
            assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in records]
            assert compute_metrics(rows,fracture_threshold=.5)==value['metrics'][condition]
        changes=[]
        for old,new in zip(value['records']['original'],value['records']['adapter']):
            if candidate_closed_question(old['question'])==is_closed_question(old['question']):
                assert old==new
            if old!=new:changes.append({'image_id':old['image_id'],'question':old['question'],
                'old_route':old['actual_predicted_route'],'new_route':new['actual_predicted_route'],
                'old_answer':old['pred'],'new_answer':new['pred']})
        changed[name]=changes
    assert report['two_fresh_constructors_replay_exact'] and report['training_updates']==0
    assert not report['modelval_calibration_test_inference_used']
    audit={'passed':True,'source_script_sha256':task.sha(__file__),
        'report_sha256':task.sha(out/'report.json'),'all_dependency_manifest_checkpoint_and_completion_hashes_verified':True,
        'all24_record_identities_metrics_original_restore_and_unaffected_predictions_exact':True,
        'grammar_tests':protocol['grammar_tests'],'record_changes':changed,
        'observed_TRAIN_accuracy_improvement':False,'model_instantiations':0,'training_updates':0,
        'independent_process_model_replay_claimed':False,
        'scope':'No-model audit of TRAIN-only experiment, not clinical efficacy or production release',
        'calibration_or_test_used':False,'long_training_authorized':False,'step3_complete':False}
    task.dump(output,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
