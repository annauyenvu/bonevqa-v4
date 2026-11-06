"""No-model verification of the 24-TRAIN-row explanatory diagnostic."""
import json
from pathlib import Path
from experiments import diagnose_T_routes_stratified24_TRAIN_v3_1 as task


def main():
    out=task.OUT;destination=out/'independent_artifact_audit.json'
    assert not destination.exists()
    protocol=json.loads((out/'protocol.json').read_text())
    progress=json.loads((out/'progress.json').read_text())
    report=json.loads((out/'report.json').read_text())
    assert progress['status']=='complete_TRAIN_diagnostic_only'
    assert task.prior.sha(out/'report.json')==progress['report_sha256']
    assert task.prior.sha(out/'protocol.json')==report['protocol_sha256']
    for path,digest in protocol['dependency_sha256'].items():assert task.prior.sha(path)==digest
    assert task.prior.sha(task.prior.CHECKPOINT)==report['checkpoint_sha256']==protocol['checkpoint_sha256']
    ds,manifests,batches,lookup=task.select_rows()
    assert manifests==protocol['actual_manifests'] and batches==protocol['selected_indices_by_batch']
    indices=[i for batch in batches for i in batch]
    records=[task.prior.records_of(ds)[i] for i in indices]
    assert records==protocol['selected_TRAIN_records']
    assert len({lookup[r['image_id']] for r in records})==24
    assert len({r['question'] for r in records})==18
    assert sum(r['answer_type']=='CLOSED' for r in records)==sum(r['answer_type']=='OPEN' for r in records)==12
    predictions={};changes={}
    for number,batch in enumerate(batches):
        path=out/('batch_'+str(number)+'.json')
        assert task.prior.sha(path)==report['batch_report_sha256'][str(number)]
        saved=json.loads(path.read_text());value=saved['results']
        assert saved['TRAIN_indices']==batch and saved['fresh_constructor_replay_exact']
        assert value['weights_unchanged'] and value['normal_endpoint_controls_and_restored_predictions_exact']
        assert all(v['all_text_tokens_masks_fusion_and_heads_exact'] for v in value['fixed_biomedical_route_checks'].values())
        for condition,detail in value['conditions'].items():
            predictions.setdefault(condition,[]).extend(detail['predictions'])
            assert detail['all_captured_states_finite']
        changes[str(number)]=value['biomedical_paraphrase_only_max_abs_core_change']
    summary=task.summarize(predictions,records)
    assert summary==report['summary']
    actual_routes={}
    for condition,values in predictions.items():
        counts={}
        for record,prediction in zip(records,values):
            key=record['answer_type']+'_GT__'+prediction['answer_type']+'_predicted_route'
            counts[key]=counts.get(key,0)+1
        actual_routes[condition]=counts
    decoder_only_GTclosed_changes=[]
    for record,baseline,changed in zip(records,predictions['clinical__clinical'],predictions['clinical__clinical_paraphrase']):
        if record['answer_type']=='CLOSED' and baseline['answer']!=changed['answer']:
            assert baseline['answer_type']==changed['answer_type']=='OPEN'
            decoder_only_GTclosed_changes.append({'image_id':record['image_id'],'question':record['question'],
                'GT_type':'CLOSED','actual_predicted_route':'OPEN',
                'meaning':'Decoder-generated answer changed; identical closed-head tensors do not force this row onto the closed prediction route'})
    assert report['training_updates']==0 and not report['modelval_calibration_or_test_inference_used']
    audit={'passed':True,'source_script_sha256':task.prior.sha(__file__),
        'report_sha256':task.prior.sha(out/'report.json'),
        'selection_and_all_source_protocol_checkpoint_batch_report_hashes_verified':True,
        '24distinct_components_18questions_12CLOSED12OPEN_verified':True,
        'all_saved_metrics_and_transitions_recomputed_exact':True,
        'all_batch_restore_and_two_constructor_replay_checks_passed':True,
        'GT_type_vs_actual_prediction_route_counts':actual_routes,
        'decoder_only_text_changes_on_GT_CLOSED_rows':decoder_only_GTclosed_changes,
        'biomedical_only_max_deltas_by_batch':changes,'model_instantiations':0,
        'training_updates':0,'calibration_or_test_used':False,
        'scope':'TRAIN diagnostic only, not generalization, clinical efficacy or release',
        'long_training_authorized':False,'step3_complete':False}
    task.prior.dump(destination,audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
