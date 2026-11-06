"""Independent no-model audit of full sealed TRAIN error diagnosis."""
import json
from collections import Counter
from src.utils import normalize_answer
from experiments import audit_B_fullTRAIN_errors_v3_1 as task


def main():
    output=task.OUT/'independent_artifact_audit.json';assert not output.exists()
    protocol=json.loads((task.OUT/'protocol.json').read_text())
    report=json.loads((task.OUT/'report.json').read_text())
    replay=json.loads((task.OUT/'fresh_process_replay.json').read_text())
    state=json.loads((task.OUT/'progress.json').read_text())
    assert state['status']=='complete_TRAIN_diagnosis_verified'
    assert task.sha(task.OUT/'report.json')==state['report_sha256']
    assert task.sha(task.OUT/'protocol.json')==report['protocol_sha256']
    for path,digest in protocol['dependency_sha256'].items():assert task.sha(path)==digest
    assert task.sha(task.CHECKPOINT)==protocol['checkpoint_sha256']==report['checkpoint_sha256']==replay['checkpoint_sha256']
    assert task.sha(task.OUT/'TRAIN_predictions.json')==report['predictions_sha256']==replay['predictions_sha256']
    assert task.sha(task.OUT/'fresh_process_replay.json')==report['fresh_process_replay_sha256']
    assert replay['passed'] and replay['all1173answers_probabilities_routes_exact']
    _,dataset=task.validate();metadata=task.records_of(dataset)
    rows=json.loads((task.OUT/'TRAIN_predictions.json').read_text())
    assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in metadata]
    assert task.summarize(rows)==report['summary']
    correct=sum(task.correct(r) for r in rows);assert correct==943 and len(rows)==1173
    frequencies=Counter(normalize_answer(r['gt']) for r in rows)
    rare=[r for r in rows if frequencies[normalize_answer(r['gt'])]<=5]
    assert len(rare)==72 and sum(task.correct(r) for r in rare)==10
    assert all(r['answer_type']=='OPEN' for r in rare)
    assert all(r['answer_type']==r['actual_predicted_route'] for r in rows)
    assert not report['modelval_calibration_test_inference_used'] and report['training_updates']==0
    dump={'passed':True,'source_script_sha256':task.sha(__file__),'report_sha256':task.sha(task.OUT/'report.json'),
        'all_hashes_manifests1173row_identities_scores_strata_and_errors_recomputed_exact':True,
        'fresh_process_replay_certificate_verified':True,'correct':943,'n':1173,'errors':230,
        'rare_answer_rows':72,'rare_answer_correct':10,'all_rare_answer_rows_are_OPEN':True,
        'actual_route_matches_GTtype_all1173':True,'model_instantiations':0,'training_updates':0,
        'scope':'TRAIN diagnostic only, not generalization or clinical efficacy',
        'new_clinical_labels':False,'long_training_authorized':False,'step3_complete':False}
    task.dump(output,dump);print(json.dumps(dump,indent=2))


if __name__=='__main__':main()
