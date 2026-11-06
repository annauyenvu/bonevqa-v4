"""No-model saved-record audit of both TRAIN diagnostic arms."""
import json
from collections import Counter
from pathlib import Path
from experiments import diagnose_B_loss_TRAIN_v3_1 as experiment

core=experiment.core
OUT=experiment.OUT


def main():
    output=OUT/'independent_artifact_audit.json';assert not output.exists()
    protocol,dataset=experiment.validate()
    progress=json.loads((OUT/'progress.json').read_text())
    report=json.loads((OUT/'report.json').read_text())
    replay=json.loads((OUT/'fresh_process_replay.json').read_text())
    assert progress['status']=='complete_verified_TRAIN_diagnosis_not_quality'
    assert core.sha(OUT/'report.json')==progress['report_sha256']
    assert core.sha(OUT/'protocol.json')==report['protocol_sha256']
    assert core.sha(OUT/'fresh_process_replay.json')==report['fresh_process_replay_sha256']
    assert replay['passed'] and replay['all1173_answers_probabilities_routes_exact_each_mode']
    assert report['training_updates']==0 and not report['reserved_role_inference']
    raw=json.loads((experiment.BASELINE/'TRAIN_predictions.json').read_text())
    original_summary=json.loads((experiment.BASELINE/'report.json').read_text())['summary']
    assert report['raw_B_TRAIN_summary']==original_summary==core.summarize(raw)
    expected=core.records_of(dataset)
    identities=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
    assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in raw]==identities
    frequencies=Counter(core.normalize_answer(r['gt']) for r in raw)
    rare_indices=[i for i,r in enumerate(raw) if frequencies[core.normalize_answer(r['gt'])]<=5]
    assert len(rare_indices)==72 and all(raw[i]['answer_type']=='OPEN' for i in rare_indices)
    comparisons={}
    for mode in experiment.MODES:
        path=OUT/mode/'TRAIN_predictions.json'
        assert core.sha(path)==report['prediction_sha256'][mode]
        rows=json.loads(path.read_text());summary=report['summaries'][mode]
        assert len(rows)==1173
        assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==identities
        assert core.summarize(rows)==summary
        assert core.compute_metrics(rows,fracture_threshold=.5)==summary['metrics']
        all_correct=sum(core.correct(row) for row in rows)
        rare_correct=sum(core.correct(rows[i]) for i in rare_indices)
        assert sum(v['correct'] for v in summary['per_GTtype'].values())==all_correct
        assert summary['TRAIN_answer_frequency_strata']['TRAIN_answer_count<=5']=={
            'n':72,'correct':rare_correct,'errors':72-rare_correct}
        transitions={key:0 for key in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong')}
        rare_transitions=dict.fromkeys(transitions,0)
        rare_set=set(rare_indices)
        for i,(reference,current) in enumerate(zip(raw,rows)):
            first,second=core.correct(reference),core.correct(current)
            key='both_correct' if first and second else 'correct_to_wrong' if first else 'wrong_to_correct' if second else 'both_wrong'
            transitions[key]+=1
            if i in rare_set:rare_transitions[key]+=1
        assert sum(transitions.values())==1173 and sum(rare_transitions.values())==72
        assert all_correct==sum(core.correct(r) for r in raw)+transitions['wrong_to_correct']-transitions['correct_to_wrong']
        comparisons[mode]={'TRAIN_correct':all_correct,'TRAIN_total':1173,
            'rare_correct':rare_correct,'rare_total':72,'transitions_vs_rawB':transitions,
            'rare_transitions_vs_rawB':rare_transitions,
            'all1173_actual_routes_match_GTtype':all(r['actual_predicted_route']==r['answer_type'] for r in rows)}
    audit={'passed':True,'source_script_sha256':core.sha(Path(__file__).resolve()),
        'report_sha256':core.sha(OUT/'report.json'),
        'all_protocol_dependency_checkpoint_prediction_replay_hashes_verified':True,
        'all1173_record_identities_and_summary_metrics_strata_recomputed_each_mode':True,
        'raw_B_TRAIN_correct':sum(core.correct(r) for r in raw),'raw_B_rare_correct':10,
        'comparisons':comparisons,'model_instantiations':0,'training_updates':0,
        'reserved_role_inference':False,'clinical_or_generalization_gain_claimed':False,
        'global_rare_stratum_is_question_conditioned_loss_weight_group':False,
        'long_training_authorized':False,'step3_complete':False}
    core.dump(output,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
