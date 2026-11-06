"""Audit existing TRAIN question-type annotations versus frozen heuristics.

No model, inference, new labels, validation inspection, or runtime changes.
"""
import json
from pathlib import Path
from src.utils import is_choice_question,is_closed_question,is_polar_question
from experiments.run_T_bounded_current_v3_1 import data_and_schedule,ROOT
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def main():
    output=ROOT/'refactor_v2/reports/TRAIN_question_routing_audit_v3_1.json'
    assert not output.exists()
    from src.protocol import records_of
    _,datasets,manifests=data_and_schedule();rows=records_of(datasets['train'])
    questions={}
    for row in rows:
        entry=questions.setdefault(row['question'],{'TRAIN_count':0,'existing_GT_answer_type_counts':{},
            'heuristic_choice':is_choice_question(row['question']),
            'heuristic_polar':is_polar_question(row['question']),
            'heuristic_closed':is_closed_question(row['question'])})
        entry['TRAIN_count']+=1
        kind=row['answer_type'];entry['existing_GT_answer_type_counts'][kind]=entry['existing_GT_answer_type_counts'].get(kind,0)+1
    excluded_polar={q:d for q,d in questions.items() if d['heuristic_polar'] and not d['heuristic_closed']
                    and d['existing_GT_answer_type_counts'].get('CLOSED',0)>0}
    report={'scope':'Existing1173TRAIN question-type metadata versus frozen routing heuristics; not quality inference',
        'source_script_sha256':sha(__file__),'frozen_utils_sha256':sha(ROOT/'refactor_v2/src/utils.py'),
        'TRAIN_manifest':manifests['train'],'TRAIN_rows':len(rows),'unique_questions':len(questions),
        'questions':questions,'existing_CLOSED_polar_questions_excluded_by_choice_heuristic':excluded_polar,
        'interpretation':'Choice heuristic scans any or/hay/hoac connector; excluded polar questions fall back to learned type probability, not guaranteed OPEN. Inspect actual trace before claiming route error.',
        'evidence_actual_24TRAIN_trace':'T_routes_stratified24_TRAIN_v3_1/independent_artifact_audit.json',
        'new_labels':False,'model_instantiations':0,'training_updates':0,
        'modelval_calibration_or_test_inference_used':False,'runtime_changed':False,
        'limitations':['Existing annotations are audited, not replaced or used as runtime inputs',
                       'TRAIN heuristic conflict alone is not an accuracy improvement or a general-purpose grammar fix'],
        'long_training_authorized':False}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(excluded_polar,indent=2))


if __name__=='__main__':main()
