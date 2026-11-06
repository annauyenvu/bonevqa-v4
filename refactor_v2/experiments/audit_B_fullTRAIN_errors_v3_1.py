"""Full sealed pilot TRAIN error attribution, frozen B LAST296, zero updates.

No model-validation/calibration/test inference, threshold search or alternate
clinical labels. Fresh subprocess must replay every TRAIN answer/score/route.
"""
import argparse
from collections import Counter,defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from src.data import collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.utils import compute_metrics,normalize_answer,normalize_yes_no,set_seed
from experiments.run_T_bounded_current_v3_1 import data_and_schedule,hashes,ROOT
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha

BASE=ROOT/'refactor_v2/reports/fracatlas_scale_pilot_v3'
OUT=ROOT/'refactor_v2/reports/B_fullTRAIN_errors_v3_1'
CHECKPOINT=ROOT/'refactor_v2/checkpoints/fracatlas_B_roco_scale_pilot_v3/last.pt'
EXPECTED_LAST='94594edf64f8ef3fd7ae974e290b86900e186661d03c09e68d90e34c51ff3ddf'


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    check_environment();assert not OUT.exists()
    _,datasets,manifests=data_and_schedule()
    completion=BASE/'fracatlas_B_roco_scale_pilot_v3_completion.json'
    assert sha(CHECKPOINT)==json.loads(completion.read_text())['last_sha256']==EXPECTED_LAST
    sources={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for path in [Path(__file__),completion,BASE/'execution_release.json',
                 ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py',
                 ROOT/'refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py',
                 ROOT/'refactor_v2/experiments/vl_bounded_proposal_contract_v3_1.py']:
        sources[str(path.resolve())]=sha(path)
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'All1173sealed pilot TRAIN QA, diagnosis not quality confirmation',
        'checkpoint_sha256':EXPECTED_LAST,'dependency_sha256':sources,'actual_manifests':manifests,
        'TRAIN_QA':len(datasets['train']),'batch_size':4,'order':'original sealed TRAIN row order',
        'public_input_keys':['images','questions'],'routing':'unchanged frozen original runtime, adapter OFF',
        'threshold':.5,'GT_type_region_ROI_inputs':False,'training_updates':0,
        'first_constructor_seed':90210,'fresh_process_replay_seed':90211,
        'predeclared_summary':['per-question and GTtype error counts','actual model-route confusion',
            'TRAIN answer-frequency strata <=5 versus >5','literal token insertion/omission patterns only, not clinical-equivalence labels',
            'TRAIN-fitted per-question majority answer comparator, explicitly resubstitution-only'],
        'modelval_calibration_test_inference_allowed':False,'long_training_authorized':False})
    print('Prepared full1173TRAIN diagnosis; frozen B LAST296; no model loaded')


def validate():
    check_environment();protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest
    assert sha(CHECKPOINT)==protocol['checkpoint_sha256']==EXPECTED_LAST
    _,datasets,manifests=data_and_schedule();assert manifests==protocol['actual_manifests']
    import src.models.bonevqa as runtime
    from src.utils import is_closed_question
    assert runtime.is_closed_question is is_closed_question,'No experimental routing adapter allowed'
    return protocol,datasets['train']


@torch.no_grad()
def predict(dataset,seed,track):
    set_seed(seed);model=BoneVQAModel.load(str(CHECKPOINT),device='cpu').eval()
    assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
    assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
    assert model.cfg.get('closed_llm_weight',0)==0 and not model.cfg.get('closed_via_llm',False)
    before=hashes(model.trainable_state_dict());rows=[]
    for offset in range(0,len(dataset),4):
        batch=collate_fn([dataset[i] for i in range(offset,min(offset+4,len(dataset)))])
        predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
        assert len(predictions)==len(batch['questions'])
        for j,pred in enumerate(predictions):
            row={'image_id':batch['image_ids'][j],'question':batch['questions'][j],
                 'gt':batch['answers'][j],'answer_type':batch['answer_types'][j],
                 'pred':pred['answer'],'actual_predicted_route':pred['answer_type']}
            if 'p_yes' in pred:row['p_yes']=pred['p_yes']
            rows.append(row)
        if track:dump(OUT/'progress.json',{'status':'running_first_TRAIN_pass','pid':os.getpid(),'completed_QA':len(rows),'total_QA':len(dataset)})
    assert hashes(model.trainable_state_dict())==before and sha(CHECKPOINT)==EXPECTED_LAST
    del model;gc.collect();return rows


def correct(row):
    norm=normalize_yes_no if row['answer_type']=='CLOSED' else normalize_answer
    return norm(row['pred'])==norm(row['gt'])


def summarize(rows):
    question_counts=defaultdict(Counter);frequencies=Counter(normalize_answer(r['gt']) for r in rows)
    routes=Counter();by_question={};by_GTtype={};rare={};literal=Counter();errors=[]
    for row in rows:question_counts[row['question']][normalize_answer(row['gt'])]+=1
    # TRAIN-fit, TRAIN-score comparator only. Ties fixed lexicographically.
    majority={q:sorted(c,key=lambda answer:(-c[answer],answer))[0] for q,c in question_counts.items()}
    majority_correct=0
    for row in rows:
        ok=correct(row);question=row['question'];kind=row['answer_type']
        norm=normalize_yes_no if kind=='CLOSED' else normalize_answer
        majority_correct+=int(norm(majority[question])==norm(row['gt']))
        for dictionary,key in [(by_question,question),(by_GTtype,kind),
            (rare,'TRAIN_answer_count<=5' if frequencies[normalize_answer(row['gt'])]<=5 else 'TRAIN_answer_count>5')]:
            item=dictionary.setdefault(key,{'n':0,'correct':0,'errors':0});item['n']+=1
            item['correct']+=int(ok);item['errors']+=int(not ok)
        routes[kind+'_GT__'+row['actual_predicted_route']+'_predicted']+=1
        if not ok:
            category='other_literal_mismatch'
            a=normalize_answer(row['gt']).split();b=normalize_answer(row['pred']).split()
            if kind=='OPEN':
                if not b:category='empty_prediction'
                elif a and len(b)>len(a):
                    if any(b[start:start+len(a)]==a for start in range(len(b)-len(a)+1)):category='GT_tokens_contiguous_in_longer_prediction'
                if category=='other_literal_mismatch' and a and b and len(a)>len(b):
                    if any(a[start:start+len(b)]==b for start in range(len(a)-len(b)+1)):category='prediction_tokens_contiguous_in_longer_GT'
                literal[category]+=1
            errors.append({**row,'literal_pattern':category,'TRAIN_GT_answer_frequency':frequencies[normalize_answer(row['gt'])]})
    return {'metrics':compute_metrics(rows,fracture_threshold=.5),'per_question':by_question,'per_GTtype':by_GTtype,
        'GTtype_vs_actual_route':dict(routes),'TRAIN_answer_frequency_strata':rare,
        'OPEN_literal_mismatch_patterns_NOTclinical_labels':dict(literal),'error_records':errors,
        'TRAIN_resubstitution_question_majority_only':{'correct':majority_correct,'n':len(rows),
            'interpretation':'Answer labels fit and scored on SAME TRAIN; optimistic question-prior comparator, not generalization'}}


def verify():
    protocol,dataset=validate();expected=json.loads((OUT/'TRAIN_predictions.json').read_text())
    actual=predict(dataset,protocol['fresh_process_replay_seed'],False)
    assert actual==expected,'All1173TRAIN predictions/scores/routes must replay EXACTLY'
    dump(OUT/'fresh_process_replay.json',{'passed':True,'all1173answers_probabilities_routes_exact':True,
        'checkpoint_sha256':sha(CHECKPOINT),'predictions_sha256':sha(OUT/'TRAIN_predictions.json'),
        'constructor_seed':90211,'training_updates':0,'modelval_calibration_test_inference_used':False})


def run():
    protocol,dataset=validate();assert not (OUT/'progress.json').exists()
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any(' -m src.train ' in row for row in commands.splitlines())
    dump(OUT/'progress.json',{'status':'starting_TRAIN_only','pid':os.getpid()});started=time.perf_counter()
    try:
        rows=predict(dataset,protocol['first_constructor_seed'],True)
        first_seconds=time.perf_counter()-started
        dump(OUT/'TRAIN_predictions.json',rows)
        summary=summarize(rows)
        dump(OUT/'progress.json',{'status':'fresh_process_replay_running','pid':os.getpid(),
            'first_pass_seconds':first_seconds,'completed_QA':len(rows)})
        subprocess.run([sys.executable,__file__,'verify'],check=True)
        report={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),'checkpoint_sha256':sha(CHECKPOINT),
            'predictions_sha256':sha(OUT/'TRAIN_predictions.json'),'summary':summary,
            'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'first_pass_seconds':first_seconds,'elapsed_seconds_including_replay':time.perf_counter()-started,
            'training_updates':0,'weights_unchanged':True,'original_routing_unchanged':True,
            'modelval_calibration_test_inference_used':False,'new_clinical_labels':False,
            'long_training_authorized':False,'step3_complete':False,
            'limitations':['TRAIN errors are not an independent quality estimate',
                'Literal overlap categories do not establish clinical correctness or justify replacing labels',
                'No objective, architecture, hyperparameter, threshold or template selected from these data alone']}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'complete_TRAIN_diagnosis_verified','report_sha256':sha(OUT/'report.json')})
        print(json.dumps({k:v for k,v in summary.items() if k!='error_records'},indent=2))
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','verify'])
    {'prepare':prepare,'run':run,'verify':verify}[parser.parse_args().mode]()
