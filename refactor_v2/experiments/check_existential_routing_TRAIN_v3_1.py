"""Prepare/test a separate question-only correction on the existing TRAIN24.

No updates, new labels, changed checkpoints, or validation/calibration/test
inference. This is a runtime diagnostic, not a released production fix.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import time

import torch
from src.data import collate_fn
from src.utils import compute_metrics,set_seed
from src.models.bonevqa import BoneVQAModel
from experiments import diagnose_T_routes_stratified24_TRAIN_v3_1 as selection
from experiments.existential_polar_routing_adapter_v3_1 import candidate_closed_question,experimental_routing,test_contract

ROOT=selection.ROOT;OUT=ROOT/'refactor_v2/reports/existential_routing_TRAIN_v3_1'
sha=selection.prior.sha;dump=selection.prior.dump
CHECKPOINTS={name:ROOT/'refactor_v2/checkpoints'/run/'last.pt' for name,run in
    [('B_ROCO','fracatlas_B_roco_scale_pilot_v3'),('T_clinical','T_clinical_seed42_bounded_current_v3_1')]}


def prepare():
    selection.prior.check_environment();assert not OUT.exists()
    tests=test_contract()
    ds,manifests,batches,_=selection.select_rows();records=selection.prior.records_of(ds)
    from src.utils import is_closed_question,is_choice_question,is_polar_question
    effects={}
    for record in records:
        q=record['question'];old=is_closed_question(q);new=candidate_closed_question(q)
        if old!=new:
            assert new and is_polar_question(q) and is_choice_question(q)
            entry=effects.setdefault(q,{'TRAIN_count':0,'existing_type_counts':{}})
            entry['TRAIN_count']+=1
            kind=record['answer_type'];entry['existing_type_counts'][kind]=entry['existing_type_counts'].get(kind,0)+1
    assert effects=={'Is there any orthopedic hardware or implant visible?':{'TRAIN_count':38,'existing_type_counts':{'CLOSED':38}}}
    selected=[records[i] for batch in batches for i in batch]
    previous=json.loads((selection.OUT/'independent_artifact_audit.json').read_text())
    assert previous['passed']
    dependencies={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for p in [Path(__file__),ROOT/'refactor_v2/experiments/existential_polar_routing_adapter_v3_1.py',
              Path(selection.__file__),Path(selection.prior.__file__),selection.OUT/'protocol.json',
              selection.OUT/'report.json',selection.OUT/'independent_artifact_audit.json',
              ROOT/'refactor_v2/reports/TRAIN_question_routing_audit_v3_1.json']:
        dependencies[str(p.resolve())]=sha(p)
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Question-only routing correction diagnostic on fixed24TRAIN; not production',
        'grammar_tests':tests,'TRAIN_only_heuristic_changes':effects,'selected_TRAIN_records':selected,
        'selected_indices_by_batch':batches,'actual_manifests':manifests,
        'checkpoint_sha256':{k:sha(v) for k,v in CHECKPOINTS.items()},'dependency_sha256':dependencies,
        'conditions':['original','candidate_question_only_adapter','original_restored'],
        'public_input_keys':['images','questions'],'GT_type_region_ROI_inputs':False,
        'fresh_constructor_seeds':[90210,90211],'threshold':.5,'training_updates':0,
        'modelval_calibration_test_inference_allowed':False,'long_training_authorized':False})
    print(json.dumps({'grammar_tests':tests,'TRAIN_only_effects':effects},indent=2))


def compact(predictions,batch):
    output=[]
    for i,prediction in enumerate(predictions):
        row={'image_id':batch['image_ids'][i],'question':batch['questions'][i],
             'gt':batch['answers'][i],'answer_type':batch['answer_types'][i],
             'pred':prediction['answer'],'actual_predicted_route':prediction['answer_type']}
        if 'p_yes' in prediction:row['p_yes']=prediction['p_yes']
        output.append(row)
    return output


@torch.no_grad()
def measure(name,dataset,batches,seed):
    set_seed(seed);model=BoneVQAModel.load(str(CHECKPOINTS[name]),device='cpu').eval()
    before=selection.prior.hashes(model.trainable_state_dict());cfg=copy.deepcopy(model.cfg)
    assert cfg['fracture_yes_threshold']==.5 and cfg.get('fracture_platt_slope') is None
    assert not any(cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_region_in_prompt'))
    results={condition:[] for condition in ('original','adapter','restored')}
    original_core=model._encode_core
    captures=[]
    def traced(*args,**kwargs):
        core=original_core(*args,**kwargs)
        captures.append({k:core[k].detach().cpu().clone() for k in ('logits_closed','type_logit','fracture_logit')})
        return core
    model._encode_core=traced
    try:
        for indices in batches:
            batch=collate_fn([dataset[i] for i in indices]);public={'images':batch['images'],'questions':batch['questions']}
            captures.clear()
            results['original'].extend(compact(model.predict_batch(public),batch))
            with experimental_routing():
                results['adapter'].extend(compact(model.predict_batch(public),batch))
            results['restored'].extend(compact(model.predict_batch(public),batch))
            assert len(captures)==3
            assert all(torch.equal(captures[0][key],value[key]) for value in captures[1:] for key in captures[0])
    finally:
        model._encode_core=original_core
    assert results['original']==results['restored']
    assert model.cfg==cfg and selection.prior.hashes(model.trainable_state_dict())==before
    for original,adapted in zip(results['original'],results['adapter']):
        from src.utils import is_closed_question
        if candidate_closed_question(original['question'])==is_closed_question(original['question']):
            assert original==adapted, 'Unaffected question predictions/scores must remain exact'
        else:
            assert adapted['actual_predicted_route']=='CLOSED'
    metrics={k:compute_metrics(v,fracture_threshold=.5) for k,v in results.items()}
    del model;gc.collect()
    return {'records':results,'metrics':metrics,'head_tensors_exact_all_conditions':True,
            'original_runtime_predictions_restored_exact':True,'weights_config_unchanged':True}


def run():
    selection.prior.check_environment();assert not (OUT/'progress.json').exists()
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any(' -m src.train ' in row or ('Python' in row and 'diagnose_T_' in row and ' run' in row) for row in commands.splitlines())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest
    assert test_contract()==protocol['grammar_tests']
    dataset,manifests,batches,_=selection.select_rows()
    assert manifests==protocol['actual_manifests'] and batches==protocol['selected_indices_by_batch']
    assert {name:sha(path) for name,path in CHECKPOINTS.items()}==protocol['checkpoint_sha256']
    started=time.perf_counter();dump(OUT/'progress.json',{'status':'running_TRAIN_only_zero_updates'})
    try:
        results={}
        for name in CHECKPOINTS:
            first=measure(name,dataset,batches,90210);second=measure(name,dataset,batches,90211)
            assert first==second
            results[name]=first
            dump(OUT/(name+'.json'),first)
        assert {name:sha(path) for name,path in CHECKPOINTS.items()}==protocol['checkpoint_sha256']
        report={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),'results':results,
            'two_fresh_constructors_replay_exact':True,'elapsed_seconds':time.perf_counter()-started,
            'training_updates':0,'modelval_calibration_test_inference_used':False,
            'long_training_authorized':False,'production_released':False,'step3_complete':False,
            'limitations':['TRAIN-only routing diagnostic; no generalization/clinical benefit established',
                'Narrow English existential pattern, not a general choice parser',
                'Original source and production runtime remain unchanged; no threshold/template fitting']}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'complete_TRAIN_diagnostic_only','report_sha256':sha(OUT/'report.json')})
        print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2))
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().mode]()
