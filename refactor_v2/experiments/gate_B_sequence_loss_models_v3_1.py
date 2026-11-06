"""Actual-Qwen TRAIN gates, disposable updates, fresh saved PREtask initials.

Never executes a quality pilot or edits the frozen training/model source.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from src.context_controls import load_component_lookup
from src.data import collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.train import seed_training_epoch
from src.utils import normalize_answer,set_seed
from experiments.run_T_bounded_current_v3_1 import ROOT,BASE,data_and_schedule,hashes
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss

OUT=ROOT/'refactor_v2/reports/B_sequence_loss_model_gates_v3_1'
PLAN=ROOT/'refactor_v2/reports/B_sequence_loss_candidate_v3_1'
INITIAL=BASE/'fracatlas_B_roco_scale_pilot_v3_initialization.pt'
INITIAL_SHA='bfc799a5fb6db140ab833db460308c98cba76d871b9d415856175ed769abc09b'
MODES=('sequence_uniform','sequence_rarity')


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def table_of_weights():
    saved=json.loads((PLAN/'TRAIN_OPEN_weights.json').read_text())
    return {(r['question'],r['normalized_answer_frequency_key']):r['weight'] for r in saved['weights']}


def selected_rows():
    base,datasets,manifests=data_and_schedule();rows=records_of(datasets['train']);table=table_of_weights()
    binding=base['train']['component_batching']
    lookup,_=load_component_lookup(binding['manifest'],binding['seal'],ROOT)
    opened=[i for i,r in enumerate(rows) if r['answer_type']=='OPEN']
    rare=min(opened,key=lambda i:(-table[(rows[i]['question'],normalize_answer(rows[i]['answer']))],i))
    used={lookup[rows[rare]['image_id']]}
    common=min([i for i in opened if rows[i]['question']==rows[rare]['question'] and lookup[rows[i]['image_id']] not in used],
        key=lambda i:(table[(rows[i]['question'],normalize_answer(rows[i]['answer']))],i))
    used.add(lookup[rows[common]['image_id']])
    selected=[]
    for answer in ('yes','no'):
        index=next(i for i,r in enumerate(rows) if r['question']=='Is there a fracture in this X-ray?'
            and r['answer']==answer and lookup[r['image_id']] not in used)
        selected.append(index);used.add(lookup[rows[index]['image_id']])
    indices=selected+[rare,common]
    assert len(used)==4 and table[(rows[rare]['question'],normalize_answer(rows[rare]['answer']))]>table[(rows[common]['question'],normalize_answer(rows[common]['answer']))]
    return datasets,manifests,indices


def prepare():
    check_environment();assert not OUT.exists()
    preparation=json.loads((PLAN/'preparation_verification.json').read_text())
    assert preparation['passed'] and preparation['protocol_sha256']==sha(PLAN/'protocol.json')
    assert sha(INITIAL)==INITIAL_SHA
    datasets,manifests,indices=selected_rows()
    dependencies={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for p in [Path(__file__),PLAN/'protocol.json',PLAN/'TRAIN_OPEN_weights.json',PLAN/'preparation_verification.json',
        ROOT/'refactor_v2/experiments/private_TRAIN_sequence_loss_adapter_v3_1.py',
        ROOT/'refactor_v2/experiments/sequence_answer_loss_candidate_v3_1.py',
        ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py',
        ROOT/'refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py',
        ROOT/'refactor_v2/experiments/evaluate_pmc_objective_pilot_v3_2.py']:
        dependencies[str(p.resolve())]=sha(p)
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Actual Qwen private-TRAIN loss gates ONLY, no quality training release',
        'dependency_sha256':dependencies,'initial_sha256':INITIAL_SHA,'actual_manifests':manifests,
        'TRAIN_indices':indices,'selection':'max-weight OPEN then min-weight same-question distinct component; first canonical positive/negative distinct components',
        'TRAIN_metadata':[records_of(datasets['train'])[i] for i in indices],
        'modes':list(MODES),'discarded_optimizer_updates_each':2,
        'fresh_initial_saved_before_any_update':True,'native_token_CE_gradient_tolerances':{'atol':1e-6,'rtol':1e-4},
        'modelval_calibration_test_inference_allowed':False,'quality_pilot_authorized':False,'long_training_authorized':False})
    print('Prepared actual-model gate on four TRAIN components; no model constructed')


def validate():
    check_environment();protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest
    assert sha(INITIAL)==protocol['initial_sha256']==INITIAL_SHA
    datasets,manifests,indices=selected_rows()
    assert manifests==protocol['actual_manifests'] and indices==protocol['TRAIN_indices']
    return protocol,datasets,indices


class PoisonTargets:
    def __iter__(self):raise AssertionError('Eval/inference loss adapter touched target metadata')
    def __getitem__(self,key):raise AssertionError('Eval/inference loss adapter touched target metadata')


def run():
    protocol,datasets,indices=validate();assert not (OUT/'progress.json').exists()
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any(' -m src.train ' in row or ('Python' in row and 'audit_B_fullTRAIN_errors_v3_1.py run' in row) for row in commands.splitlines())
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    assert len(initial['state'])==263 and len(initial['answer_vocab'])==68
    batch=collate_fn([datasets['train'][i] for i in indices]);public={'images':batch['images'],'questions':batch['questions']}
    table=table_of_weights();results={};native_controls=[];started=time.perf_counter()
    dump(OUT/'progress.json',{'status':'running_private_TRAIN_gates','pid':os.getpid()})
    try:
        for mode in MODES:
            set_seed(42);model=BoneVQAModel(copy.deepcopy(initial['config']),initial['answer_vocab'],device='cpu')
            model.load_state_dict(initial['state'],strict=False)
            assert hashes(model.trainable_state_dict())==hashes(initial['state'])
            assert not any(model.cfg[k] for k in ('use_textual_prompt','use_visual_prompt','use_lens','use_latent_prompt','use_region_in_prompt'))
            fresh=OUT/(mode+'_PREtask.pt')
            model.save(str(fresh),{'scope':'PREtask_zeroQAupdates_sequence_loss_candidate',
                'mode':mode,'gate_protocol_sha256':sha(OUT/'protocol.json'),'original_PREtask_sha256':INITIAL_SHA})
            cfg=copy.deepcopy(model.cfg);before_eval=hashes(model.trainable_state_dict())
            model.eval()
            with torch.no_grad():
                original=model.predict_batch(public)
                with private_training_loss(model,PoisonTargets(),3,mode,table) as skipped:
                    repeated=model.predict_batch(public)
                assert not skipped['applied'] and skipped['calls']==0
                assert [{k:v for k,v in p.items() if k!='masks'} for p in original]==[{k:v for k,v in p.items() if k!='masks'} for p in repeated]
            assert hashes(model.trainable_state_dict())==before_eval
            params=dict(model.named_parameters())
            names=['qformer.query','prefix_proj.0.weight',
                next(k for k in params if k.startswith('visual.') and 'lora_B' in k),
                next(k for k in params if k.startswith('llm.') and 'lora_B' in k)]
            selected=[params[k] for k in names];before={k:params[k].detach().clone() for k in names}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.98),weight_decay=.01)
            steps=[]
            for step in range(2):
                seed_training_epoch(42,step);model.train();optimizer.zero_grad(set_to_none=True)
                with private_training_loss(model,batch,3,mode,table) as trace:
                    output=model(batch,stage=3)
                assert trace['applied'] and trace['calls']==1
                assert torch.isfinite(output['loss']) and torch.isfinite(output['loss_answer'])
                native=trace['native_loss'];token=trace['token_mean']
                assert torch.allclose(native,token,atol=1e-6,rtol=1e-5)
                native_grad=torch.autograd.grad(native,selected,retain_graph=True,allow_unused=True)
                token_grad=torch.autograd.grad(token,selected,retain_graph=True,allow_unused=True)
                difference={}
                for name,a,b in zip(names,native_grad,token_grad):
                    assert a is not None and b is not None
                    assert torch.isfinite(a).all() and torch.isfinite(b).all()
                    assert torch.allclose(a,b,atol=1e-6,rtol=1e-4),name
                    difference[name]=float((a-b).abs().max())
                answer_grad=torch.autograd.grad(output['loss_answer'],selected,retain_graph=True,allow_unused=True)
                maxima={k:0. if g is None else float(g.abs().max()) for k,g in zip(names,answer_grad)}
                assert all(g is not None and torch.isfinite(g).all() for g in answer_grad)
                assert all(v>0 for v in maxima.values())
                assert abs(float(output['loss_lm'])-float(trace['selected_loss'].detach()))<1e-6
                controls={k:float(output[k]) for k in ('loss_cls','loss_type','loss_fracture','loss_con')}
                if step==0:native_controls.append({'native_LM':float(native.detach()),**controls})
                steps.append({'native_LM':float(native.detach()),'recomputed_token_LM':float(token.detach()),
                    'selected_LM':float(trace['selected_loss'].detach()),'OPEN_weights':trace['weights'],
                    'selected_native_token_gradient_max_errors':difference,'answer_gradient_max':maxima,
                    'other_losses':controls})
                output['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                assert finite_tree(model.trainable_state_dict())
                del output,trace,native,token,native_grad,token_grad,answer_grad
            updates={k:float((params[k].detach()-before[k]).abs().max()) for k in names}
            assert all(v>0 for v in updates.values())
            assert model.cfg==cfg and sha(fresh)
            results[mode]={'passed':True,'steps':steps,'actual_selected_updates':updates,
                'fresh_initial_path':str(fresh),'fresh_initial_sha256':sha(fresh),
                'all263_initial_tensors_byteexact_before_updates':True,'inference_eval_bypass_no_target_reads_passed':True,
                'discarded_smoke_updates':2,'smoke_weights_discarded':True}
            dump(OUT/(mode+'_gate.json'),results[mode])
            del model,optimizer,params,selected,before,original,repeated;gc.collect()
        assert native_controls[0]==native_controls[1], 'All first-forward native and non-LM loss values must match across arms'
        report={'passed':True,'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),
            'results':results,'initial_shared263_tensor_hashes':hashes(initial['state']),
            'first_forward_native_LM_and_other_objectives_exact_across_arms':True,
            'elapsed_seconds':time.perf_counter()-started,'total_disposable_updates':4,
            'quality_training_updates':0,'modelval_calibration_test_inference_used':False,
            'bounded_pilot_authorized':False,'long_training_authorized':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'gates_passed_pending_fresh_initial_verification','report_sha256':sha(OUT/'report.json')})
        print(json.dumps(report,indent=2))
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


def verify():
    protocol,_,_=validate();destination=OUT/'independent_gate_verification.json';assert not destination.exists()
    report=json.loads((OUT/'report.json').read_text());state=json.loads((OUT/'progress.json').read_text())
    assert report['passed'] and sha(OUT/'report.json')==state['report_sha256']
    initial=torch.load(INITIAL,map_location='cpu',weights_only=False)
    for mode in MODES:
        row=report['results'][mode];assert row['passed'] and row['smoke_weights_discarded']
        assert sha(row['fresh_initial_path'])==row['fresh_initial_sha256']
        saved=torch.load(row['fresh_initial_path'],map_location='cpu',weights_only=False)
        assert saved['config']==initial['config'] and saved['answer_vocab']==initial['answer_vocab']
        assert hashes(saved['state'])==hashes(initial['state'])==report['initial_shared263_tensor_hashes']
        assert saved['extra']['scope']=='PREtask_zeroQAupdates_sequence_loss_candidate'
        assert saved['extra']['mode']==mode and saved['extra']['gate_protocol_sha256']==sha(OUT/'protocol.json')
        assert finite_tree(saved['state'])
    dump(destination,{'passed':True,'report_sha256':sha(OUT/'report.json'),
        'fresh_process_saved263_initial_tensors_config_vocab_zeroQAupdates_verified':True,
        'actual_Qwen_native_token_CE_and_selected_gradient_equivalence_passed':True,
        'finite_nonzero_answer_gradients_and_discarded_updates_passed':True,
        'public_eval_inference_bypass_no_target_reads_passed':True,
        'quality_training_updates':0,'bounded_pilot_authorized':False,'long_training_authorized':False})
    print('Fresh process verified both PREtask initials and model/loss gate bindings; no quality pilot released')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','verify'])
    {'prepare':prepare,'run':run,'verify':verify}[parser.parse_args().mode]()
