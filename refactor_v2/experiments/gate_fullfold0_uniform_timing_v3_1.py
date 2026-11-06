"""Four TRAIN rows, two disposable steps/arm; no quality training release."""
import copy
import gc
import json
import os
import time
from collections import Counter
from pathlib import Path
import torch
from src.data import build_dataset, collate_fn
from src.train import build_split, load_init_weights, seed_training_epoch
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.utils import set_seed, normalize_answer
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.run_T_bounded_current_v3_1 import hashes
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
from experiments.gate_B_sequence_loss_models_v3_1 import PoisonTargets
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump

OUT=ROOT/'refactor_v2/reports/fullfold0_uniform_timing_gate_v3_1'


def main():
    check_environment();OUT.mkdir(exist_ok=False)
    cfgpath=ROOT/'refactor_v2/configs/revised_sealed_candidates_v3/b.json'
    cfg=json.loads(cfgpath.read_text());cfg=copy.deepcopy(cfg)
    base=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0'
    train=base/'vqa_train.json';vocabpath=base/'answer_vocab_runtime_normalized_v3.json'
    sealpath=ROOT/'refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json'
    seal=json.loads(sealpath.read_text())
    for p in (train,vocabpath,base/'detector_split_v3.json'):
        assert sha(p)==seal['artifacts'][str(p.relative_to(ROOT))]
    loader=cfg['data']['loader_kwargs']
    loader.update(seal_path=str(sealpath),project_root=str(ROOT),with_masks=False,with_boxes=False,
        boxes_from=None,boxes_as_masks=False,root=str(ROOT/'data'),
        sealed_records_by_split={'train':str(train),'model_val':None,'calibration':None,'test':None})
    dataset=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,loader)
    rows=records_of(dataset);assert len(rows)==11502
    lookup=json.loads((base/'detector_split_v3.json').read_text())['component_by_image']
    chosen=[];used=set()
    for kind,answer in [('CLOSED','yes'),('CLOSED','no'),('OPEN',None),('OPEN',None)]:
        i=next(i for i,r in enumerate(rows) if r['answer_type']==kind
            and lookup[r['image_id']] not in used
            and (answer is None or (r['question']=='Is there a fracture in this X-ray?' and r['answer']==answer)))
        chosen.append(i);used.add(lookup[rows[i]['image_id']])
    batch=collate_fn([dataset[i] for i in chosen]);public={'images':batch['images'],'questions':batch['questions']}
    vocab=json.loads(vocabpath.read_text());assert len(vocab)==125
    counts=Counter(r['answer'] for r in rows if r['question']=='Is there a fracture in this X-ray?')
    modelcfg=copy.deepcopy(cfg['model']);modelcfg.update(module_init_seed=1771,fracture_positive_weight=counts['no']/counts['yes'])
    assert not any(modelcfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
    upstream=ROOT/'refactor_v2/checkpoints/revised_roco_alignment_v3/best.pt'
    assert sha(upstream)=='08cd132ef1097da8f64758225b3f66b7a5329a5d008accfe84e352e94cb98302'
    deps={str(p):sha(p) for p in [Path(__file__).resolve(),cfgpath,train,vocabpath,sealpath,
        ROOT/'refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json',
        ROOT/'refactor_v2/experiments/private_TRAIN_sequence_loss_adapter_v3_1.py',
        ROOT/'refactor_v2/experiments/sequence_answer_loss_candidate_v3_1.py']}
    dump(OUT/'protocol.json',{'scope':'Four real TRAIN rows only, fullfold0 vocabulary/initialization/timing gate',
        'dependencies_sha256':deps,'upstream_sha256':sha(upstream),'TRAIN_indices':chosen,
        'TRAIN_image_ids':[rows[i]['image_id'] for i in chosen],'TRAIN_components':[lookup[rows[i]['image_id']] for i in chosen],
        'selection':'first canonical yes/no then first two OPEN on unused components, no prediction selection',
        'vocabulary_size':125,'TRAIN_canonical_counts':dict(counts),'model_config':modelcfg,
        'modes':['token_mean','sequence_uniform'],'disposable_updates_each':2,
        'seed':42,'lr':1e-4,'clip':1.,'optimizer':'AdamW betas0.9,0.98 weight_decay0.01',
        'weights_after_updates_saved':False,'reserved_role_inference_allowed':False,
        'quality_training_released':False,'long_training_authorized':False})
    dump(OUT/'progress.json',{'status':'running','pid':os.getpid()})
    table={(r['question'],normalize_answer(r['answer'])):1. for r in rows if r['answer_type']=='OPEN'}
    results={};initials={};controls=[];started=time.perf_counter()
    try:
        for mode in ('token_mean','sequence_uniform'):
            armstart=time.perf_counter();set_seed(42)
            model=BoneVQAModel(copy.deepcopy(modelcfg),vocab,device='cpu')
            heads={k:v.clone() for k,v in model.trainable_state_dict().items() if k.startswith(('closed_head.','fracture_head.'))}
            load_init_weights(model,str(upstream),cfg['checkpoint_allow_mismatch_prefixes'],False)
            model.load_state_dict(heads,strict=False)
            initials[mode]=hashes(model.trainable_state_dict());assert len(initials[mode])==263
            assert finite_tree(model.trainable_state_dict())
            model.eval();pre=hashes(model.trainable_state_dict())
            with torch.no_grad():
                first=model.predict_batch(public)
                with private_training_loss(model,PoisonTargets(),3,mode,table) as bypass:
                    repeat=model.predict_batch(public)
            assert bypass['calls']==0 and not bypass['applied']
            assert [{k:v for k,v in r.items() if k!='masks'} for r in first]==[{k:v for k,v in r.items() if k!='masks'} for r in repeat]
            assert hashes(model.trainable_state_dict())==pre
            params=dict(model.named_parameters());names=['qformer.query','prefix_proj.0.weight',
                next(k for k in params if k.startswith('visual.') and 'lora_B' in k),
                next(k for k in params if k.startswith('llm.') and 'lora_B' in k)]
            before={n:params[n].detach().clone() for n in names}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.98),weight_decay=.01)
            steps=[]
            for step in range(2):
                begin=time.perf_counter();seed_training_epoch(42,step);model.train();optimizer.zero_grad(set_to_none=True)
                with private_training_loss(model,batch,3,mode,table) as trace:out=model(batch,stage=3)
                assert trace['applied'] and trace['calls']==1
                assert torch.allclose(trace['native_loss'],trace['token_mean'],atol=1e-6,rtol=1e-5)
                grads=torch.autograd.grad(out['loss_answer'],[params[n] for n in names],retain_graph=True,allow_unused=True)
                maxima={n:float(g.abs().max()) if g is not None else 0. for n,g in zip(names,grads)}
                assert all(g is not None and torch.isfinite(g).all() for g in grads) and all(v>0 for v in maxima.values())
                if step==0:controls.append({k:float(out[k]) for k in ('loss_cls','loss_fracture','loss_type','loss_con')}|{'native_LM':float(trace['native_loss'].detach())})
                loss=float(out['loss'].detach());assert torch.isfinite(out['loss'])
                out['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                assert finite_tree(model.trainable_state_dict())
                steps.append({'loss':loss,'answer_gradient_max':maxima,'seconds':time.perf_counter()-begin})
                del out,trace,grads
            updates={n:float((params[n].detach()-before[n]).abs().max()) for n in names}
            assert all(v>0 for v in updates.values())
            results[mode]={'steps':steps,'selected_updates':updates,'elapsed_seconds':time.perf_counter()-armstart,
                'public_inference_poison_target_bypass_passed':True,'all_trainable_states_finite':True}
            del model,optimizer,params,heads,before,first,repeat;gc.collect()
        assert initials['token_mean']==initials['sequence_uniform'] and controls[0]==controls[1]
        for p,h in deps.items():assert sha(Path(p))==h
        report={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'results':results,
            'all263_initial_tensor_hashes':initials['token_mean'],'matched_initial_states_and_native_first_forward_controls_exact':True,
            'elapsed_seconds':time.perf_counter()-started,'disposable_optimizer_updates':4,
            'quality_training_updates':0,'all_smoke_weights_discarded':True,'model_checkpoints_saved':0,
            'modelval_calibration_outer_inference_used':False,
            'limits':['Four TRAIN rows only, not generalization or full-epoch runtime estimate',
                'No new checkpoint reload or uninterrupted-versus-resumed training equivalence demonstrated',
                'Fold0 only, does not certify other fold vocabularies/configurations'],
            'long_training_authorized':False,'step3_complete':False,'production_released':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'complete_pending_artifact_verification','report_sha256':sha(OUT/'report.json')})
        print(json.dumps({k:v for k,v in report.items() if k!='all263_initial_tensor_hashes'},indent=2))
    except Exception as e:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_retry','error':repr(e)});raise


if __name__=='__main__':main()
