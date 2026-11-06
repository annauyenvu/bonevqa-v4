"""Matched first16 TRAIN microbatches per arm, accumulation4, discarded weights."""
import copy
import gc
import json
import math
import os
import time
from pathlib import Path
import torch
from src.data import build_dataset,collate_fn
from src.train import build_split,seed_training_epoch
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.utils import set_seed,normalize_answer
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.run_T_bounded_current_v3_1 import hashes
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT,sha,dump

SCREEN=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
OUT=ROOT/'refactor_v2/reports/fullfold0_accumulation_timing_v3_1'


def main():
    check_environment();assert not OUT.exists()
    protocol=json.loads((SCREEN/'protocol.json').read_text())
    for p,h in protocol['dependency_sha256'].items():assert sha(Path(p))==h
    assert sha(SCREEN/'component_schedule.json')==protocol['schedule_sha256']
    batches=json.loads((SCREEN/'component_schedule.json').read_text())[0]['ordered_indices'][:16]
    assert len(batches)==16 and all(len(b)==4 for b in batches)
    for mode,info in protocol['configs'].items():assert sha(Path(info['path']))==info['sha256']
    OUT.mkdir();dump(OUT/'protocol.json',{'scope':'TRAIN-only real accumulation timing, no quality release',
        'screen_protocol_sha256':sha(SCREEN/'protocol.json'),'script_sha256':sha(Path(__file__).resolve()),
        'TRAIN_microbatch_indices':batches,'microbatches_each':16,'grad_accum':4,'updates_each':4,
        'weights_after_updates_saved':False,'reserved_role_inference_allowed':False,
        'model_checkpoint_saved':False,'quality_training_authorized':False})
    start=time.perf_counter();results={};initials={}
    dump(OUT/'progress.json',{'status':'running','pid':os.getpid()})
    try:
        for mode in protocol['arm_order']:
            cfg=json.loads(Path(protocol['configs'][mode]['path']).read_text());set_seed(42)
            ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
            rows=records_of(ds);table={(r['question'],normalize_answer(r['answer'])):1. for r in rows if r['answer_type']=='OPEN'}
            model=BoneVQAModel.load(cfg['init_from'],device='cpu')
            initials[mode]=hashes(model.trainable_state_dict());assert len(initials[mode])==263
            assert sha(Path(cfg['init_from']))==protocol['initial_checkpoint']['sha256']
            named=dict(model.named_parameters());selected=['qformer.query','prefix_proj.0.weight',
                next(k for k in named if k.startswith('visual.') and 'lora_B' in k),
                next(k for k in named if k.startswith('llm.') and 'lora_B' in k)]
            before={n:named[n].detach().clone() for n in selected};grads={n:0. for n in selected}
            params=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.98),weight_decay=.01)
            horizon=1438;warmup=int(.05*horizon)
            def factor(step):
                if step<warmup:return (step+1)/max(1,warmup)
                return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,horizon-warmup))))
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
            model.train();seed_training_epoch(42,0);optimizer.zero_grad(set_to_none=True)
            update_seconds=[];open_rows=0;adapted_calls=0;armstart=time.perf_counter();groupstart=armstart
            for number,indices in enumerate(batches,1):
                batch=collate_fn([ds[i] for i in indices]);open_rows+=sum(str(t).upper()!='CLOSED' for t in batch['answer_types'])
                with private_training_loss(model,batch,3,mode,table) as trace:out=model(batch,stage=3)
                adapted_calls+=int(trace['applied']);assert torch.isfinite(out['loss'])
                (out['loss']/4).backward()
                if number%4==0:
                    for n in selected:
                        g=named[n].grad
                        if g is not None:
                            assert torch.isfinite(g).all();grads[n]=max(grads[n],float(g.abs().max()))
                    torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
                    update_seconds.append(time.perf_counter()-groupstart);groupstart=time.perf_counter()
                del out,trace,batch
            assert len(update_seconds)==4 and open_rows>0 and all(v>0 for v in grads.values())
            changes={n:float((named[n].detach()-before[n]).abs().max()) for n in selected}
            assert all(v>0 for v in changes.values()) and finite_tree(model.trainable_state_dict())
            results[mode]={'TRAIN_rows':64,'TRAIN_OPEN_rows':open_rows,'adapted_LM_calls':adapted_calls,
                'optimizer_updates':4,'update_seconds':update_seconds,'TRAIN_loop_seconds':sum(update_seconds),
                'mean_seconds_per_accumulated_update':sum(update_seconds)/4,'selected_gradient_max':grads,
                'selected_actual_updates':changes,'all_trainable_states_finite':True,
                'all_updated_weights_discarded':True,'checkpoints_saved':0}
            dump(OUT/(mode+'.json'),results[mode])
            dump(OUT/'progress.json',{'status':'running','pid':os.getpid(),'completed_modes':list(results)})
            del model,optimizer,scheduler,params,named,before;gc.collect()
        assert initials['token_mean']==initials['sequence_uniform']
        report={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'screen_protocol_sha256':sha(SCREEN/'protocol.json'),
            'results':results,'all263_initial_hashes_equal_across_arms':True,
            'elapsed_seconds_including_load_and_checks':time.perf_counter()-start,
            'total_disposable_optimizer_updates':8,'quality_training_updates':0,
            'all_smoke_weights_discarded':True,'modelval_calibration_test_used':False,
            'limits':['First16 TRAIN microbatches/64rows only, not a full-epoch timing guarantee',
                'TRAIN loop includes input decoding,forward,backward,accumulation4,clip/optimizer/scheduler',
                'Validation,full training QC,checkpoint writes and later independent evaluation are additional',
                'No trained checkpoint or optimizer/RNG-resume equivalence from discarded smoke weights'],
            'long_training_authorized':False,'step3_complete':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'complete_pending_artifact_audit','report_sha256':sha(OUT/'report.json')})
        print(json.dumps(report,indent=2))
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_retry','error':repr(error),'completed_modes':list(results)});raise


if __name__=='__main__':main()
