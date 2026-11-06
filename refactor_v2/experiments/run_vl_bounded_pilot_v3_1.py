"""Explicit five-arm bounded research executor; never a full-fold CLI override.

Preflight seals dependencies without updates. --run runs ONE sealed arm only;
an existing output is a hard stop, not an implicit resume/retry.
"""
import argparse
import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.context_controls import ComponentBatchSampler, component_disjoint_donors, load_component_lookup
from src.data import build_dataset, collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_split, evaluate_split, seed_training_epoch
from src.utils import set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.vl_bounded_execution_contract_v3_1 import ARMS, validate_plan
from experiments.vl_bounded_proposal_contract_v3_1 import sha

ROOT=Path.cwd()
PLANS=ROOT/'refactor_v2/reports/vl_bounded_execution_contract_v3_1'
GATES=ROOT/'refactor_v2/reports/vl_control_update_gates_v3_1'
OUT=ROOT/'refactor_v2/reports/vl_bounded_quality_v3_1'
BASE=ROOT/'refactor_v2/reports/fracatlas_scale_pilot_v3'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def check_environment():
    expected={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'8',
              'HF_HOME':str(ROOT/'.evaluation_runtime/hf_cache'),
              'BONEVQA_DATA':str(ROOT/'.evaluation_runtime/data')}
    assert all(os.environ.get(k)==v for k,v in expected.items())
    assert sys.flags.optimize==0, 'Assertions are required'
    torch.set_num_threads(8)
    assert torch.get_num_threads()==8


def dependency_paths():
    paths=list((ROOT/'refactor_v2/src').rglob('*.py'))
    paths+=list((ROOT/'refactor_v2/experiments').glob('*.py'))
    paths+=list(PLANS.glob('*.json'))
    paths+=list(GATES.glob('*_gate.json'))
    paths+=[GATES/'report.json',GATES/'independent_initialization_verification.json',
            BASE/'fracatlas_B_roco_scale_pilot_v3.json',BASE/'execution_release.json',
            ROOT/'refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json']
    return sorted(set(paths))


def setup_data(plan):
    base=json.loads((BASE/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    kwargs=copy.deepcopy(base['data']['loader_kwargs'])
    assert kwargs['sealed_records_by_split']['calibration'] is None
    assert kwargs['sealed_records_by_split']['test'] is None
    kwargs.update(with_masks=False,with_boxes=False,boxes_from=plan['proposals']['path'],
                  boxes_as_masks=True,roi_control='real')
    datasets={r:build_split(build_dataset,base['data']['dataset'],r,None,True,kwargs)
              for r in ('train','model_val')}
    expected=json.loads((BASE/'execution_release.json').read_text())['actual_manifests']
    assert {r:manifest(ds) for r,ds in datasets.items()}==expected
    batch_cfg=base['train']['component_batching']
    lookup,_=load_component_lookup(batch_cfg['manifest'],batch_cfg['seal'],ROOT)
    schedules=json.loads((PLANS/'component_epoch_schedule.json').read_text())
    sampler=ComponentBatchSampler([lookup[r['image_id']] for r in records_of(datasets['train'])],4,42)
    for epoch,entry in enumerate(schedules):
        sampler.set_epoch(epoch)
        batches=list(sampler)
        assert batches==entry['ordered_indices'] and len(batches)==294
        assert sorted(i for b in batches for i in b)==list(range(1173))
        for batch in batches:
            groups=[lookup[records_of(datasets['train'])[i]['image_id']] for i in batch]
            donors=component_disjoint_donors(groups)
            assert sorted(donors)==list(range(len(batch)))
            assert all(groups[i]!=groups[j] for i,j in enumerate(donors))
    return datasets,lookup,schedules


def assert_update_budget(phase_updates,global_updates,phase_cap):
    if not (0<=phase_updates<phase_cap and 0<=global_updates<296):
        raise RuntimeError('Optimizer update outside sealed phase/total budget')


def preflight():
    check_environment()
    assert not OUT.exists(), 'Existing execution artifacts; inspect, never overwrite'
    subprocess.run([sys.executable,'refactor_v2/scripts/seal_technical_reference_v3.py','--verify'],check=True)
    verified=json.loads((GATES/'independent_initialization_verification.json').read_text())
    gates=json.loads((GATES/'report.json').read_text())
    caps=json.loads((PLANS/'report.json').read_text())
    assert verified['passed'] and gates['passed'] and caps['passed_plan_and_caps_tests']
    assert verified['source_control_gates_sha256']==sha(GATES/'report.json')
    assert len(caps['negative_cases_rejected'])==13 and all(caps['negative_cases_rejected'].values())
    init_hashes={}
    for arm in ARMS:
        path=PLANS/(arm+'.json')
        assert sha(path)==caps['config_sha256'][arm]==gates['results'][arm]['config_sha256']
        plan=json.loads(path.read_text()); validate_plan(plan)
        row=gates['results'][arm]
        assert verified['checks'][arm]['passed'] and verified['checks'][arm]['zeroQAupdates']
        assert row['smoke_updated_weights_discarded']
        assert sha(row['fresh_initialization_path'])==row['fresh_initialization_sha256']
        init_hashes[arm]={'path':row['fresh_initialization_path'],'sha256':row['fresh_initialization_sha256']}
    datasets,_,_=setup_data(json.loads((PLANS/'V_real.json').read_text()))
    del datasets;gc.collect()
    rejected=[]
    for phase,total,cap in ((296,296,296),(148,148,148),(-1,0,148),(0,296,148),(0,-1,148)):
        try:assert_update_budget(phase,total,cap)
        except RuntimeError:rejected.append([phase,total,cap])
        else:raise RuntimeError('Budget overrun accepted')
    for phase,total,cap in ((295,295,296),(147,295,148),(0,148,148)):
        assert_update_budget(phase,total,cap)
    release={'scope':'ONLY five named seed42 bounded V/L research arms; not long training',
             'passed':True,'created_utc':now(),'dependency_sha256':{str(p):sha(p) for p in dependency_paths()},
             'fresh_initializations':init_hashes,'arms':list(ARMS),'total_updates_per_arm':296,
             'budget_negative_cases_rejected':rejected,'exact_sampler_and_manifests_verified':True,
             'primary_checkpoint':'LAST296','long_training_released':False,
             'calibration_or_test_allowed':False,'production_fullfold_guard_unchanged':True,
             'evaluation_scope':'142QA reused model-validation; no independent confirmation',
             'best_checkpoint':'secondary stage3 fracture_auprc only; never replaces primary LAST',
             'L_stage1_no_supervision':'same frozen zero-connected backward, schedule preserved; count explicitly',
             'loss_and_accumulation':'frozen model loss, divide every microbatch by4 including tail;74 updates per epoch'}
    OUT.mkdir();dump(OUT/'execution_release.json',release)
    print('Bounded executor preflight sealed; zero quality updates',flush=True)


def run(arm):
    check_environment()
    release=json.loads((OUT/'execution_release.json').read_text())
    assert release['passed'] and arm in release['arms'] and not release['long_training_released']
    for path,digest in release['dependency_sha256'].items():
        assert sha(path)==digest, 'Sealed dependency changed: '+path
    plan=json.loads((PLANS/(arm+'.json')).read_text());validate_plan(plan)
    init=release['fresh_initializations'][arm]
    assert sha(init['path'])==init['sha256']
    run_dir=OUT/arm
    assert not run_dir.exists(), 'Existing arm; no implicit retry/resume'
    # Exclusive scope-wide lock prevents duplicate different-arm jobs too.
    lock_fd=os.open(OUT/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.close(lock_fd)
    run_dir.mkdir()
    state={'status':'starting','arm':arm,'pid':os.getpid(),'started_utc':now(),
           'global_step':0,'quality_gate_passed':False,'primary_checkpoint':'LAST296'}
    dump(OUT/'ACTIVE.lock',state)
    dump(run_dir/'progress.json',state)
    started=time.perf_counter()
    try:
        datasets,lookup,schedules=setup_data(plan)
        location=json.loads(Path(plan['controls']['geometry_path']).read_text())
        def collate(samples):
            batch=collate_fn(samples)
            batch['component_ids']=[lookup[i] for i in batch['image_ids']]
            if arm.endswith('location_placebo'):
                batch['boxes']=[location[i] or None for i in batch['image_ids']]
                batch['masks']=[FracAtlasVQADataset._boxes_to_masks(location[i],im.size)
                                for i,im in zip(batch['image_ids'],batch['images'])]
            return batch
        val_loader=DataLoader(datasets['model_val'],batch_size=4,shuffle=False,num_workers=0,collate_fn=collate)
        payload=torch.load(init['path'],map_location='cpu',weights_only=False)
        assert payload['config']==plan['model'] and len(payload['answer_vocab'])==68
        set_seed(42)
        model=BoneVQAModel(plan['model'],payload['answer_vocab'],device='cpu')
        model.load_state_dict(payload['state'],strict=False)
        current=model.trainable_state_dict()
        assert current.keys()==payload['state'].keys()
        assert all(torch.equal(current[k],v) for k,v in payload['state'].items())
        del payload,current;gc.collect()
        params=[p for p in model.parameters() if p.requires_grad]
        named=dict(model.named_parameters())
        selected=['prompt_mixer.gamma_visual','prompt_mixer.visual_proj.0.weight'] if arm.startswith('V_') else ['lens_fusion.view_emb',next(k for k in named if k.startswith('visual.') and 'lora_B' in k)]
        history=[];global_step=0;best=None
        for phase in plan['phases']:
            stage=phase['stage'];cap=phase['optimizer_steps_cap'];phase_updates=0
            optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.98),weight_decay=.01)
            warmup=int(.05*cap)
            def lr_lambda(step):
                if step<warmup:return (step+1)/max(1,warmup)
                return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,cap-warmup))))
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
            before={k:named[k].detach().clone() for k in selected}
            grads={k:0. for k in selected}
            for local_epoch in range(phase['epochs']):
                epoch=phase['global_epoch_offset']+local_epoch
                seed_training_epoch(42,epoch);model.train();optimizer.zero_grad(set_to_none=True)
                train_loader=DataLoader(datasets['train'],batch_sampler=schedules[epoch]['ordered_indices'],num_workers=0,collate_fn=collate)
                running={};skipped=0;epoch_started=time.perf_counter()
                for it,batch in enumerate(train_loader):
                    out=model(batch,stage=stage)
                    assert torch.isfinite(out['loss']), 'Nonfinite training loss'
                    loss=out['loss']/4
                    if not loss.requires_grad:
                        skipped+=1
                        loss=sum(p.reshape(-1)[0]*0. for p in params)
                    loss.backward()
                    for k in selected:
                        if named[k].grad is not None:
                            assert torch.isfinite(named[k].grad).all()
                            grads[k]=max(grads[k],float(named[k].grad.abs().max()))
                    for k in ('loss','loss_cls','loss_lm','loss_cs','loss_con','loss_lens_align','loss_type','loss_fracture','visual_contribution_norm','lens_contribution_norm'):
                        running[k]=running.get(k,0.)+float(out[k].detach())
                    if (it+1)%4==0 or it+1==len(train_loader):
                        assert_update_budget(phase_updates,global_step,cap)
                        norm=torch.nn.utils.clip_grad_norm_(params,1.)
                        assert torch.isfinite(norm), 'Nonfinite global gradient norm'
                        optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
                        phase_updates+=1;global_step+=1
                        state.update(status='running',stage=stage,global_epoch=epoch,phase_updates=phase_updates,
                                     global_step=global_step,microbatch=it+1,elapsed_seconds=time.perf_counter()-started,
                                     mean_loss=running['loss']/(it+1),updated_utc=now())
                        dump(run_dir/'progress.json',state)
                        if global_step%10==0:print(json.dumps(state),flush=True)
                assert len(train_loader)==294 and skipped<294
                row={'global_epoch':epoch,'stage':stage,'global_step':global_step,
                     'train_seconds':time.perf_counter()-epoch_started,'microbatches':294,
                     'skipped_no_supervision_batches':skipped,
                     **{'train_'+k:v/294 for k,v in running.items()}}
                with torch.no_grad():
                    metrics,predictions=evaluate_split(model,val_loader,stage,None,None,torch.device('cpu'),component_lookup=lookup)
                assert finite_tree(metrics)
                if stage==3:assert len(predictions)==142
                row.update(metrics);history.append(row)
                extra={'scope':release['scope'],'arm':arm,'stage':stage,'global_step':global_step,
                       'history':history,'base_seed':42,'execution_release_sha256':sha(OUT/'execution_release.json'),
                       'initialization_sha256':init['sha256'],'primary_is_LAST':True}
                model.save(str(run_dir/'last.pt'),extra)
                torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                            'global_step':global_step,'phase_updates':phase_updates,'stage':stage,
                            'global_epoch':epoch,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}},run_dir/'optimizer.pt')
                if stage==3:
                    score=metrics['fracture_auprc']
                    if best is None or score>best:
                        best=score;model.save(str(run_dir/'best_secondary.pt'),extra)
                    dump(run_dir/('model_val_epoch'+str(epoch)+'.json'),predictions)
                dump(run_dir/'history.json',history)
                assert phase_updates==74*(local_epoch+1)
                print(json.dumps(row),flush=True)
            assert phase_updates==cap
            updates={k:float((named[k].detach()-before[k]).abs().max()) for k in selected}
            assert all(v>0 for v in grads.values()) and all(v>0 for v in updates.values())
            dump(run_dir/('stage'+str(stage)+'_module_qc.json'),{'gradient_max':grads,'actual_update_max':updates,'finite':finite_tree(model.trainable_state_dict())})
            del optimizer,scheduler,before;gc.collect()
        assert global_step==296 and finite_tree(model.trainable_state_dict())
        assert sha(init['path'])==init['sha256']
        state.update(status='complete_pending_independent_evaluation',global_step=global_step,
                     elapsed_seconds=time.perf_counter()-started,finished_utc=now(),last_sha256=sha(run_dir/'last.pt'),
                     optimizer_sha256=sha(run_dir/'optimizer.pt'),calibration_or_test_used=False)
        dump(run_dir/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error),finished_utc=now())
        dump(run_dir/'progress.json',state)
        raise
    finally:
        # Only this newly created scope lock; preserve all run/failure artifacts.
        (OUT/'ACTIVE.lock').unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight',action='store_true')
    group.add_argument('--run',choices=ARMS)
    args=parser.parse_args()
    preflight() if args.preflight else run(args.run)
