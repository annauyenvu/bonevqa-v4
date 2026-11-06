"""Exact two-arm fullfold0 worker; only gated, capped execution may enter trainer."""
import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime,timezone
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

OUT=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
MODES=('token_mean','sequence_uniform')


def now():return datetime.now(timezone.utc).isoformat()


def idle():
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        command=parts[1];exe=Path(command.split()[0]).name.lower()
        if exe in ('python','python3','python3.11'):
            assert not (' -m src.train ' in command or ('run_fullfold0_B_uniform_screen_v3_1.py worker ' in command)), 'Existing worker; no duplicate'


def validate():
    check_environment();p=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json')=='b7a02dae0455abc5535242981364125d87f3f5681e1019ff794e28031e89ab4f'
    for name,h in p['dependency_sha256'].items():assert sha(Path(name))==h
    assert sha(OUT/'component_schedule.json')==p['schedule_sha256']
    audit=json.loads((OUT/'independent_design_timing_verification.json').read_text())
    assert audit['passed'] and audit['screen_protocol_sha256']==sha(OUT/'protocol.json')
    binding=OUT/'runtime_binding.json'
    if binding.exists():
        for name,h in json.loads(binding.read_text())['dependency_sha256'].items():assert sha(Path(name))==h
    for mode in MODES:
        c=p['configs'][mode];assert sha(Path(c['path']))==c['sha256']
        cfg=json.loads(Path(c['path']).read_text());assert cfg['recipe_status']=='candidate_not_launch_sealed'
        assert cfg['seed']==42 and cfg['train']['epochs']==2 and cfg['train']['grad_accum']==4
        assert sha(Path(cfg['init_from']))==c['initial_sha256']
        assert cfg['data']['loader_kwargs']['sealed_records_by_split']['calibration'] is None
        assert cfg['data']['loader_kwargs']['sealed_records_by_split']['test'] is None
    return p


def data(cfg):
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    rows=records_of(ds);assert len(rows)==11502
    return ds,{(r['question'],normalize_answer(r['answer'])):1. for r in rows if r['answer_type']=='OPEN'}


def counters():return {'TRAIN_calls':0,'TRAIN_rows':0,'TRAIN_OPEN_rows':0,'adapted_LM_calls':0,'native_eval_rows':0}


@contextmanager
def integration(mode,table,statistics,binding=None):
    original_forward=BoneVQAModel.forward;original_save=BoneVQAModel.save;started=time.monotonic()
    def forward(self,batch,stage=2):
        if binding is not None:assert time.monotonic()-started<=10800,'Per-arm3hour execution cap reached; preserve artifacts, no retry'
        training=bool(self.training)
        with private_training_loss(self,batch,stage,mode,table) as trace:result=original_forward(self,batch,stage=stage)
        n=len(batch['questions'])
        if training:
            assert stage==3
            statistics['TRAIN_calls']+=1;statistics['TRAIN_rows']+=n
            statistics['TRAIN_OPEN_rows']+=sum(str(t).upper()!='CLOSED' for t in batch['answer_types'])
            statistics['adapted_LM_calls']+=int(trace['applied'])
            assert statistics['TRAIN_calls']<=5752 and statistics['TRAIN_rows']<=23004
        else:
            assert not trace['applied'] and trace['calls']==0
            statistics['native_eval_rows']+=n
        return result
    def save(self,path,extra=None):
        assert binding is not None,'No checkpoint writes during integration verification'
        extra=copy.deepcopy(extra or {});assert extra['epoch'] in (1,2) and extra['stage']==3
        assert extra['global_step']==719*extra['epoch']<=1438
        assert Path(path).resolve().parent==Path(binding['checkpoint_dir']).resolve()
        extra['fullfold0_uniform_screen']={**binding,'mode':mode,'statistics_at_save':dict(statistics)}
        return original_save(self,path,extra)
    BoneVQAModel.forward=forward;BoneVQAModel.save=save
    try:yield
    finally:BoneVQAModel.forward=original_forward;BoneVQAModel.save=original_save


def verify():
    idle();p=validate();destination=OUT/'trainer_integration_verification.json'
    assert not destination.exists() and not (OUT/'runtime_binding.json').exists()
    source_paths=list((ROOT/'refactor_v2/src').rglob('*.py'))+[Path(__file__).resolve(),
        ROOT/'refactor_v2/experiments/private_TRAIN_sequence_loss_adapter_v3_1.py',
        ROOT/'refactor_v2/experiments/sequence_answer_loss_candidate_v3_1.py',OUT/'protocol.json',OUT/'component_schedule.json',
        OUT/'independent_design_timing_verification.json']
    dump(OUT/'runtime_binding.json',{'dependency_sha256':{str(q.resolve()):sha(q) for q in source_paths},
        'max_updates_each':1438,'epochs_each':2,'per_arm_seconds_cap_at_next_forward':10800,'no_automatic_retry':True})
    indices=json.loads((OUT/'component_schedule.json').read_text())[0]['ordered_indices'][0];results={}
    for mode in MODES:
        cfg=json.loads(Path(p['configs'][mode]['path']).read_text());ds,table=data(cfg);batch=collate_fn([ds[i] for i in indices])
        set_seed(42);model=BoneVQAModel.load(cfg['init_from'],device='cpu');before=hashes(model.trainable_state_dict())
        model.train();seed_training_epoch(42,0)
        with private_training_loss(model,batch,3,mode,table):direct=model(batch,stage=3)
        values={k:float(direct[k].detach()) for k in ('loss','loss_answer','loss_lm')};del direct
        statistics=counters();model.train();seed_training_epoch(42,0)
        with integration(mode,table,statistics):
            integrated=model(batch,stage=3);assert values=={k:float(integrated[k].detach()) for k in values};del integrated
            model.eval()
            with torch.no_grad():native=model(batch,stage=3)
            assert torch.isfinite(native['loss']);del native
        assert statistics['TRAIN_calls']==1 and statistics['TRAIN_rows']==4 and statistics['native_eval_rows']==4
        assert hashes(model.trainable_state_dict())==before
        results[mode]={'passed':True,'manual_vs_scoped_trainer_loss_exact':True,'native_eval_bypass':True,
            'all263_weights_unchanged':True,'statistics':statistics}
        del model;gc.collect()
    validate();dump(destination,{'passed':True,'screen_protocol_sha256':sha(OUT/'protocol.json'),
        'runtime_binding_sha256':sha(OUT/'runtime_binding.json'),'results':results,
        'optimizer_updates':0,'quality_training_updates':0,'reserved_role_model_inference':False,'long_training_authorized':False})
    print('Two-arm frozen-trainer integration/native-eval verification passed, zero optimizer updates',flush=True)


def worker(mode):
    idle();p=validate();release=json.loads((OUT/'scoped_execution_release.json').read_text())
    assert release['passed'] and release['screen_protocol_sha256']==sha(OUT/'protocol.json')
    assert release['trainer_integration_sha256']==sha(OUT/'trainer_integration_verification.json')
    assert release['runtime_binding_sha256']==sha(OUT/'runtime_binding.json') and (OUT/'ACTIVE.lock').exists()
    info=p['configs'][mode];cfg=json.loads(Path(info['path']).read_text());target=ROOT/cfg['checkpoint_dir']/cfg['run_name']
    assert not target.exists();_,table=data(cfg)
    import src.train as trainer
    old_load=trainer.load_config;old_argv=sys.argv
    def exact_load(path):
        assert Path(path).resolve()==Path(info['path']).resolve() and sha(Path(path))==info['sha256']
        c=old_load(path);assert c==cfg
        c['recipe_status']='released_EXACT_fullfold0_twoarm_1438updates_ONLY';return c
    binding={'checkpoint_dir':str(target),'screen_protocol_sha256':sha(OUT/'protocol.json'),
        'config_sha256':info['sha256'],'initial_sha256':info['initial_sha256'],
        'runtime_binding_sha256':sha(OUT/'runtime_binding.json'),'scoped_release_sha256':sha(OUT/'scoped_execution_release.json')}
    statistics=counters();trainer.load_config=exact_load
    try:
        sys.argv=['src.train','--config',info['path'],'--device','cpu']
        with integration(mode,table,statistics,binding):trainer.main()
    finally:trainer.load_config=old_load;sys.argv=old_argv
    assert statistics['TRAIN_calls']==5752 and statistics['TRAIN_rows']==23004 and statistics['TRAIN_OPEN_rows']==10204
    assert statistics['native_eval_rows']==2864
    dump(OUT/(mode+'_worker_audit.json'),{'passed':True,'statistics':statistics,
        'loss_adapter_TRAIN_only_native_eval':True,'max_updates':1438,'long_training_authorized':False})


def run():
    idle();p=validate();assert not (OUT/'progress.json').exists()
    verified=json.loads((OUT/'trainer_integration_verification.json').read_text());assert verified['passed']
    assert verified['screen_protocol_sha256']==sha(OUT/'protocol.json') and verified['runtime_binding_sha256']==sha(OUT/'runtime_binding.json')
    os.close(os.open(OUT/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
    dump(OUT/'scoped_execution_release.json',{'passed':True,'screen_protocol_sha256':sha(OUT/'protocol.json'),
        'trainer_integration_sha256':sha(OUT/'trainer_integration_verification.json'),
        'runtime_binding_sha256':sha(OUT/'runtime_binding.json'),'allowed_modes':list(MODES),
        'seed':42,'epochs_each':2,'max_updates_each':1438,'per_arm_seconds_cap_at_next_forward':10800,
        'calibration_test_allowed':False,'fivefold_study_released':False,'other_long_training_authorized':False})
    state={'status':'starting','driver_pid':os.getpid(),'started_utc':now(),'completed_modes':[]}
    dump(OUT/'progress.json',state)
    try:
        for mode in MODES:
            start=time.perf_counter()
            with (OUT/(mode+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker',mode],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                state.update(status='running',mode=mode,worker_pid=child.pid,arm_started_utc=now());dump(OUT/'progress.json',state)
                assert child.wait()==0,mode+' failed; preserve artifacts, no automatic retry'
            cfg=json.loads(Path(p['configs'][mode]['path']).read_text());target=ROOT/cfg['checkpoint_dir']/cfg['run_name']
            last=torch.load(target/'last.pt',map_location='cpu',weights_only=False);optimizer=torch.load(target/'optimizer.pt',map_location='cpu',weights_only=False)
            history=json.loads((target/'history.json').read_text())
            assert last['extra']['global_step']==optimizer['global_step']==1438 and len(history)==2
            assert finite_tree(last['state']) and finite_tree(optimizer)
            assert all(r['val_n']==1432 and r['skipped_no_supervision_batches']==0 for r in history)
            assert json.loads((OUT/(mode+'_worker_audit.json')).read_text())['passed']
            dump(OUT/(mode+'_completion.json'),{'complete':True,'elapsed_seconds':time.perf_counter()-start,
                'last_sha256':sha(target/'last.pt'),'optimizer_sha256':sha(target/'optimizer.pt'),
                'independent_quality_evaluation_pending':True,'primary_is_LAST1438':True})
            state['completed_modes'].append(mode);dump(OUT/'progress.json',state)
            del last,optimizer;gc.collect()
        state.update(status='both_trained_pending_independent_quality_evaluation');dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(OUT/'progress.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify','run','worker']);parser.add_argument('mode',nargs='?',choices=MODES);args=parser.parse_args()
    worker(args.mode) if args.action=='worker' else {'verify':verify,'run':run}[args.action]()
