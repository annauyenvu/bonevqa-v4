"""Sealed two-arm bounded loss screen using unchanged src.train internally.

Original CLI rejects these candidate configs. This exact worker may release
the recipe flag ONLY after config/source/init/schedule/gate bindings pass.
Private loss adapter never runs in model.eval or public prediction paths.
"""
import argparse
from contextlib import contextmanager
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from src.data import collate_fn
from src.models.bonevqa import BoneVQAModel,DEFAULT_MODEL_CFG
from src.train import seed_training_epoch
from src.utils import set_seed
from experiments.gate_B_sequence_loss_models_v3_1 import OUT as GATES,MODES,validate as validate_gates,table_of_weights
from experiments.run_T_bounded_current_v3_1 import ROOT,BASE,hashes
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss

OUT=ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_1'


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def idle():
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    for row in rows:
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        executable=Path(parts[1].split()[0]).name.lower()
        if executable in ('python','python3','python3.11'):
            assert not (' -m src.train ' in parts[1] or 'run_B_sequence_loss_pilots_v3_1.py worker ' in parts[1]),'Existing training; no duplicate'


def resolved_base_model(base):
    resolved=copy.deepcopy(DEFAULT_MODEL_CFG)
    for key,value in base['model'].items():
        if isinstance(value,dict) and isinstance(resolved.get(key),dict):resolved[key].update(value)
        else:resolved[key]=copy.deepcopy(value)
    return resolved


def prepare():
    check_environment();idle();assert not OUT.exists()
    _,_,_=validate_gates()
    verified=json.loads((GATES/'independent_gate_verification.json').read_text());assert verified['passed']
    gate=json.loads((GATES/'report.json').read_text());assert verified['report_sha256']==sha(GATES/'report.json')
    base=json.loads((BASE/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    OUT.mkdir();configs={}
    for mode in MODES:
        cfg=copy.deepcopy(base);cfg['model']=resolved_base_model(base)
        cfg['run_name']='B_'+mode+'_seed42_bounded_v3_1'
        cfg['figure_dir']=str(OUT/(mode+'_figures'));cfg['init_from']=gate['results'][mode]['fresh_initial_path']
        cfg['recipe_status']='candidate_not_launch_sealed'
        cfg['sequence_loss_candidate']={'mode':mode,'gate_report_sha256':sha(GATES/'report.json'),
            'weights_path':str(ROOT/'refactor_v2/reports/B_sequence_loss_candidate_v3_1/TRAIN_OPEN_weights.json'),
            'weights_sha256':sha(ROOT/'refactor_v2/reports/B_sequence_loss_candidate_v3_1/TRAIN_OPEN_weights.json')}
        assert not (ROOT/cfg['checkpoint_dir']/cfg['run_name']).exists()
        path=OUT/(mode+'.json');dump(path,cfg)
        configs[mode]={'path':str(path),'sha256':sha(path),'initial_sha256':gate['results'][mode]['fresh_initial_sha256']}
    dependencies={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for p in [Path(__file__),GATES/'protocol.json',GATES/'report.json',GATES/'independent_gate_verification.json',
        ROOT/'refactor_v2/reports/B_sequence_loss_candidate_v3_1/protocol.json',
        ROOT/'refactor_v2/experiments/private_TRAIN_sequence_loss_adapter_v3_1.py',
        ROOT/'refactor_v2/experiments/sequence_answer_loss_candidate_v3_1.py',
        ROOT/'refactor_v2/experiments/gate_B_sequence_loss_models_v3_1.py']:
        dependencies[str(p.resolve())]=sha(p)
    dump(OUT/'plan.json',{'scope':'ONLY two B loss pilots seed42 each296updates4epochs; not long training',
        'configs':configs,'dependencies_sha256':dependencies,'styles_in_order':list(MODES),
        'quality_rule':'LAST296 primary, no best epoch/seed/threshold/template selection;142QA321inspecteddevelopment and fresh-process replay/paired components',
        'private_TRAIN_rows_expected':1173*4,'private_TRAIN_OPEN_rows_expected':534*4,
        'native_eval_rows_expected':142*4,'TRAIN_forward_calls_expected':294*4,
        'actual_update_cap_each':296,'save_steps_each_epoch':[74,148,222,296],
        'modelval_labels_not_used_for_weights':True,'calibration_test_allowed':False,
        'long_training_authorized':False,'prepared_only':True})
    print('Prepared candidate configs; ordinary CLI still rejects them; no training release yet')


def validate_plan():
    check_environment();plan=json.loads((OUT/'plan.json').read_text())
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    validate_gates()
    base=json.loads((BASE/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    initial_hashes=json.loads((GATES/'report.json').read_text())['initial_shared263_tensor_hashes']
    for mode in MODES:
        info=plan['configs'][mode];assert sha(info['path'])==info['sha256']
        cfg=json.loads(Path(info['path']).read_text())
        assert cfg['model']==resolved_base_model(base) and cfg['train']==base['train'] and cfg['data']==base['data']
        assert cfg['seed']==42 and cfg['recipe_status']=='candidate_not_launch_sealed'
        assert sha(cfg['init_from'])==info['initial_sha256']
        payload=torch.load(cfg['init_from'],map_location='cpu',weights_only=False)
        assert hashes(payload['state'])==initial_hashes and payload['config']==cfg['model']
        assert payload['extra']['scope']=='PREtask_zeroQAupdates_sequence_loss_candidate'
        assert cfg['sequence_loss_candidate']['mode']==mode
        assert sha(cfg['sequence_loss_candidate']['weights_path'])==cfg['sequence_loss_candidate']['weights_sha256']
        del payload
    gc.collect();return plan


@contextmanager
def integrated_model_adapter(mode,statistics,save_binding=None):
    original_forward=BoneVQAModel.forward;original_save=BoneVQAModel.save;table=table_of_weights()
    def forward(self,batch,stage=2):
        training=bool(self.training)
        with private_training_loss(self,batch,stage,mode,table) as trace:
            output=original_forward(self,batch,stage=stage)
        size=len(batch['questions'])
        if training:
            assert stage==3
            statistics['TRAIN_calls']+=1;statistics['TRAIN_rows']+=size
            statistics['TRAIN_OPEN_rows']+=sum(str(x).upper()!='CLOSED' for x in batch['answer_types'])
            statistics['adapted_LM_calls']+=int(trace['applied'])
            assert statistics['TRAIN_calls']<=1176
        else:
            assert not trace['applied'] and trace['calls']==0
            statistics['native_eval_rows']+=size
        return output
    def save(self,path,extra=None):
        assert save_binding is not None,'No checkpoint writes in integration smoke'
        extra=copy.deepcopy(extra or {})
        assert 1<=extra['epoch']<=4 and extra['stage']==3
        assert extra['global_step']==74*extra['epoch']<=296
        assert Path(path).resolve().parent==Path(save_binding['checkpoint_dir']).resolve()
        extra['sequence_loss_experiment']={**save_binding,'mode':mode,'statistics_at_save':dict(statistics)}
        return original_save(self,path,extra)
    BoneVQAModel.forward=forward;BoneVQAModel.save=save
    try:yield
    finally:BoneVQAModel.forward=original_forward;BoneVQAModel.save=original_save


def counters():return {'TRAIN_calls':0,'TRAIN_rows':0,'TRAIN_OPEN_rows':0,'adapted_LM_calls':0,'native_eval_rows':0}


def verify():
    idle();plan=validate_plan();destination=OUT/'independent_execution_verification.json';assert not destination.exists()
    _,datasets,indices=validate_gates();batch=collate_fn([datasets['train'][i] for i in indices])
    integration={}
    for mode in MODES:
        cfg=json.loads(Path(plan['configs'][mode]['path']).read_text())
        set_seed(42);model=BoneVQAModel.load(cfg['init_from'],device='cpu')
        before=hashes(model.trainable_state_dict());model.train()
        seed_training_epoch(42,0)
        with private_training_loss(model,batch,3,mode,table_of_weights()):
            direct=model(batch,stage=3)
        values={k:float(direct[k].detach()) for k in ('loss','loss_answer','loss_lm')};del direct
        statistics=counters();seed_training_epoch(42,0);model.train()
        with integrated_model_adapter(mode,statistics):
            integrated=model(batch,stage=3)
            assert values=={k:float(integrated[k].detach()) for k in values}
            del integrated
            model.eval()
            with torch.no_grad():native=model(batch,stage=3)
            assert torch.isfinite(native['loss'])
            del native
        assert statistics['TRAIN_rows']==4 and statistics['TRAIN_OPEN_rows']==2 and statistics['adapted_LM_calls']==1
        assert statistics['native_eval_rows']==4 and statistics['TRAIN_calls']==1
        assert hashes(model.trainable_state_dict())==before
        integration[mode]={'passed':True,'manual_vs_trainer_integration_loss_values_exact':True,
            'eval_bypass_native':True,'weights_unchanged':True,'statistics':statistics}
        del model;gc.collect()
    dump(destination,{'passed':True,'plan_sha256':sha(OUT/'plan.json'),'integration':integration,
        'all_configs_initials_source_actualmanifests_schedules_verified':True,
        'quality_training_updates':0,'long_training_authorized':False})
    print('Independent execution/integration verification passed; no quality updates yet')


def worker(mode):
    idle();plan=validate_plan()
    release=json.loads((OUT/'execution_release.json').read_text())
    assert release['plan_sha256']==sha(OUT/'plan.json') and release['passed'] and release['allowed_modes']==list(MODES)
    assert sha(OUT/'independent_execution_verification.json')==release['verification_sha256']
    assert (OUT/'ACTIVE.lock').exists()
    info=plan['configs'][mode];cfg=json.loads(Path(info['path']).read_text());target=ROOT/cfg['checkpoint_dir']/cfg['run_name']
    assert not target.exists()
    import src.train as trainer
    original_load=trainer.load_config
    def scoped_load(path):
        assert Path(path).resolve()==Path(info['path']).resolve() and sha(path)==info['sha256']
        value=original_load(path);assert value==cfg and value['recipe_status']=='candidate_not_launch_sealed'
        value['recipe_status']='released_EXACT_twoarm_capped_loss_worker_ONLY';return value
    statistics=counters();binding={'checkpoint_dir':str(target),'plan_sha256':sha(OUT/'plan.json'),
        'config_sha256':info['sha256'],'adapter_sha256':sha(ROOT/'refactor_v2/experiments/private_TRAIN_sequence_loss_adapter_v3_1.py'),
        'weights_sha256':cfg['sequence_loss_candidate']['weights_sha256'],'execution_release_sha256':sha(OUT/'execution_release.json')}
    trainer.load_config=scoped_load;old_argv=sys.argv
    try:
        sys.argv=['src.train','--config',info['path'],'--device','cpu']
        with integrated_model_adapter(mode,statistics,binding):trainer.main()
    finally:trainer.load_config=original_load;sys.argv=old_argv
    assert statistics['TRAIN_calls']==1176 and statistics['TRAIN_rows']==4692 and statistics['TRAIN_OPEN_rows']==2136
    assert statistics['native_eval_rows']==568
    dump(OUT/(mode+'_worker_audit.json'),{'passed':True,'statistics':statistics,
        'all_TRAIN_rows_counted_native_eval_bypassed_weights':True,'long_training_authorized':False})


def run():
    idle();plan=validate_plan();assert not (OUT/'progress.json').exists()
    verified=json.loads((OUT/'independent_execution_verification.json').read_text())
    assert verified['passed'] and verified['plan_sha256']==sha(OUT/'plan.json')
    descriptor=os.open(OUT/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(descriptor)
    state={'status':'starting_two_serial_bounded_loss_pilots','pid':os.getpid(),'completed_modes':[],'long_training_authorized':False}
    dump(OUT/'progress.json',state)
    dump(OUT/'execution_release.json',{'passed':True,'scope':plan['scope'],'allowed_modes':list(MODES),
        'plan_sha256':sha(OUT/'plan.json'),'verification_sha256':sha(OUT/'independent_execution_verification.json'),
        'max_updates_each':296,'max_epochs_each':4,'long_training_authorized':False})
    try:
        for mode in MODES:
            started=time.perf_counter()
            with (OUT/(mode+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,__file__,'worker',mode],stdout=log,stderr=subprocess.STDOUT)
                state.update(status='running',mode=mode,worker_pid=child.pid);dump(OUT/'progress.json',state)
                assert child.wait()==0,mode+' failed; preserve all artifacts, no automatic retry'
            cfg=json.loads(Path(plan['configs'][mode]['path']).read_text());target=ROOT/cfg['checkpoint_dir']/cfg['run_name']
            saved=torch.load(target/'last.pt',map_location='cpu',weights_only=False)
            optimizer=torch.load(target/'optimizer.pt',map_location='cpu',weights_only=False)
            history=json.loads((target/'history.json').read_text())
            assert saved['extra']['global_step']==optimizer['global_step']==296 and len(history)==4
            assert saved['extra']['sequence_loss_experiment']['mode']==mode
            assert finite_tree(saved['state']) and finite_tree(optimizer)
            assert all(r['val_n']==142 and r['skipped_no_supervision_batches']==0 for r in history)
            assert json.loads((OUT/(mode+'_worker_audit.json')).read_text())['passed']
            dump(OUT/(mode+'_completion.json'),{'complete':True,'elapsed_timer_seconds':time.perf_counter()-started,
                'last_sha256':sha(target/'last.pt'),'optimizer_sha256':sha(target/'optimizer.pt'),
                'history':history,'independent_quality_evaluation_pending':True})
            state['completed_modes'].append(mode);dump(OUT/'progress.json',state)
            del saved,optimizer;gc.collect()
        state.update(status='two_loss_pilots_trained_pending_independent_quality_evaluation');dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(OUT/'progress.json',state);raise
    finally:(OUT/'ACTIVE.lock').unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','run','worker']);parser.add_argument('mode',nargs='?',choices=MODES)
    args=parser.parse_args()
    if args.action=='worker':assert args.mode;worker(args.mode)
    else:{'prepare':prepare,'verify':verify,'run':run}[args.action]()
