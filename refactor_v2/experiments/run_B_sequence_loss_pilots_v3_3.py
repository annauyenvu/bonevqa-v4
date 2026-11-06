"""Explicit user-requested continuation after v3_2 process interruption.

Preserve the interrupted run; restart both predeclared arms from the SAME clean
PREtask initialization in new directories. No mid-epoch resume or model choice.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import torch
from experiments import run_B_sequence_loss_pilots_v3_2 as previous

core=previous.core
PREVIOUS_WRAPPER=Path(previous.__file__).resolve()
PREVIOUS_FOLDER=core.ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_2'
core.__file__=str(Path(__file__).resolve())
core.OUT=core.ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_3'


def idle():
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    for row in rows:
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if Path(parts[1].split()[0]).name.lower() in ('python','python3','python3.11'):
            assert ' -m src.train ' not in parts[1]
            assert not ('run_B_sequence_loss_pilots_' in parts[1] and ' worker ' in parts[1]), 'Existing capped worker'


core.idle=idle


def prepare():
    core.check_environment();idle();assert not core.OUT.exists()
    plan=json.loads((PREVIOUS_FOLDER/'plan.json').read_text())
    for path,digest in plan['dependencies_sha256'].items():assert core.sha(path)==digest,path
    old_progress=json.loads((PREVIOUS_FOLDER/'progress.json').read_text())
    assert old_progress['status']=='running' and old_progress['completed_modes']==[]
    assert old_progress['mode']=='sequence_uniform'
    core.validate_gates()
    old_config=json.loads((PREVIOUS_FOLDER/'sequence_uniform.json').read_text())
    old_target=core.ROOT/old_config['checkpoint_dir']/old_config['run_name']
    saved=torch.load(old_target/'last.pt',map_location='cpu',weights_only=False)
    optimizer=torch.load(old_target/'optimizer.pt',map_location='cpu',weights_only=False)
    assert saved['extra']['epoch']==optimizer['epoch']==2
    assert saved['extra']['global_step']==optimizer['global_step']==148
    assert core.finite_tree(saved['state']) and core.finite_tree(optimizer)
    history=json.loads((old_target/'history.json').read_text());assert len(history)==2
    interruption={'observed_local':'2026-10-20T23:43:47+07:00',
        'old_progress_stale':old_progress,'process_check_success_no_training_worker':True,
        'cause':'Process absent after interrupted chat; exact termination cause unverified',
        'last_logged_update':167,'saved_complete_epoch':2,'saved_complete_update':148,
        'old_checkpoint_directory':str(old_target),
        'artifact_sha256':{str(p):core.sha(p) for p in [old_target/'last.pt',old_target/'optimizer.pt',
            old_target/'history.json',PREVIOUS_FOLDER/'progress.json',PREVIOUS_FOLDER/'sequence_uniform.log']},
        'old_files_deleted_or_overwritten':False,'old_partial_weights_used_as_initialization':False,
        'recovery':'Clean PREtask restart both unchanged arms in new directories, user said continue',
        'mid_epoch_resume_claimed':False,'quality_selection_from_partial_results':False}
    del saved,optimizer
    core.OUT.mkdir();core.dump(core.OUT/'interruption_audit.json',interruption)
    configs={}
    for mode in core.MODES:
        info=plan['configs'][mode];assert core.sha(info['path'])==info['sha256']
        cfg=json.loads(Path(info['path']).read_text())
        cfg['run_name']='B_'+mode+'_seed42_bounded_v3_3'
        cfg['figure_dir']=str(core.OUT/(mode+'_figures'))
        assert not (core.ROOT/cfg['checkpoint_dir']/cfg['run_name']).exists()
        path=core.OUT/(mode+'.json');core.dump(path,cfg)
        configs[mode]={'path':str(path),'sha256':core.sha(path),'initial_sha256':info['initial_sha256']}
    plan=copy.deepcopy(plan);plan['configs']=configs
    for path in [Path(__file__).resolve(),PREVIOUS_WRAPPER,PREVIOUS_FOLDER/'plan.json',
                 core.OUT/'interruption_audit.json']:
        plan['dependencies_sha256'][str(path)]=core.sha(path)
    plan['recovery']='User-requested continuation; old partial run preserved; clean PREtask restart; same scientific recipe and LAST296 endpoints'
    core.dump(core.OUT/'plan.json',plan)
    core.validate_plan()
    print('Prepared clean recovery; old interrupted files preserved, no training yet')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','run','worker'])
    parser.add_argument('mode',nargs='?',choices=core.MODES);args=parser.parse_args()
    if args.action=='worker':
        assert args.mode;core.worker(args.mode)
    else:{'prepare':prepare,'verify':core.verify,'run':core.run}[args.action]()
