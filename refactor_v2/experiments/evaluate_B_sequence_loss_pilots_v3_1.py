"""Locked LAST296 two-arm loss evaluation; inspected development, not confirmation.

Preparation constructs datasets only. Model calls wait for both capped workers.
Reuse verified public-input inference/metric contracts, not T scientific settings.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from experiments import evaluate_T_bounded_current_v3_3 as shared
from experiments import run_B_sequence_loss_pilots_v3_2 as training

core = shared.core
ROOT = Path.cwd()
OUT = ROOT/'refactor_v2/reports/B_sequence_loss_quality_v3_1'
FOLDER = training.core.OUT
MODES = ('sequence_uniform', 'sequence_rarity')
sha, dump = core.sha, core.dump


def prepare():
    core.check_environment()
    assert not OUT.exists(), 'Preserve existing preparation'
    qa, full, indices, _, full_config = core.cohorts()
    dependencies = {str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    # Bind every local experimental dependency rather than relying on imported
    # module names alone; no model/checkpoint reads in preparation.
    for p in (ROOT/'refactor_v2/experiments').glob('*.py'):
        dependencies[str(p.resolve())] = sha(p)
    for p in [full_config, FOLDER/'plan.json', FOLDER/'execution_release.json',
              FOLDER/'independent_execution_verification.json', core.BASE/'independent_evaluation.json']:
        dependencies[str(p.resolve())] = sha(p)
    baseline = json.loads((core.BASE/'independent_evaluation_progress.json').read_text())
    assert baseline['report_sha256']==sha(core.BASE/'independent_evaluation.json')
    OUT.mkdir()
    dump(OUT/'evaluation_plan.json', {
        'created_utc':core.now(), 'dependencies_sha256':dependencies,
        'QA_manifest':core.manifest(qa), 'full_modelval_manifest':core.manifest(full),
        'canonical_image_ids':[core.records_of(full)[i]['image_id'] for i in indices],
        'modes':list(MODES), 'checkpoint':'LAST296_only', 'raw_threshold':.5,
        'public_input_keys':['images','questions'], 'training_loss_adapter_enabled':False,
        'comparison_primary':'sequence_rarity', 'controls':['sequence_uniform','raw_B_seed42'],
        'bootstrap_seed':20261020, 'bootstrap_resamples':500,
        'QA_metrics':['overall_acc','open_acc'],
        'clinical_metrics':['fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity'],
        'constructor_seeds':[90210,90211], 'fresh_process_exact_replay_required':True,
        'single_seed42_training':True, 'calibration_or_test_allowed':False,
        'independent_confirmation':False, 'training_updates':0,
        'long_training_authorized':False,
        'limitations':['Repeatedly inspected development; not untouched confirmation',
            'No epoch, seed, threshold, weight or answer-template selection',
            'One paired training seed; wider confirmation remains outstanding']})
    print('Prepared locked development evaluation; no model job launched')


def validate_ready():
    core.check_environment()
    plan = json.loads((OUT/'evaluation_plan.json').read_text())
    for path,digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest, path
    assert not (FOLDER/'ACTIVE.lock').exists(), 'Wait for both training workers'
    progress = json.loads((FOLDER/'progress.json').read_text())
    assert progress['status']=='two_loss_pilots_trained_pending_independent_quality_evaluation'
    assert progress['completed_modes']==list(MODES)
    commands = subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    for row in commands.splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if Path(parts[1].split()[0]).name.lower() in ('python','python3','python3.11'):
            assert ' -m src.train ' not in parts[1] and not (
                'run_B_sequence_loss_pilots_' in parts[1] and ' worker ' in parts[1])
    qa,full,indices,lookup,_ = core.cohorts()
    assert core.manifest(qa)==plan['QA_manifest'] and core.manifest(full)==plan['full_modelval_manifest']
    assert [core.records_of(full)[i]['image_id'] for i in indices]==plan['canonical_image_ids']
    execution = json.loads((FOLDER/'plan.json').read_text())
    checkpoints={}
    for mode in MODES:
        info=execution['configs'][mode]
        assert sha(info['path'])==info['sha256']
        cfg=json.loads(Path(info['path']).read_text())
        target=ROOT/cfg['checkpoint_dir']/cfg['run_name']
        done=json.loads((FOLDER/(mode+'_completion.json')).read_text())
        assert done['complete'] and sha(target/'last.pt')==done['last_sha256']
        assert sha(target/'optimizer.pt')==done['optimizer_sha256']
        saved=torch.load(target/'last.pt',map_location='cpu',weights_only=False)
        optimizer=torch.load(target/'optimizer.pt',map_location='cpu',weights_only=False)
        history=json.loads((target/'history.json').read_text())
        assert history==done['history'] and len(history)==4
        assert all(r['val_n']==142 and r['skipped_no_supervision_batches']==0 for r in history)
        assert saved['config']==cfg['model'] and len(saved['state'])==263
        assert saved['extra']['global_step']==optimizer['global_step']==296
        assert core.finite_tree(saved['state']) and core.finite_tree(optimizer)
        provenance=saved['extra']['sequence_loss_experiment']
        assert provenance['mode']==mode and provenance['plan_sha256']==sha(FOLDER/'plan.json')
        assert provenance['config_sha256']==info['sha256']
        audit=json.loads((FOLDER/(mode+'_worker_audit.json')).read_text())
        assert audit['passed'] and audit['statistics']['TRAIN_rows']==4692
        checkpoints[mode]={'path':str(target/'last.pt'),'sha256':done['last_sha256'],'history':history}
        del saved,optimizer
    gc.collect()
    return plan,qa,full,indices,lookup,checkpoints


def evaluate(qa,full,indices,checkpoints,seed):
    for mode in MODES:
        info=checkpoints[mode];core.set_seed(seed)
        model=core.BoneVQAModel.load(info['path'],device='cpu').eval()
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt',
                                              'use_textual_prompt','use_region_in_prompt'))
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        assert not hasattr(model,'_bonevqa_sequence_loss_active')
        result={'QA142':core.predict(model,qa,list(range(len(qa)))),
                'fracture321':core.predict(model,full,indices)}
        shared.compare_QA_metrics(result['QA142']['metrics'],info['history'][-1])
        assert core.compute_metrics(result['QA142']['records'])=={
            k:v for k,v in result['QA142']['metrics'].items() if k!='fracture_yes_threshold'}
        assert sha(info['path'])==info['sha256']
        del model;gc.collect()
        yield mode,result


def replay():
    _,qa,full,indices,_,checkpoints=validate_ready()
    for mode,result in evaluate(qa,full,indices,checkpoints,90211):
        assert json.loads((OUT/(mode+'.json')).read_text())==result
    dump(OUT/'fresh_process_replay.json',{'passed':True,'constructor_seed':90211,
        'all_records_and_metrics_exact':True,'training_updates':0,
        'checkpoint_sha256':{m:x['sha256'] for m,x in checkpoints.items()},
        'calibration_or_test_used':False})


def run():
    plan,qa,full,indices,lookup,checkpoints=validate_ready()
    assert not (OUT/'progress.json').exists(), 'No automatic retry or overwrite'
    state={'status':'running','pid':os.getpid(),'started_utc':core.now()}
    dump(OUT/'progress.json',state);started=time.perf_counter()
    try:
        results={}
        for mode,result in evaluate(qa,full,indices,checkpoints,90210):
            results[mode]=result;dump(OUT/(mode+'.json'),result)
            state['completed_mode']=mode;dump(OUT/'progress.json',state)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify'],check=True)
        baseline=json.loads((core.BASE/'independent_evaluation.json').read_text())
        reference={'QA142':baseline['pilot142_QA']['fracatlas_B_roco_scale_pilot_v3/correct_image'],
                   'fracture321':baseline['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']}
        endpoints={name:{m:r[name]['records'] for m,r in results.items()} for name in reference}
        for name in endpoints:endpoints[name]['raw_B_seed42']=reference[name]['records']
        qa_boot=core.bootstrap(endpoints['QA142'],'sequence_rarity',lookup,plan['QA_metrics'])
        clinical_boot=core.bootstrap(endpoints['fracture321'],'sequence_rarity',lookup,plan['clinical_metrics'])
        assert qa_boot['component_count']==32 and clinical_boot['component_count']==314
        report={'scope':'Two predeclared LAST296 loss pilots; inspected development only',
            'evaluation_plan_sha256':sha(OUT/'evaluation_plan.json'),
            'results':{m:{k:v['metrics'] for k,v in r.items()} for m,r in results.items()},
            'raw_B_reference':{k:v['metrics'] for k,v in reference.items()},
            'QA_component_bootstrap':qa_boot,'clinical_component_bootstrap':clinical_boot,
            'checkpoint_sha256':{m:x['sha256'] for m,x in checkpoints.items()},
            'prediction_sha256':{m:sha(OUT/(m+'.json')) for m in MODES},
            'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'elapsed_seconds_including_replay':time.perf_counter()-started,
            'limitations':plan['limitations'],'raw_threshold':.5,'training_updates':0,
            'calibration_or_test_used':False,'long_training_authorized':False,
            'Final_Solution_selected':False,'production_released':False}
        dump(OUT/'report.json',report)
        state.update(status='complete_verified_development_only_NOTfull_step3',finished_utc=core.now(),report_sha256=sha(OUT/'report.json'))
        dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error))
        dump(OUT/'progress.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify'])
    args=parser.parse_args();{'prepare':prepare,'run':run,'verify':replay}[args.action]()
