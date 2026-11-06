"""Tiny real-model capacity gate, not a quality pilot or production loss change.

Same task-trained rawB warm start for two16TRAIN-fixture controls,64updates each.
All smoke weights discarded. No validation/test, no early stopping or retries.
"""
import argparse
from collections import Counter,defaultdict
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from src.context_controls import load_component_lookup
from src.data import collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.train import seed_training_epoch
from src.utils import normalize_answer,set_seed
from experiments.run_T_bounded_current_v3_1 import data_and_schedule,hashes
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.audit_B_fullTRAIN_errors_v3_1 import correct

ROOT=Path.cwd();OUT=ROOT/'refactor_v2/reports/rare_TRAIN_capacity_v3_1'
CHECKPOINT=ROOT/'refactor_v2/checkpoints/fracatlas_B_roco_scale_pilot_v3/last.pt'
EXPECTED='94594edf64f8ef3fd7ae974e290b86900e186661d03c09e68d90e34c51ff3ddf'
MODES=('native_total','answer_only');SNAPSHOTS=(0,16,32,64)


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def fixture():
    base,datasets,manifests=data_and_schedule();rows=records_of(datasets['train'])
    counts=Counter(normalize_answer(r['answer']) for r in rows)
    binding=base['train']['component_batching']
    lookup,_=load_component_lookup(binding['manifest'],binding['seal'],ROOT)
    candidates=defaultdict(list)
    for i,row in enumerate(rows):
        if row['answer_type']=='OPEN' and counts[normalize_answer(row['answer'])]<=5:
            candidates[row['question']].append(i)
    indices=[];used=set()
    while len(indices)<16:
        before=len(indices)
        for question in sorted(candidates):
            index=next((i for i in candidates[question] if lookup[rows[i]['image_id']] not in used),None)
            if index is not None:
                indices.append(index);used.add(lookup[rows[index]['image_id']])
            if len(indices)==16:break
        assert len(indices)>before,'Insufficient distinct TRAIN components'
    rng=random.Random(20261021);batches=[]
    for _ in range(16):
        permutation=list(indices);rng.shuffle(permutation)
        batches.extend([permutation[i:i+4] for i in range(0,16,4)])
    assert len(batches)==64 and all(n==16 for n in Counter(i for b in batches for i in b).values())
    return datasets['train'],manifests,indices,batches


def idle():
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if Path(parts[1].split()[0]).name.lower() in ('python','python3','python3.11'):
            assert ' -m src.train ' not in parts[1] and ' worker ' not in parts[1]
            assert 'gate_rare_TRAIN_capacity_v3_1.py run' not in parts[1]


def prepare():
    check_environment();assert not OUT.exists();assert sha(CHECKPOINT)==EXPECTED
    dataset,manifests,indices,batches=fixture()
    dependencies={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for path in [Path(__file__),ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py',
        ROOT/'refactor_v2/experiments/audit_B_fullTRAIN_errors_v3_1.py',
        ROOT/'refactor_v2/reports/B_loss_TRAIN_diagnosis_v3_1/independent_artifact_audit.json',
        ROOT/'refactor_v2/reports/TRAIN_token_budget_v3_1.json']:
        dependencies[str(path.resolve())]=sha(path)
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Tiny rareTRAIN technical capacity ONLY, not validation or a new candidate release',
        'dependency_sha256':dependencies,'checkpoint_sha256':EXPECTED,
        'initialization':'Task-trained rawB LAST296 on the SAME TRAIN, identical warm start both arms; NOT clean PREtask quality initialization',
        'actual_manifests':manifests,'TRAIN_indices':indices,
        'selection':'Existing OPEN GT frequency<=5 across TRAIN; sorted-question round robin, first unused component; no prediction-based selection',
        'fixture_metadata':[records_of(dataset)[i] for i in indices],
        'modes':list(MODES),'schedule':batches,'updates_each':64,'batch_size':4,'exposures_per_row_each':16,
        'optimizer':{'name':'AdamW','lr':1e-4,'betas':[.9,.98],'weight_decay':.01,'clip_norm':1.,'schedule':'constant'},
        'snapshots':list(SNAPSHOTS),'primary_capacity_rule':'All16 public decoded answers exact at LAST64; otherwise not demonstrated within this cap',
        'objectives':'native_total versus exposed original loss_answer; identical data/order/RNG/parameters/budget; no runtime or production loss change',
        'public_input_keys':['images','questions'],'teacher_forcing':'TRAIN GT permitted only in private TRAIN loss',
        'weights_saved_or_reused':False,'validation_calibration_test_inference_allowed':False,
        'do_not_infer':'Failure at64updates does not prove the architecture cannot fit; success does not prove generalization or justify auxiliary loss removal',
        'long_training_authorized':False})
    print('Prepared16component64update-per-control disposable capacity gate, no model constructed')


def validate():
    check_environment();protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest,path
    assert sha(CHECKPOINT)==protocol['checkpoint_sha256']==EXPECTED
    dataset,manifests,indices,batches=fixture()
    assert manifests==protocol['actual_manifests'] and indices==protocol['TRAIN_indices'] and batches==protocol['schedule']
    return protocol,dataset


@torch.no_grad()
def evaluate(model,dataset,indices):
    model.eval();before=hashes(model.trainable_state_dict());records=[];losses=[]
    for offset in range(0,16,4):
        batch=collate_fn([dataset[i] for i in indices[offset:offset+4]])
        # Private teacher-forced diagnostic, then a separate public-input call.
        output=model(batch,stage=3);losses.append(float(output['loss_lm']))
        predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
        for j,pred in enumerate(predictions):
            records.append({'image_id':batch['image_ids'][j],'question':batch['questions'][j],
                'gt':batch['answers'][j],'answer_type':batch['answer_types'][j],
                'pred':pred['answer'],'actual_predicted_route':pred['answer_type']})
    assert hashes(model.trainable_state_dict())==before
    return {'correct':sum(correct(r) for r in records),'n':16,'teacher_forced_token_CE_mean':sum(losses)/4,'records':records}


def run():
    idle();protocol,dataset=validate();assert not (OUT/'progress.json').exists()
    started=time.perf_counter();results={};initial=None
    state={'status':'running_disposable_capacity_gate','pid':os.getpid(),'completed_modes':[]};dump(OUT/'progress.json',state)
    try:
        for mode in MODES:
            state.update(mode=mode,step=0);dump(OUT/'progress.json',state)
            set_seed(42);model=BoneVQAModel.load(str(CHECKPOINT),device='cpu')
            assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
            start_hashes=hashes(model.trainable_state_dict());assert len(start_hashes)==263
            snapshots={'0':evaluate(model,dataset,protocol['TRAIN_indices'])}
            if initial is None:initial={'hashes':start_hashes,'snapshot':snapshots['0']}
            else:assert start_hashes==initial['hashes'] and snapshots['0']==initial['snapshot']
            params=dict(model.named_parameters())
            names=['qformer.query','prefix_proj.0.weight',next(k for k in params if k.startswith('visual.') and 'lora_B' in k),
                next(k for k in params if k.startswith('llm.') and 'lora_B' in k)]
            before={k:params[k].detach().clone() for k in names};tracked=[]
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.98),weight_decay=.01)
            for step,indices in enumerate(protocol['schedule'],1):
                seed_training_epoch(42,step-1);model.train();optimizer.zero_grad(set_to_none=True)
                batch=collate_fn([dataset[i] for i in indices]);output=model(batch,stage=3)
                objective=output['loss'] if mode=='native_total' else output['loss_answer']
                assert torch.isfinite(objective)
                if step in (1,32,64):
                    gradients=torch.autograd.grad(output['loss_answer'],[params[k] for k in names],retain_graph=True,allow_unused=True)
                    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
                    maxima={k:float(g.abs().max()) for k,g in zip(names,gradients)}
                    assert all(v>0 for v in maxima.values())
                    tracked.append({'step':step,'answer_gradient_maxima':maxima})
                objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                del output,objective
                if step in SNAPSHOTS:
                    assert finite_tree(model.trainable_state_dict())
                    snapshots[str(step)]=evaluate(model,dataset,protocol['TRAIN_indices'])
                state['step']=step;dump(OUT/'progress.json',state)
            updates={k:float((params[k].detach()-before[k]).abs().max()) for k in names};assert all(v>0 for v in updates.values())
            results[mode]={'snapshots':snapshots,'tracked_answer_gradients':tracked,'actual_selected_updates':updates,
                'capacity_rule_passed_at_LAST64':snapshots['64']['correct']==16,
                'disposable_updates':64,'smoke_weights_discarded':True,'weights_saved':False}
            dump(OUT/(mode+'.json'),results[mode]);state['completed_modes'].append(mode);dump(OUT/'progress.json',state)
            del model,optimizer,params,before;gc.collect()
        assert sha(CHECKPOINT)==EXPECTED
        report={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),'results':results,
            'all263_initial_tensors_and_initial_predictions_equal':True,'elapsed_seconds':time.perf_counter()-started,
            'total_disposable_updates':128,'quality_training_updates':0,'weights_saved':False,
            'reserved_role_inference':False,'generalization_or_clinical_gain_claimed':False,
            'long_training_authorized':False,'step3_complete':False}
        dump(OUT/'report.json',report);state.update(status='complete_capacity_diagnostic_not_quality',report_sha256=sha(OUT/'report.json'));dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(OUT/'progress.json',state);raise


def launch():
    idle();validate();assert not (OUT/'detached_launch.json').exists() and not (OUT/'progress.json').exists()
    with (OUT/'detached_driver.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,str(Path(__file__).resolve()),'run'],
            cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(OUT/'detached_launch.json',{'launch_pid':child.pid,'protocol_sha256':sha(OUT/'protocol.json'),'start_new_session':True,'no_automatic_retry':True})
    print('Detached tiny capacity gate PID',child.pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','launch']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'launch':launch}[args.action]()
