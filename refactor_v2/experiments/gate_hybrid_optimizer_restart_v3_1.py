"""Disposable two-update hybrid AdamW/export/restart gate on four TRAIN rows.

Two tiny synthetic epoch boundaries are NOT fullfold epoch or mid-epoch resume
certification. Artifacts must never initialize a quality run.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/hybrid_optimizer_restart_gate_v3_1'
PRIOR=ROOT/'refactor_v2/reports/hybrid_vision_MPS_qualification_v3_1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists()
    prior=read(PRIOR/'protocol.json');assert read(PRIOR/'report.json')['bounded_screen_passed']
    assert read(PRIOR/'independent_artifact_review.json')['report_sha256']==sha(PRIOR/'report.json')
    deps={str(Path(__file__).resolve()):sha(__file__),**prior['dependencies_sha256']}
    for p in [PRIOR/'protocol.json',PRIOR/'report.json',PRIOR/'independent_artifact_review.json']:deps[str(p)]=sha(p)
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Disposable hybrid optimizer/restart toy gate, not qualitytraining or fullfold resume',
        'dependencies_sha256':deps,'initial_checkpoint':prior['checkpoint'],
        'initial_checkpoint_sha256':prior['checkpoint_sha256'],'config':prior['config'],'indices':prior['indices'],
        'examples':4,'positive_negative_canonical_plus2OPEN':True,'batch_size':4,'grad_accum':4,
        'continuous_updates':2,'restart_replay_updates':1,'actual_optimizer_updates_total':3,
        'optimizer':{'name':'AdamW','lr':1e-4,'betas':[.9,.98],'weight_decay':.01,'max_grad_norm':1.0},
        'schedule':'Frozen trainer cosine/warmup1438 horizon, first2updates only',
        'epoch_seed_policy':'seed_training_epoch(42,0)then(42,1)atTOYoneupdateboundaries;NOmid-epochRNGrestoreclaim',
        'public_input_keys':['images','questions'],'runtime':'Visual trunk/head/proj MPS;restCPUfloat32_CPU8interop1',
        'disposable_artifacts_only':True,'no_quality_initialization_allowed':True,
        'checkpoint_export_and_fresh_process_strict_reload_required':True,
        'final263parameter_and_optimizer_scheduler_state_hashes_exact_required':True,
        'no_calibration_test_or_modelval_inference':True,'quality_training_released':False,
        'fullfold_or_mid_epoch_resume_certified':False,'no_automatic_retry':True})
    print('Prepared3actualdisposableupdates maximum, no model yet')


def validate():
    p=read(OUT/'protocol.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    assert sha(p['initial_checkpoint'])==p['initial_checkpoint_sha256']
    return p


def setup(checkpoint):
    import torch
    from src.models.bonevqa import BoneVQAModel
    model=BoneVQAModel.load(str(checkpoint),device='cpu')
    assert not model.cfg['grad_checkpoint'] and not model.cfg['llm_grad_checkpoint']
    original=model.visual.forward
    for module in (model.visual.trunk,model.visual.head,model.visual.proj_tokens):module.to('mps')
    def forward(pixel):
        result=original(pixel.to('mps'))
        return {k:v.to('cpu') if torch.is_tensor(v) else v for k,v in result.items()}
    model.visual.forward=forward
    assert next(model.llm.parameters()).device.type==next(model.text.parameters()).device.type=='cpu'
    params=[param for param in model.parameters() if param.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.98),weight_decay=.01)
    def rate(step):
        warmup=int(.05*1438)
        if step<warmup:return (step+1)/max(1,warmup)
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,1438-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    return model,optimizer,scheduler,params


def runtime():
    import torch
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    check_environment();torch.set_num_interop_threads(1);p=validate()
    c=read(p['config']);ds=build_split(build_dataset,c['data']['dataset'],'train',None,True,c['data']['loader_kwargs'])
    b=collate_fn([ds[i] for i in p['indices']])
    return p,{k:b[k] for k in ('images','questions','answers','answer_types')}


def hashes(model):return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}


def treehash(value):
    import torch
    h=hashlib.sha256()
    def feed(x):
        if torch.is_tensor(x):h.update(str((str(x.dtype),list(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):h.update(repr(k).encode());feed(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:feed(v)
        else:h.update(repr(x).encode())
    feed(value);return h.hexdigest()


def update(model,optimizer,scheduler,params,batch,epoch):
    import torch
    from src.train import seed_training_epoch
    model.train();seed_training_epoch(42,epoch);optimizer.zero_grad(set_to_none=True)
    losses=[]
    for _ in range(4):
        out=model(batch,stage=3);losses.append(float(out['loss'].detach().cpu()))
        assert torch.isfinite(out['loss']);(out['loss']/4).backward()
    gradients=[param.grad for param in params if param.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    norm=torch.nn.utils.clip_grad_norm_(params,1.0);assert torch.isfinite(norm)
    optimizer.step();scheduler.step();torch.mps.synchronize()
    assert all(torch.isfinite(param).all() for param in params)
    return {'microbatch_losses':losses,'preclip_gradient_norm':float(norm.cpu()),'lr_after_step':optimizer.param_groups[0]['lr']}


def replay():
    import torch
    p,batch=runtime();model,optimizer,scheduler,params=setup(OUT/'DISPOSABLE_after_update1.pt')
    saved=torch.load(OUT/'DISPOSABLE_optimizer1.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    expected=read(OUT/'continuous_reference.json')
    assert hashes(model)==expected['after_first_parameter_hashes']
    assert treehash(optimizer.state_dict())==expected['after_first_optimizer_hash']
    assert treehash(scheduler.state_dict())==expected['after_first_scheduler_hash']
    step=update(model,optimizer,scheduler,params,batch,1)
    assert step==expected['second_update']
    assert hashes(model)==expected['final_parameter_hashes']
    assert treehash(optimizer.state_dict())==expected['final_optimizer_hash']
    assert treehash(scheduler.state_dict())==expected['final_scheduler_hash']
    dump(OUT/'fresh_process_restart.json',{'passed':True,'all263final_parameter_hashes_exact':True,
        'optimizer_scheduler_states_exact':True,'second_update_loss_norm_LR_exact':True,
        'optimizer_updates_this_process':1,'scope':'Toyepochboundary_reseedonly_NOTmid_epochorfullfoldresume'})


def run():
    import torch
    p,batch=runtime();assert not (OUT/'progress.json').exists()
    dump(OUT/'progress.json',{'status':'running_disposable_gate','optimizer_updates_completed_in_driver':0});started=time.perf_counter()
    try:
        model,optimizer,scheduler,params=setup(p['initial_checkpoint']);before=hashes(model)
        first=update(model,optimizer,scheduler,params,batch,0)
        after_first=hashes(model);assert any(after_first[k]!=before[k] for k in before)
        model.save(str(OUT/'DISPOSABLE_after_update1.pt'),extra={'disposable_technical_gate':True,'quality_initialization_forbidden':True,'global_step':1})
        torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict()},OUT/'DISPOSABLE_optimizer1.pt')
        reference={'first_update':first,'after_first_parameter_hashes':after_first,
            'after_first_optimizer_hash':treehash(optimizer.state_dict()),'after_first_scheduler_hash':treehash(scheduler.state_dict())}
        reference['second_update']=update(model,optimizer,scheduler,params,batch,1)
        reference.update(final_parameter_hashes=hashes(model),final_optimizer_hash=treehash(optimizer.state_dict()),final_scheduler_hash=treehash(scheduler.state_dict()))
        dump(OUT/'continuous_reference.json',reference)
        del model,optimizer,scheduler,params;gc.collect();torch.mps.empty_cache()
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        assert sha(p['initial_checkpoint'])==p['initial_checkpoint_sha256']
        report={'protocol_sha256':sha(OUT/'protocol.json'),'reference_sha256':sha(OUT/'continuous_reference.json'),
            'restart_sha256':sha(OUT/'fresh_process_restart.json'),'passed':True,
            'elapsed_seconds_including_restart':time.perf_counter()-started,'actual_optimizer_updates':3,
            'all_smoke_updated_weights_disposable':True,'initial_quality_checkpoint_unchanged':True,
            'disposable_export_sha256':sha(OUT/'DISPOSABLE_after_update1.pt'),
            'disposable_optimizer_sha256':sha(OUT/'DISPOSABLE_optimizer1.pt'),
            'independent_artifact_audit_pending':True,'quality_training_released':False,'fullfold_or_mid_epoch_resume_certified':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'toy_restart_passed_pending_independent_audit','report_sha256':sha(OUT/'report.json')})
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
