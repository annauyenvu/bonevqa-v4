"""Vision trunk/head only on MPS, text/fusion/QFormer/LLM CPU; zero updates.

Preprocessing stays CPU, cross-device copies remain differentiable. Never move
the full shared BiomedCLIP object: its text encoder is shared with CPU text.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/hybrid_vision_MPS_qualification_torch26_v1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists()
    parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
    cfgpath=read(parent/'protocol.json')['configs']['token_mean']['path'];cfg=read(cfgpath)
    fixture=ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/protocol.json'
    checkpoint=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
    deps={str(Path(__file__).resolve()):sha(__file__)}
    deps.update(read(parent/'runtime_binding.json')['dependency_sha256'])
    for path in [Path(cfgpath),fixture,Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train'])]:deps[str(path)]=sha(path)
    assert sha(checkpoint)=='c34779eb82b9ec2dd30dac49823386afeadf2c7b7add9f60dee0715a96309626'
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Fixed4TRAIN hybrid routing/speed/gradient feasibility, not qualityrelease',
        'dependencies_sha256':deps,'config':cfgpath,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'indices':read(fixture)['TRAIN_indices'],'public_input_keys':['images','questions'],
        'GPU_modules':['visual.trunk','visual.head','visual.proj_tokens'],
        'CPU_modules':'Shared text encoder, original preprocessing, QFormer, fusion, heads, LLM and all remaining modules',
        'dtype':'float32','CPU_threads':8,'interop_threads':1,
        'mode_order':[['cpu',False],['hybrid',False],['hybrid',False],['cpu',True],['cpu',True],['cpu',True],['hybrid',True],['hybrid',True],['hybrid',True]],
        'eval_loss_atol':1e-5,'eval_gradient_atol':1e-5,'eval_gradient_rtol':1e-3,
        'public_probability_atol':1e-4,'public_answers_exact_required':True,
        'internal_eval_and_reseed_repeat_exact_required':True,
        'fresh_process_recorded_science_results_exact_required':True,
        'timing':'FirstTRAINpassperdevicewarmup;medianremainingtwoTRAINfwdbackpasses;includesdevicecopies/synchronizationnotmodelsetup',
        'optimizer_updates':0,'new_weights_saved':False,'calibration_test_or_new_modelval_inference':False,
        'quality_training_released':False,'update_or_resume_certified':False,'no_automatic_retry':True})
    print('Prepared hybrid4TRAIN protocol, no model launched')


def measure():
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();torch.set_num_interop_threads(1);p=read(OUT/'protocol.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    assert torch.backends.mps.is_available() and sha(p['checkpoint'])==p['checkpoint_sha256']
    cfg=read(p['config']);ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    full=collate_fn([ds[i] for i in p['indices']]);private={k:full[k] for k in ('images','questions','answers','answer_types')}
    public={k:full[k] for k in p['public_input_keys']}
    model=BoneVQAModel.load(p['checkpoint'],device='cpu').eval()
    assert not model.cfg['grad_checkpoint'] and not model.cfg['llm_grad_checkpoint']
    original=model.visual.forward
    def gpu_forward(pixel):
        result=original(pixel.to('mps'))
        return {k:v.to('cpu') if torch.is_tensor(v) else v for k,v in result.items()}
    active='cpu'
    def route(device):
        nonlocal_unused=None
        for module in (model.visual.trunk,model.visual.head,model.visual.proj_tokens):module.to('mps' if device=='hybrid' else 'cpu')
        model.visual.forward=gpu_forward if device=='hybrid' else original
        assert next(model.llm.parameters()).device.type=='cpu'
        assert next(model.text.parameters()).device.type=='cpu'
        assert next(model.fusion.parameters()).device.type=='cpu'
    def state():return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    before=state();params=dict(model.named_parameters())
    names=['qformer.query','prefix_proj.0.weight','fracture_head.net.3.weight',next(n for n in params if n.startswith('visual.') and 'lora_B' in n),next(n for n in params if n.startswith('llm.') and 'lora_B' in n)]
    reference=None;repeat_eval=None;train_refs={};records=[];timings={'cpu':[],'hybrid':[]}
    for device,training in p['mode_order']:
        if device!=active:route(device);active=device
        model.train(training);set_seed(42);model.zero_grad(set_to_none=True)
        torch.mps.synchronize();started=time.perf_counter()
        out=model(private,stage=3);out['loss'].backward();torch.mps.synchronize()
        seconds=time.perf_counter()-started
        params=dict(model.named_parameters());loss=float(out['loss'].detach().cpu())
        grads={n:params[n].grad.detach().cpu().clone() for n in names}
        assert all(torch.isfinite(g).all() for g in grads.values())
        row={'device':device,'training':training,'loss':loss,'selected_gradient_norms':{n:float(g.norm()) for n,g in grads.items()}}
        if not training:
            with torch.no_grad():predictions=model.predict_batch(public)
            row['public_predictions']=[{k:v for k,v in result.items() if k in ('answer','p_yes')} for result in predictions]
            if reference is None:reference=(loss,grads,row['public_predictions'])
            else:
                row['CPU_loss_error']=abs(loss-reference[0]);row['CPU_gradient_max_errors']={n:float((grads[n]-reference[1][n]).abs().max()) for n in names}
                row['CPU_loss_gradient_screen_passed']=abs(loss-reference[0])<=p['eval_loss_atol'] and all(torch.allclose(grads[n],reference[1][n],atol=p['eval_gradient_atol'],rtol=p['eval_gradient_rtol']) for n in names)
                row['CPU_answers_exact']=[x['answer'] for x in row['public_predictions']]==[x['answer'] for x in reference[2]]
                row['CPU_probability_max_error']=max((abs(a['p_yes']-b['p_yes']) for a,b in zip(row['public_predictions'],reference[2]) if 'p_yes' in a and 'p_yes' in b),default=0.)
                if repeat_eval is None:repeat_eval=(loss,grads,row['public_predictions'])
                else:row['hybrid_eval_repeat_exact']=loss==repeat_eval[0] and all(torch.equal(grads[n],repeat_eval[1][n]) for n in names) and row['public_predictions']==repeat_eval[2]
        else:
            timings[device].append(seconds)
            if device not in train_refs:train_refs[device]=(loss,grads)
            else:row['same_device_manual_reseed_exact']=loss==train_refs[device][0] and all(torch.equal(grads[n],train_refs[device][1][n]) for n in names)
        records.append(row);del out
    route('cpu');assert state()==before and sha(p['checkpoint'])==p['checkpoint_sha256']
    passed=all(row['CPU_loss_gradient_screen_passed'] and row['CPU_answers_exact'] and row['CPU_probability_max_error']<=p['public_probability_atol'] for row in records[1:3])
    passed=passed and records[2]['hybrid_eval_repeat_exact'] and all(row.get('same_device_manual_reseed_exact',True) for row in records)
    medians={k:statistics.median(v[1:]) for k,v in timings.items()}
    return {'science':{'records':records,'bounded_screen_passed':bool(passed),'all263weights_and_checkpoint_unchanged':True,
        'optimizer_updates':0,'quality_training_released':False,'update_or_resume_certified':False},
        'timing':{'TRAIN_seconds_including_warmup':timings,'warmed_medians':medians,'CPU_over_hybrid_speedup':medians['cpu']/medians['hybrid']}}


def replay():
    result=measure();assert result['science']==read(OUT/'results.json')['science']
    dump(OUT/'fresh_process_replay.json',{'passed':True,'recorded_science_results_exact':True,'timing_not_compared':True,'optimizer_updates':0})


def run():
    assert not (OUT/'progress.json').exists();dump(OUT/'progress.json',{'status':'running'});begin=time.perf_counter()
    try:
        result=measure();dump(OUT/'results.json',result)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        report={'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
            'replay_sha256':sha(OUT/'fresh_process_replay.json'),'bounded_screen_passed':result['science']['bounded_screen_passed'],
            'timing':result['timing'],'elapsed_seconds_including_replay':time.perf_counter()-begin,
            'independent_artifact_audit_pending':True,'optimizer_updates':0,'quality_training_released':False,'update_or_resume_certified':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'replayed_pending_independent_artifact_audit','report_sha256':sha(OUT/'report.json')})
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
