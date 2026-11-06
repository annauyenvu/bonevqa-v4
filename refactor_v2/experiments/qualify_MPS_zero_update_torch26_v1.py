"""Fixed four-TRAIN CPU/MPS parity and MPS reseed probe, no optimizer.

Eval parity isolates stochastic dropout. TRAIN reseed is NOT optimizer/resume
certification and never releases long training or a device swap mid-experiment.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/MPS_zero_update_qualification_torch26_v1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists()
    parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
    cfgpath=read(parent/'protocol.json')['configs']['token_mean']['path'];cfg=read(cfgpath)
    checkpoint=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
    fixture=ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/protocol.json'
    train=Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train'])
    deps={str(Path(__file__).resolve()):sha(__file__)}
    deps.update(read(parent/'runtime_binding.json')['dependency_sha256'])
    for path in [Path(cfgpath),train,fixture]:deps[str(path)]=sha(path)
    assert sha(checkpoint)=='c34779eb82b9ec2dd30dac49823386afeadf2c7b7add9f60dee0715a96309626'
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Mixed4TRAIN zero-update runtime qualification only',
        'dependencies_sha256':deps,'config_path':cfgpath,'checkpoint':str(checkpoint),
        'checkpoint_sha256':sha(checkpoint),'indices':read(fixture)['TRAIN_indices'],
        'private_input':'TRAIN images/questions/answers/answer_types only',
        'public_input_keys':['images','questions'],'CPU_threads':8,'interop_threads':1,
        'devices':['cpu','mps'],'dtype':'float32','gradient_checkpointing':False,
        'eval_repeats':{'cpu':1,'mps':2},'MPS_TRAIN_reseed_repeats':2,
        'seed':42,'CPU_MPS_loss_atol':1e-5,'CPU_MPS_gradient_atol':1e-5,
        'CPU_MPS_gradient_rtol':1e-3,'CPU_MPS_public_probability_atol':1e-4,
        'public_answers_must_match':True,'MPS_repeat_gradients_loss_public_exact_required':True,
        'fresh_process_recorded_results_exact_required':True,
        'optimizer_updates':0,'mid_epoch_resume_certified':False,
        'quality_training_released':False,'no_automatic_retry':True})
    print('Prepared bounded MPS parity/reseed protocol, no models')


def measure():
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();torch.set_num_interop_threads(1)
    p=read(OUT/'protocol.json')
    for path,digest in p['dependencies_sha256'].items():assert sha(path)==digest,path
    assert sha(p['checkpoint'])==p['checkpoint_sha256'] and torch.backends.mps.is_available()
    cfg=read(p['config_path'])
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    full=collate_fn([ds[i] for i in p['indices']])
    private={k:full[k] for k in ('images','questions','answers','answer_types')}
    public={k:full[k] for k in ('images','questions')}
    model=BoneVQAModel.load(p['checkpoint'],device='cpu').eval()
    assert not model.cfg['grad_checkpoint'] and not model.cfg['llm_grad_checkpoint']
    def digest():return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    before=digest();names=None;cpu=None;mps_eval=None;mps_train=None;records=[]
    for device,training in [('cpu',False),('mps',False),('mps',False),('mps',True),('mps',True)]:
        model.to(device);model.train(training);set_seed(42);model.zero_grad(set_to_none=True)
        params=dict(model.named_parameters())
        if names is None:names=['qformer.query','prefix_proj.0.weight','fracture_head.net.3.weight',next(n for n in params if n.startswith('visual.') and 'lora_B' in n),next(n for n in params if n.startswith('llm.') and 'lora_B' in n)]
        out=model(private,stage=3);out['loss'].backward()
        if device=='mps':torch.mps.synchronize()
        loss=float(out['loss'].detach().cpu());grads={n:params[n].grad.detach().cpu().clone() for n in names}
        assert all(torch.isfinite(g).all() for g in grads.values())
        row={'device':device,'training':training,'loss':loss,'selected_gradient_norms':{n:float(g.norm()) for n,g in grads.items()}}
        if not training:
            with torch.no_grad():predictions=model.predict_batch(public)
            row['public_predictions']=[{k:v for k,v in r.items() if k in ('answer','p_yes')} for r in predictions]
            if cpu is None:cpu=(loss,grads,row['public_predictions'])
            else:
                row['CPU_loss_absolute_error']=abs(loss-cpu[0])
                row['CPU_gradient_max_errors']={n:float((g-cpu[1][n]).abs().max()) for n,g in grads.items()}
                row['CPU_loss_gradient_screen_passed']=abs(loss-cpu[0])<=p['CPU_MPS_loss_atol'] and all(torch.allclose(grads[n],cpu[1][n],atol=p['CPU_MPS_gradient_atol'],rtol=p['CPU_MPS_gradient_rtol']) for n in names)
                row['CPU_answers_equal']=[r['answer'] for r in row['public_predictions']]==[r['answer'] for r in cpu[2]]
                differences=[abs(a['p_yes']-b['p_yes']) for a,b in zip(row['public_predictions'],cpu[2]) if 'p_yes' in a and 'p_yes' in b]
                row['CPU_public_probability_max_error']=max(differences,default=0.)
                if mps_eval is None:mps_eval=(loss,grads,row['public_predictions'])
                else:row['MPS_eval_repeat_exact']=loss==mps_eval[0] and all(torch.equal(grads[n],mps_eval[1][n]) for n in names) and row['public_predictions']==mps_eval[2]
        else:
            if mps_train is None:mps_train=(loss,grads)
            else:row['MPS_TRAIN_manual_reseed_loss_and_selected_gradients_exact']=loss==mps_train[0] and all(torch.equal(grads[n],mps_train[1][n]) for n in names)
        records.append(row);del out
    model.to('cpu');assert digest()==before and sha(p['checkpoint'])==p['checkpoint_sha256']
    passed=all(r['CPU_loss_gradient_screen_passed'] and r['CPU_answers_equal'] and r['CPU_public_probability_max_error']<=p['CPU_MPS_public_probability_atol'] for r in records if r['device']=='mps' and not r['training'])
    passed=passed and records[2]['MPS_eval_repeat_exact'] and records[4]['MPS_TRAIN_manual_reseed_loss_and_selected_gradients_exact']
    del model;gc.collect()
    return {'records':records,'bounded_screen_passed':bool(passed),'all263weights_and_checkpoint_unchanged':True,
        'optimizer_updates':0,'mid_epoch_resume_certified':False,'quality_training_released':False}


def replay():
    assert measure()==read(OUT/'results.json')
    dump(OUT/'fresh_process_replay.json',{'passed':True,'recorded_results_and_internal_selected_gradient_repeat_checks_exact':True,'optimizer_updates':0})


def run():
    assert not (OUT/'progress.json').exists();dump(OUT/'progress.json',{'status':'running'});begin=time.perf_counter()
    try:
        result=measure();dump(OUT/'results.json',result)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        report={'protocol_sha256':sha(OUT/'protocol.json'),'results_sha256':sha(OUT/'results.json'),
            'replay_sha256':sha(OUT/'fresh_process_replay.json'),'bounded_screen_passed':result['bounded_screen_passed'],
            'elapsed_seconds_including_replay':time.perf_counter()-begin,'independent_artifact_audit_pending':True,
            'optimizer_updates':0,'quality_training_released':False,'mid_epoch_resume_certified':False}
        dump(OUT/'report.json',report);dump(OUT/'progress.json',{'status':'replayed_pending_independent_artifact_audit','report_sha256':sha(OUT/'report.json')})
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
