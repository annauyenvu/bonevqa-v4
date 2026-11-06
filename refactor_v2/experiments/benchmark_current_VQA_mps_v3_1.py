"""Bounded CPU/MPS current TRAIN throughput screen, zero optimizer steps."""
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/current_VQA_mps_resources_v3_1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    assert not OUT.exists()
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();torch.set_num_interop_threads(1)
    OUT.mkdir();started=time.perf_counter()
    report={'scope':'Current mixed4TRAIN CPU/MPS forward-backward microbenchmark only',
        'source_sha256':sha(__file__),'optimizer_updates':0,'quality_runtime_released':False,
        'MPS_available':torch.backends.mps.is_available(),'results':{},'mid_epoch_resume_certified':False}
    try:
        assert report['MPS_available']
        parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
        cfg=read(read(parent/'protocol.json')['configs']['token_mean']['path'])
        checkpoint=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
        digest='c34779eb82b9ec2dd30dac49823386afeadf2c7b7add9f60dee0715a96309626'
        assert sha(checkpoint)==digest
        assert not cfg['model']['grad_checkpoint'] and not cfg['model']['llm_grad_checkpoint']
        ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
        indices=read(ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/protocol.json')['TRAIN_indices']
        batch=collate_fn([ds[i] for i in indices])
        model=BoneVQAModel.load(str(checkpoint),device='cpu').train()
        def state():return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
        before=state();params=dict(model.named_parameters())
        names=['qformer.query','prefix_proj.0.weight','fracture_head.net.3.weight',
            next(n for n in params if n.startswith('visual.') and 'lora_B' in n),
            next(n for n in params if n.startswith('llm.') and 'lora_B' in n)]
        reference=None
        for device in ('cpu','mps'):
            model.to(device);params=dict(model.named_parameters());times=[];losses=[];errors=[]
            for repeat in range(3):
                set_seed(42);model.zero_grad(set_to_none=True)
                if device=='mps':torch.mps.synchronize()
                begin=time.perf_counter();out=model(batch,stage=3);out['loss'].backward()
                if device=='mps':torch.mps.synchronize()
                elapsed=time.perf_counter()-begin
                loss=float(out['loss'].detach().cpu())
                grads={n:params[n].grad.detach().cpu().clone() for n in names}
                assert all(torch.isfinite(g).all() for g in grads.values())
                if reference is None:reference=(loss,grads)
                if repeat:
                    times.append(elapsed);losses.append(loss)
                    errors.append({'loss_abs_delta_vsCPU':abs(loss-reference[0]),
                        'selected_gradient_max_errors':{n:float((grads[n]-reference[1][n]).abs().max()) for n in names},
                        'selected_gradient_allclose_atol1e-5_rtol1e-3':all(torch.allclose(grads[n],reference[1][n],atol=1e-5,rtol=1e-3) for n in names)})
                del out,grads
            report['results'][device]={'seconds':times,'median_seconds':statistics.median(times),'losses':losses,'CPU_numerical_comparison':errors}
            (OUT/'progress.json').write_text(json.dumps({'completed_devices':list(report['results'])})+'\n')
        model.to('cpu');assert state()==before and sha(checkpoint)==digest
        report.update(status='benchmark_complete_not_runtime_release',trainable_weights_and_checkpoint_unchanged=True,
            throughput_ratio_CPU_over_MPS=report['results']['cpu']['median_seconds']/report['results']['mps']['median_seconds'])
    except Exception as error:
        report.update(status='failed_preserved_no_automatic_retry',failure=repr(error),traceback=traceback.format_exc())
    report['elapsed_seconds']=time.perf_counter()-started
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
