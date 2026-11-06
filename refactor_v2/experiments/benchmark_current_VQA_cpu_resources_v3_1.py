"""One resident model; warmed mixed-TRAIN CPU8/10/12 timing, zero updates."""
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/current_VQA_cpu_resources_v3_1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    assert not OUT.exists(),'Preserve previous benchmark'
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();torch.set_num_interop_threads(1)
    parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
    config=read(read(parent/'protocol.json')['configs']['token_mean']['path'])
    fixture=ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/protocol.json'
    indices=read(fixture)['TRAIN_indices']
    assert len(indices)==4
    checkpoint=ROOT/config['checkpoint_dir']/config['run_name']/'last.pt'
    expected='c34779eb82b9ec2dd30dac49823386afeadf2c7b7add9f60dee0715a96309626'
    assert sha(checkpoint)==expected
    dataset=build_split(build_dataset,config['data']['dataset'],'train',None,True,config['data']['loader_kwargs'])
    batch=collate_fn([dataset[i] for i in indices])
    OUT.mkdir()
    begin=time.perf_counter();set_seed(42)
    model=BoneVQAModel.load(str(checkpoint),device='cpu').train()
    load_seconds=time.perf_counter()-begin
    def digest():return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    before=digest();params=dict(model.named_parameters())
    names=['qformer.query','prefix_proj.0.weight',
        next(n for n in params if n.startswith('visual.') and 'lora_B' in n),
        next(n for n in params if n.startswith('llm.') and 'lora_B' in n),'fracture_head.net.3.weight']
    results={n:[] for n in (8,10,12)};references=None
    # Warm each thread setting once. Two measured rounds reverse order to
    # reduce a simple warmup/order advantage, without repeatedly loading model.
    order=[(8,False),(10,False),(12,False)]+[(n,True) for n in (8,10,12,12,10,8)]
    for threads,measured in order:
        torch.set_num_threads(threads);assert torch.get_num_threads()==threads
        set_seed(42);model.zero_grad(set_to_none=True)
        started=time.perf_counter();out=model(batch,stage=3);out['loss'].backward()
        elapsed=time.perf_counter()-started;loss=float(out['loss'].detach())
        grads={n:params[n].grad.detach().clone() for n in names}
        assert all(torch.isfinite(g).all() for g in grads.values())
        if references is None:references=(loss,grads)
        errors={n:float((grads[n]-references[1][n]).abs().max()) for n in names}
        screen=abs(loss-references[0])<=1e-5 and all(torch.allclose(grads[n],references[1][n],atol=1e-6,rtol=1e-4) for n in names)
        if measured:results[threads].append({'seconds':elapsed,'loss':loss,
            'loss_abs_delta_vs8':abs(loss-references[0]),'selected_gradient_max_errors':errors,
            'finite_gradients':True,'numerical_screen_passed':bool(screen)})
        del out
    torch.set_num_threads(8)
    assert digest()==before and sha(checkpoint)==expected
    summary={str(n):{'median_seconds':statistics.median(r['seconds'] for r in rows),
        'numerical_screen_passed':all(r['numerical_screen_passed'] for r in rows),'runs':rows} for n,rows in results.items()}
    eligible=[n for n in summary if summary[n]['numerical_screen_passed']]
    fastest=min(eligible,key=lambda n:summary[n]['median_seconds']) if eligible else None
    report={'scope':'Current mixed4TRAINrows native forward/backward microbenchmark; not long-run training/update/resume certification',
        'script_sha256':sha(__file__),'fixture_sha256':sha(fixture),'TRAIN_indices':indices,
        'checkpoint_sha256':expected,'resident_model_load_seconds':load_seconds,
        'thread_results':summary,'fastest_screened_threads':fastest,
        'speedup_vs8':summary['8']['median_seconds']/summary[fastest]['median_seconds'] if fastest else None,
        'single_resident_model_constructions':1,'optimizer_updates':0,'checkpoint_or_trainable_weights_modified':False,
        'peak_process_RSS_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'runtime_changed_for_quality_training':False,'interop_threads':1,
        'remaining':'Use measured winner for new diagnostic work only after its exact replay; any new quality training needs scoped numerical/update/resume gates. MPS remains separately uncertified.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
