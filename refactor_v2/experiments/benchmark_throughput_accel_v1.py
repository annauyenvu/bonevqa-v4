"""Zero-update TRAIN forward/backward throughput: device x batch size x threads.
Usage: python -m experiments.benchmark_throughput_accel_v1 --device mps --batch 8 --threads 8 --tag solo
Writes refactor_v2/accel_v1/bench/<tag>_<device>_b<batch>_t<threads>.json. No optimizer step, weights unchanged."""
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA'); OUT=ROOT/'refactor_v2/accel_v1/bench'
def read(p): return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',required=True); ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--threads',type=int,default=8); ap.add_argument('--repeats',type=int,default=6); ap.add_argument('--tag',default='solo')
    a=ap.parse_args()
    import torch
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
    cfg=read(read(parent/'protocol.json')['configs']['token_mean']['path'])
    ckpt=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    idx=read(ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/protocol.json')['TRAIN_indices']
    idx=(list(idx)+[i for i in range(len(ds)) if i not in idx])[:a.batch]
    batch=collate_fn([ds[i] for i in idx])
    model=BoneVQAModel.load(str(ckpt),device='cpu').train().to(a.device)
    sync=(lambda: torch.mps.synchronize()) if a.device=='mps' else (lambda: None)
    times=[];losses=[]; t0=time.perf_counter()
    for r in range(a.repeats+1):
        set_seed(42); model.zero_grad(set_to_none=True); sync(); b=time.perf_counter()
        out=model(batch,stage=3); out['loss'].backward(); sync(); e=time.perf_counter()-b
        loss=float(out['loss'].detach().cpu()); assert loss==loss
        if r: times.append(e); losses.append(loss)
        del out
    med=statistics.median(times)
    res={'device':a.device,'batch':a.batch,'threads':a.threads,'tag':a.tag,'torch':torch.__version__,
         'median_seconds_per_batch':med,'samples_per_second':a.batch/med,'seconds':times,'losses':losses,
         'wall_including_load':time.perf_counter()-t0,'optimizer_updates':0,'pid':os.getpid()}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{a.tag}_{a.device}_b{a.batch}_t{a.threads}.json').write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps(res))
if __name__=='__main__': main()
