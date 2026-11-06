"""Fresh common PRE, fixed schedules, actual input contracts and binding seal."""
import json
from pathlib import Path
import random
import shutil
import time
import torch
from experiments.lens_bounded_route_v1 import ROOT,SOURCE,INPUTS,PREP,sha,inputs,load_native,make_batch,trainable,save_mutable,state_hash


def main():
    PREP.mkdir(exist_ok=False);start=time.perf_counter()
    receipt,cases=inputs();n=len(cases);assert n>=16 and n%2==0
    assert shutil.disk_usage(ROOT).free>8*1024**3
    schedules={}
    rng=random.Random(42)
    for phase,epochs in [('align',1),('vqa',2)]:
        schedules[phase]=[]
        for epoch in range(epochs):
            order=list(range(n));rng.shuffle(order)
            micro=[order[i:i+2] for i in range(0,n,2)]
            groups=[micro[i:i+4] for i in range(0,len(micro),4)]
            schedules[phase].append(groups)
    config={'arms':['ce_only','lens_losses'],'seed':42,'training_ids':[r['image']['image_id'] for r in cases],
        'train_count':n,'model_validation':receipt['model_validation'],'schedules':schedules,
        'updates_per_stage':{p:sum(len(epoch) for epoch in epochs) for p,epochs in schedules.items()},
        'actual_microbatches_per_stage':{p:sum(sum(len(g) for g in epoch) for epoch in epochs) for p,epochs in schedules.items()},
        'partial_accumulation':'divide micro-loss by actual microbatch count in group, including short final group',
        'stage_optimizer_reset':True,'lr':2e-4,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'clip':1.,
        'lr_schedule':'linear from base LR to zero separately per stage, no warmup','batch':2,'accumulation':4,
        'processor_longest_edge':1024,'tile_size':512,'max_new_tokens':32,'quality_released':False}
    (PREP/'config.json').write_text(json.dumps(config,indent=2))
    model,processor=load_native()
    # Walk every actual training view through source collation and independent tile provenance before release.
    audits=[]
    for phase in ['align','vqa']:
        for i in range(0,n,2):
            batch,info=make_batch(processor,cases,[i,i+1],phase,check_pixels=True)
            audits.append({'phase':phase,**info});del batch
            (PREP/'progress.json').write_text(json.dumps({'phase':'input_contracts','checked_pairs':len(audits)}))
    (PREP/'input_contract_audit.json').write_text(json.dumps(audits,indent=2))
    params=trainable(model)
    save_mutable(model,PREP/'shared_PREupdate.pt')
    (PREP/'trainability.json').write_text(json.dumps({'trainable_names':list(params),'parameters':sum(p.numel() for p in params.values()),
        'initial_tensor_digests':{n:__import__('experiments.gate_lens_native_trainability_v1',fromlist=['tensor_hash']).tensor_hash(p) for n,p in params.items()},
        'full_state_digest':state_hash(model),'frozen_digest':state_hash(model,True),'updates':0},indent=2))
    dependencies=dict(receipt['dependencies'])
    for case in cases:
        dependencies[case['image']['path']]=case['image']['file_sha256']
        dependencies.update(case['artifacts'])
    paths=[Path(__file__),ROOT/'experiments/lens_bounded_route_v1.py',ROOT/'experiments/lens_native_common_v1.py',
        ROOT/'experiments/gate_lens_native_trainability_v1.py',ROOT/'reports/lens_runtime_v2/receipt.json',INPUTS/'receipt.json',
        PREP/'config.json',PREP/'input_contract_audit.json',PREP/'trainability.json',PREP/'shared_PREupdate.pt']
    for rel in ['training.py','modeling.py','inference.py']:paths.append(SOURCE/'src/localization_lens'/rel)
    asset=ROOT/'references/native_assets_lens_v1/SmolVLM-256M-Instruct'
    package=json.loads((asset/'receipt.json').read_text())
    paths+=[asset/'receipt.json']+[asset/f['name'] for f in package['files']]
    for path in paths:dependencies[str(path)]=sha(path)
    assert all(sha(Path(p))==h for p,h in dependencies.items())
    (PREP/'preparation_seal.json').write_text(json.dumps({'complete':True,'updates':0,'dependencies':dependencies,
        'seconds':time.perf_counter()-start,'quality_released':False},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if PREP.exists():
            with (PREP/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
