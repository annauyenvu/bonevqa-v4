"""Actual source-sized accumulated and stage-reset scheduled optimizer gate."""
import gc
import json
from pathlib import Path
import resource
import sys
import time
import torch
from experiments.lens_bounded_route_v1 import ROOT,PREP,sha,inputs,load_native,restore,trainable,make_batch,objective,optimizer_and_scheduler,gradient_audit,save_mutable,state_hash
from experiments.gate_lens_native_trainability_v1 import tensor_hash,eval_signature


def main():
    arm=sys.argv[1];assert arm in ('ce_only','lens_losses');replay='--replay' in sys.argv
    out=ROOT/'reports/lens_bounded_optimizer_gates_v1'/arm
    if not replay:out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();prep=json.loads((PREP/'preparation_seal.json').read_text());assert prep['complete']
    for p,h in prep['dependencies'].items():assert sha(Path(p))==h
    config=json.loads((PREP/'config.json').read_text());receipt,cases=inputs();model,processor=load_native()
    restore(model,PREP/'shared_PREupdate.pt')
    audit=json.loads((PREP/'trainability.json').read_text())
    assert state_hash(model)==audit['full_state_digest'] and state_hash(model,True)==audit['frozen_digest']
    params=trainable(model);initial={n:tensor_hash(p) for n,p in params.items()}
    dependencies={str(Path(__file__)):sha(Path(__file__)),str(PREP/'preparation_seal.json'):sha(PREP/'preparation_seal.json')}
    replay_indices=config['schedules']['vqa'][0][0][0]
    if replay:
        saved=json.loads((out/'update.json').read_text());assert saved['dependencies']==dependencies
        assert sha(out/'mutable.pt')==saved['mutable_sha256']
        restore(model,out/'mutable.pt');assert state_hash(model)==saved['full_state_digest']
        batch,_=make_batch(processor,cases,replay_indices,'vqa',check_pixels=True)
        signature=eval_signature(model,batch);assert signature==saved['eval_signature']
        (out/'replay.json').write_text(json.dumps({'exact':True,'full_state_digest':state_hash(model),'signature':signature,
            'seconds':time.perf_counter()-start,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
        return
    (out/'started.json').write_text(json.dumps({'arm':arm,'initial_full_state_digest':state_hash(model),'dependencies':dependencies}))
    records=[]
    for phase in ['align','vqa']:
        optimizer,scheduler=optimizer_and_scheduler(model,config['updates_per_stage'][phase])
        group=config['schedules'][phase][0][0];assert len(group)==4
        model.train();optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();micro=[]
        lr=optimizer.param_groups[0]['lr']
        for indices in group:
            batch,contract=make_batch(processor,cases,indices,phase,check_pixels=True)
            loss,terms,features=objective(model,batch,phase,arm=='lens_losses')
            assert torch.isfinite(loss)
            (loss/len(group)).backward()
            micro.append({'contract':contract,**terms,'loss':float(loss.detach())})
            del loss,features,batch;gc.collect()
            (out/'progress.json').write_text(json.dumps({'phase':phase,'microbatches':len(micro),'seconds':time.perf_counter()-tick}))
        grads=gradient_audit(model);optimizer.step();scheduler.step()
        record={'phase':phase,'microbatches':micro,'lr_used':lr,'next_lr':optimizer.param_groups[0]['lr'],
                'optimizer_reset_at_phase':True,**grads,'seconds':time.perf_counter()-tick}
        records.append(record);(out/(phase+'.json')).write_text(json.dumps(record,indent=2))
        del optimizer,scheduler;gc.collect()
    assert state_hash(model,True)==audit['frozen_digest']
    changed=[n for n,p in params.items() if tensor_hash(p)!=initial[n]]
    assert len(changed)==len(params),'Some LoRA tensors not changed after both updates'
    save_mutable(model,out/'mutable.pt');batch,_=make_batch(processor,cases,replay_indices,'vqa',check_pixels=True)
    signature=eval_signature(model,batch)
    (out/'update.json').write_text(json.dumps({'arm':arm,'updates':2,'microbatches':8,'phases':records,
        'initial_full_state_digest':audit['full_state_digest'],'frozen_digest':audit['frozen_digest'],'frozen_invariance':True,
        'changed_names':changed,'full_state_digest':state_hash(model),'eval_signature':signature,'mutable_sha256':sha(out/'mutable.pt'),
        'dependencies':dependencies,'seconds':time.perf_counter()-start,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'quality_released':False},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        out=ROOT/'reports/lens_bounded_optimizer_gates_v1'/sys.argv[1]
        if out.exists():
            with (out/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f:json.dump({'error':repr(exc)},f)
        raise
