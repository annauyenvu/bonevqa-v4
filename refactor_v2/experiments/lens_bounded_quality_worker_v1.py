"""Fixed two-stage native-asset Lens pilot; strict fresh selected generation replay."""
import gc
import json
from pathlib import Path
import resource
import sys
import time
import torch
from experiments.lens_bounded_route_v1 import ROOT,PREP,QUALITY,sha,inputs,load_native,restore,trainable,make_batch,objective,optimizer_and_scheduler,gradient_audit,save_mutable,state_hash,evaluate
from experiments.gate_lens_native_trainability_v1 import tensor_hash


def main():
    arm=sys.argv[1];assert arm in ('ce_only','lens_losses');replay='--replay' in sys.argv
    out=QUALITY/arm
    if not replay:out.mkdir(exist_ok=False)
    start=time.perf_counter()
    release=json.loads((QUALITY/'execution_release.json').read_text());assert release['quality_released']
    assert all(sha(Path(p))==h for p,h in release['dependencies'].items())
    config=json.loads((PREP/'config.json').read_text());audit=json.loads((PREP/'trainability.json').read_text())
    receipt,cases=inputs();model,processor=load_native();restore(model,PREP/'shared_PREupdate.pt')
    assert state_hash(model)==audit['full_state_digest'] and state_hash(model,True)==audit['frozen_digest']
    if replay:
        completed=json.loads((out/'completed.json').read_text());assert sha(out/'LAST.pt')==completed['LAST_sha256']
        restore(model,out/'LAST.pt');assert state_hash(model)==completed['final_full_state_digest']
        fresh=evaluate(model,processor,config['model_validation'],out,'fresh_replay',completed['updates'])
        previous=json.loads((out/'LAST.json').read_text())
        assert fresh['rows']==previous['rows'] and fresh['metrics']==previous['metrics']
        (out/'replay.json').write_text(json.dumps({'exact_tokens_text_targets_metrics':True,
            'full_state_digest':state_hash(model),'seconds':time.perf_counter()-start,
            'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
        return
    initial={n:tensor_hash(p) for n,p in trainable(model).items()}
    (out/'started.json').write_text(json.dumps({'arm':arm,'initial_full_state_digest':state_hash(model),
        'fresh_PRE_sha256':sha(PREP/'shared_PREupdate.pt'),'updates':0,'seed':42},indent=2))
    evaluate(model,processor,config['model_validation'],out,'PRE',0)
    total_updates=0;all_epochs=[]
    for phase in ['align','vqa']:
        optimizer,scheduler=optimizer_and_scheduler(model,config['updates_per_stage'][phase]);phase_updates=0
        for epoch,groups in enumerate(config['schedules'][phase],1):
            epoch_start=time.perf_counter();micro_terms=[]
            for group_no,group in enumerate(groups,1):
                model.train();optimizer.zero_grad(set_to_none=True);tick=time.perf_counter()
                lr=optimizer.param_groups[0]['lr']
                for indices in group:
                    batch,contract=make_batch(processor,cases,indices,phase)
                    loss,terms,features=objective(model,batch,phase,arm=='lens_losses')
                    assert torch.isfinite(loss)
                    (loss/len(group)).backward()
                    micro_terms.append({'indices':indices,'total':float(loss.detach()),**terms})
                    del loss,features,batch;gc.collect()
                gradients=gradient_audit(model);optimizer.step();scheduler.step()
                phase_updates+=1;total_updates+=1
                (out/'training_progress.json').write_text(json.dumps({'phase':phase,'epoch':epoch,'group':group_no,
                    'phase_updates':phase_updates,'total_updates':total_updates,'lr_used':lr,'next_lr':optimizer.param_groups[0]['lr'],
                    'accumulated_microbatches':len(group),'group_seconds':time.perf_counter()-tick,'gradients':gradients},indent=2))
            stage='stage1' if phase=='align' else ('qa_epoch1' if epoch==1 else 'LAST')
            train_seconds=time.perf_counter()-epoch_start
            result=evaluate(model,processor,config['model_validation'],out,stage,total_updates)
            epoch_info={'phase':phase,'epoch':epoch,'stage':stage,'updates':total_updates,'microbatches':len(micro_terms),
                'mean_loss':sum(t['total'] for t in micro_terms)/len(micro_terms),
                'mean_CE':sum(t['CE'] for t in micro_terms)/len(micro_terms),
                'mean_contrastive':sum(t['contrastive'] for t in micro_terms)/len(micro_terms),
                'train_seconds':train_seconds,'evaluation_seconds':result['seconds'],'micro_terms':micro_terms}
            all_epochs.append(epoch_info);(out/(stage+'_training.json')).write_text(json.dumps(epoch_info,indent=2))
            save_mutable(model,out/(stage+'.pt'))
        assert phase_updates==config['updates_per_stage'][phase]
        del optimizer,scheduler;gc.collect()
    assert total_updates==sum(config['updates_per_stage'].values())
    assert state_hash(model,True)==audit['frozen_digest']
    changed=[n for n,p in trainable(model).items() if tensor_hash(p)!=initial[n]]
    assert len(changed)==len(initial)
    (out/'completed.json').write_text(json.dumps({'arm':arm,'updates':total_updates,'epochs':all_epochs,
        'changed_tensor_count':len(changed),'frozen_invariance':True,'frozen_digest':audit['frozen_digest'],
        'final_full_state_digest':state_hash(model),'LAST_sha256':sha(out/'LAST.pt'),
        'seconds':time.perf_counter()-start,'peak_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        out=QUALITY/sys.argv[1]
        if out.exists():
            with (out/('replay_failure.json' if '--replay' in sys.argv else 'failure.json')).open('x') as f:json.dump({'error':repr(exc)},f)
        raise
