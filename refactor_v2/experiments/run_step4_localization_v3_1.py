"""Execute one released four-epoch arm; exclusive lock, no implicit resume."""
import os,sys,time,json
import torch
from src.models.bonevqa import BoneVQAModel
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.train import build_split,seed_training_epoch
from src.utils import compute_metrics
from experiments.prepare_step4_localization_v3_1 import OUT,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha

def main():
    arm=sys.argv[1]; assert arm in ('B','V_real','V_placebo')
    smoke='--smoke' in sys.argv
    release=read(OUT/('smoke_release.json' if smoke else 'execution_release.json'))
    assert release['smoke_only'] if smoke else release['training_released']
    for p,h in release['dependencies'].items(): assert sha(p)==h,p
    torch.set_num_threads(8)
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    run=OUT/(('runner_smoke_' if smoke else 'run_')+arm); assert not run.exists()
    fd=os.open(OUT/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.close(fd)
    run.mkdir(); started=time.perf_counter()
    try:
        cfg=read(OUT/(arm+'.json')); model=BoneVQAModel.load(cfg['init_from'],device='cpu')
        datasets={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')}
        expected=read(OUT/'model_val.json'); boxes=read(cfg['pilot_contract']['proposal_path'])
        if smoke: expected=expected[:8]
        lookup=read(OUT/'component_lookup.json')
        def batch(role,indices):
            b=collate_fn([datasets[role][i] for i in indices])
            b['component_ids']=[lookup[i] for i in b['image_ids']]
            if arm!='B':
                b['boxes']=[boxes[i] or None for i in b['image_ids']]
                b['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
            return b
        def evaluate(epoch):
            model.eval(); rows=[]
            with torch.no_grad():
                for i in range(0,len(expected),4):
                    b=batch('model_val',range(i,min(i+4,len(expected))))
                    public={k:b[k] for k in ('images','questions','boxes','masks') if k in b and (arm!='B' or k in ('images','questions'))}
                    pred=model.predict_batch(public)
                    assert len(pred)==len(b['questions'])
                    assert b['image_ids']==[r['image_id'] for r in expected[i:i+4]]
                    for e,p in zip(expected[i:i+4],pred):
                        row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                        if 'p_yes' in p: row['p_yes']=float(p['p_yes'])
                        rows.append(row)
            assert len(rows)==len(expected)
            metrics=compute_metrics(rows,fracture_threshold=.5)
            dump(run/f'epoch_{epoch}_evaluation.json',{'metrics':metrics,'records':rows})
            return metrics
        history=[{'epoch':0,'metrics':evaluate(0)}]
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        schedules=read(OUT/'schedules.json'); updates=0
        if smoke:
            train_rows=read(OUT/'train.json')
            active=[indices for indices in schedules[0] if any(boxes[train_rows[j]['image_id']] for j in indices)]
            assert len(active)>=8
            schedules=[active[:8]]
        cap=2 if smoke else cfg['pilot_contract']['updates']
        selected={k:p for k,p in model.named_parameters() if k.startswith('fracture_head.') or k in ('prompt_mixer.gamma_visual','prompt_mixer.visual_proj.0.weight')}
        gradients={k:0. for k in selected}; initial={k:p.detach().clone() for k,p in selected.items()}
        for epoch,batches in enumerate(schedules):
            seed_training_epoch(42,epoch); model.train(); optimizer.zero_grad(set_to_none=True); losses=[]; t=time.perf_counter()
            for i,indices in enumerate(batches):
                output=model(batch('train',indices),stage=3); assert torch.isfinite(output['loss'])
                losses.append(float(output['loss'].detach())); (output['loss']/4).backward()
                for k,p in selected.items():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
                        gradients[k]=max(gradients[k],float(p.grad.abs().max()))
                if (i+1)%4==0 or i+1==len(batches):
                    assert updates<cap
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1))
                    optimizer.step(); optimizer.zero_grad(set_to_none=True); updates+=1
            seconds=time.perf_counter()-t
            history.append({'epoch':epoch+1,'updates':updates,'mean_train_loss':sum(losses)/len(losses),'train_seconds':seconds,'metrics':evaluate(epoch+1)})
            model.save(str(run/f'epoch_{epoch+1}.pt'),{'updates':updates,'primary':not smoke and epoch==3,'disposable':smoke,'release_sha256':sha(OUT/('smoke_release.json' if smoke else 'execution_release.json'))})
            torch.save({'optimizer':optimizer.state_dict(),'updates':updates},run/f'epoch_{epoch+1}_optimizer.pt')
            print(json.dumps(history[-1]),flush=True)
        assert updates==cap
        changes={k:float((p-initial[k]).detach().abs().max()) for k,p in selected.items()}
        dump(run/'completion.json',{'history':history,'updates':updates,'elapsed_seconds':time.perf_counter()-started,'primary':'epoch_1.pt' if smoke else 'epoch_4.pt','disposable':smoke,'gradients':gradients,'selected_parameter_updates':changes,'independent_audit_pending':True})
    except Exception as e:
        dump(run/'failure.json',{'error':repr(e),'preserved':True}); raise
    finally:
        (OUT/'ACTIVE.lock').unlink()
if __name__=='__main__': main()
