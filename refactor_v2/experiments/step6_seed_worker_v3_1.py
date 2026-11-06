"""Seed-specific initialization and bounded training worker; no reserved-role access."""
import gc,json,sys,time
import torch
from src.models.bonevqa import BoneVQAModel
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.train import build_split,seed_training_epoch
from src.utils import set_seed,compute_metrics
from experiments.prepare_step4_localization_v3_1 import OUT as PARENT,ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step6_confirmation_v3_1'

def initialize(seed,arms):
    torch.set_num_threads(8)
    original=torch.load(PARENT/'B_PREtask.pt',map_location='cpu',weights_only=False)
    base=read(PARENT/'B.json');base['model']=dict(original['config']);base['seed']=seed
    base['model']['module_init_seed']=seed+1729
    set_seed(seed);model=BoneVQAModel(base['model'],original['answer_vocab'],device='cpu')
    upstream={k:v for k,v in original['state'].items() if not k.startswith(('closed_head.','type_head.','fracture_head.'))}
    model.load_state_dict(upstream,strict=False)
    shared={k:v.clone() for k,v in model.trainable_state_dict().items()}
    assert all(torch.equal(shared[k],v) for k,v in upstream.items())
    regenerated=[k for k in shared if k.startswith(('closed_head.','type_head.','fracture_head.')) and not torch.equal(shared[k],original['state'][k])]
    assert regenerated,'A new seed must not silently clone all seed42 task heads'
    del model;gc.collect()
    for arm in arms:
        cfg=read(PARENT/'B.json') if arm=='B' else read(R/'step56_screening_v3_1'/(arm+'.json'))
        cfg['model']['module_init_seed']=seed+1729;cfg['seed']=seed
        set_seed(seed);m=BoneVQAModel(cfg['model'],original['answer_vocab'],device='cpu');m.load_state_dict(shared,strict=False)
        assert all(torch.equal(m.trainable_state_dict()[k],v) for k,v in shared.items())
        folder=OUT/f'seed{seed}_{arm}';assert not folder.exists();folder.mkdir()
        m.save(str(folder/'PREtask.pt'),{'seed':seed,'zero_QA_updates':True,'upstream_parent':sha(PARENT/'B_PREtask.pt'),'task_heads_regenerated':True})
        cfg['init_from']=str(folder/'PREtask.pt');dump(folder/'config.json',cfg)
        dump(folder/'initialization.json',{'seed':seed,'shared_tensors':len(shared),'non_task_upstream_tensors':len(upstream),'regenerated_head_tensors_differing_from_seed42':regenerated,'initial_sha256':sha(folder/'PREtask.pt'),'source_sha256':sha(__file__)})
        del m;gc.collect()

def run(seed,arm,smoke):
    torch.set_num_threads(8);folder=OUT/f'seed{seed}_{arm}';cfg=read(folder/'config.json')
    release=read(OUT/('gate_release.json' if smoke else 'training_release.json'))
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    if not smoke:assert release['training_released']
    target=folder/('smoke' if smoke else 'quality');assert not target.exists();target.mkdir()
    model=BoneVQAModel.load(cfg['init_from'],device='cpu')
    ds={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')}
    lookup=read(PARENT/'component_lookup.json');boxes=read(PARENT/('placebo.json' if arm in ('L_placebo','V_placebo') else 'boxes.json'))
    schedule=read(OUT/f'seed{seed}_schedule.json');expected=read(PARENT/'model_val.json')
    if smoke:expected=expected[:8]
    def batch(role,indices,public=False):
        b=collate_fn([ds[role][i] for i in indices]);b['component_ids']=[lookup[i] for i in b['image_ids']]
        if arm.startswith('V') or (arm.startswith('L') and not public):
            b['boxes']=[boxes[i] or None for i in b['image_ids']]
            b['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
        if public:return {k:b[k] for k in ('images','questions','boxes','masks') if k in b and (arm.startswith('V') or k in ('images','questions'))}
        return b
    def evaluate(epoch):
        model.eval();rows=[]
        with torch.no_grad():
            for i in range(0,len(expected),4):
                predictions=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
                assert len(predictions)==len(expected[i:i+4])
                for e,p in zip(expected[i:i+4],predictions):
                    r={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                    if 'p_yes' in p:r['p_yes']=float(p['p_yes'])
                    rows.append(r)
        dump(target/f'epoch{epoch}_evaluation.json',{'records':rows,'metrics':compute_metrics(rows,fracture_threshold=.5)})
        return compute_metrics(rows,fracture_threshold=.5)
    stages=([1,1,3,3] if arm.startswith('L') else [3]*4)
    if smoke:stages=[1,3] if arm.startswith('L') else [3]
    named={k:p for k,p in model.named_parameters() if p.requires_grad};selected={k:p for k,p in named.items() if k.startswith(('fracture_head.','latent.','lens_fusion.')) or k=='prompt_mixer.gamma_visual'}
    before={k:p.detach().clone() for k,p in selected.items()};grads={k:0. for k in selected};updates=0;laststage=None;history=[];start=time.perf_counter()
    for epoch,stage in enumerate(stages):
        if stage!=laststage:optimizer=torch.optim.AdamW(named.values(),lr=1e-4,weight_decay=.01);laststage=stage
        seed_training_epoch(seed,epoch);model.train();optimizer.zero_grad(set_to_none=True);batches=schedule[epoch];losses=[]
        if smoke:
            records=read(PARENT/'train.json');batches=[b for b in batches if any(boxes[records[i]['image_id']] for i in b)][:8]
        for i,indices in enumerate(batches):
            o=model(batch('train',indices),stage=stage);assert torch.isfinite(o['loss']);losses.append(float(o['loss'].detach()));(o['loss']/4).backward()
            for k,p in selected.items():
                if p.grad is not None:assert torch.isfinite(p.grad).all();grads[k]=max(grads[k],float(p.grad.abs().max()))
            if (i+1)%4==0 or i+1==len(batches):
                assert updates<(4 if smoke and arm.startswith('L') else 2 if smoke else 292)
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(named.values(),1));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        history.append({'epoch':epoch+1,'stage':stage,'updates':updates,'loss':sum(losses)/len(losses),'metrics':evaluate(epoch+1) if stage==3 else None})
        model.save(str(target/f'epoch{epoch+1}.pt'),{'seed':seed,'updates':updates,'disposable':smoke})
        torch.save({'optimizer':optimizer.state_dict(),'updates':updates},target/f'epoch{epoch+1}_optimizer.pt')
        print(json.dumps(history[-1]),flush=True)
    changes={k:float((p-before[k]).detach().abs().max()) for k,p in selected.items()}
    assert any(grads[k]>0 and changes[k]>0 for k in selected if k.startswith('fracture_head.'))
    if not smoke:assert updates==292
    dump(target/'completion.json',{'history':history,'updates':updates,'gradients':grads,'parameter_updates':changes,'disposable':smoke,'seconds':time.perf_counter()-start,'reserved_inference':False})
if __name__=='__main__':
    if sys.argv[1]=='initialize':initialize(int(sys.argv[2]),sys.argv[3:])
    else:run(int(sys.argv[2]),sys.argv[3],sys.argv[1]=='smoke')
