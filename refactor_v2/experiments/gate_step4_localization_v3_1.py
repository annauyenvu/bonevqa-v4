"""Disposable gradient/update/reload gates and measured bounded timing."""
import gc,json,time,copy,hashlib
from pathlib import Path
import torch
from src.models.bonevqa import BoneVQAModel
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.train import build_split,seed_training_epoch
from src.utils import set_seed
from experiments.prepare_step4_localization_v3_1 import ROOT,R,OUT,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha

def hashes(state): return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in state.items()}
def plain(value):
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [plain(v) for v in value]
    if hasattr(value,'tolist'): return value.tolist()
    return value
def main():
    torch.set_num_threads(8); assert read(OUT/'preparation.json')['passed']
    assert not (OUT/'gate_report.json').exists()
    initial_path=R/'fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_initialization.pt'
    assert sha(initial_path)=='bfc799a5fb6db140ab833db460308c98cba76d871b9d415856175ed769abc09b'
    initial=torch.load(initial_path,map_location='cpu',weights_only=False)
    vocab=read(OUT/'vocab.json'); configs={a:read(OUT/(a+'.json')) for a in ('B','V_real','V_placebo')}
    reference=None; reports={}; began=time.perf_counter()
    for arm,cfg in configs.items():
        set_seed(42); model=BoneVQAModel(cfg['model'],vocab,device='cpu')
        target=model.trainable_state_dict()
        if reference is None:
            incoming={k:v for k,v in initial['state'].items() if not k.startswith('closed_head.')}
            assert all(k in target and v.shape==target[k].shape for k,v in incoming.items())
            model.load_state_dict(incoming,strict=False)
            reference={k:v.clone() for k,v in model.trainable_state_dict().items()}
        else:
            assert all(k in target and v.shape==target[k].shape for k,v in reference.items())
            model.load_state_dict(reference,strict=False)
        assert all(torch.equal(model.trainable_state_dict()[k],v) for k,v in reference.items())
        fresh=OUT/(arm+'_PREtask.pt'); model.save(str(fresh),{'zero_QA_updates':True,'parent_sha256':sha(initial_path),'new_vocab_from_selected_TRAIN':True})
        before_hash=sha(fresh); before=hashes(model.trainable_state_dict())
        ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
        proposals=read(OUT/('placebo.json' if arm=='V_placebo' else 'boxes.json'))
        def batch(indices):
            b=collate_fn([ds[i] for i in indices])
            if arm!='B':
                b['boxes']=[proposals[i] or None for i in b['image_ids']]
                b['masks']=[FracAtlasVQADataset._boxes_to_masks(proposals[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
            return b
        rows=read(OUT/'train.json'); selected=[]
        for nonempty in (True,False):
            iid=next(r['image_id'] for r in rows if bool(proposals[r['image_id']])==nonempty)
            for typ in ('CLOSED','OPEN'):
                selected.append(next(i for i,r in enumerate(rows) if r['image_id']==iid and r['answer_type']==typ))
        b=batch(selected)
        params=dict(model.named_parameters()); names=['fracture_head.'+k.split('fracture_head.',1)[1] for k in params if k.startswith('fracture_head.')]
        if arm!='B': names+=['prompt_mixer.gamma_visual','prompt_mixer.visual_proj.0.weight']
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        original={k:params[k].detach().clone() for k in names}; gradients=[]; times=[]
        for step in range(2):
            seed_training_epoch(42,step); model.train(); optimizer.zero_grad(set_to_none=True); t=time.perf_counter()
            output=model(b,stage=3)
            isolated=torch.autograd.grad(output['loss_answer'],[params[k] for k in names],retain_graph=True,allow_unused=True)
            gradients.append({k:float(g.abs().max()) if g is not None else 0 for k,g in zip(names,isolated)})
            assert torch.isfinite(output['loss']); output['loss'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); times.append(time.perf_counter()-t)
        updates={k:float((params[k]-original[k]).detach().abs().max()) for k in names}
        assert all(v>0 for v in gradients[-1].values()) and all(v>0 for v in updates.values())
        # Public input construction deliberately excludes labels, regions and types.
        public={'images':b['images'],'questions':b['questions']}
        if arm!='B': public.update(boxes=b['boxes'],masks=b['masks'])
        model.eval(); t=time.perf_counter()
        with torch.no_grad(): prediction=model.predict_batch(public)
        inference=time.perf_counter()-t
        smoke=OUT/(arm+'_DISPOSABLE.pt'); model.save(str(smoke),{'quality_initialization_forbidden':True})
        expected=hashes(model.trainable_state_dict())
        del optimizer,output,isolated,params,model; gc.collect()
        reloaded=BoneVQAModel.load(str(smoke),device='cpu').eval()
        assert hashes(reloaded.trainable_state_dict())==expected
        with torch.no_grad(): assert plain(reloaded.predict_batch(public))==plain(prediction)
        assert sha(fresh)==before_hash
        reports[arm]={'passed':True,'initialization_sha256':before_hash,'initial_tensor_hashes':before,'gradients':gradients,'updates':updates,'reload_exact':True,'disposable_updates':2,'gate_batch_seconds':times,'gate_inference_4QA_seconds':inference,'quality_initialization_unchanged':True}
        dump(OUT/(arm+'_gate.json'),reports[arm]); print(json.dumps({'arm':arm,'passed':True,'batch_seconds':times,'inference_seconds':inference}),flush=True)
        del reloaded; gc.collect()
    dump(OUT/'gate_report.json',{'passed':True,'arms':reports,'elapsed_seconds':time.perf_counter()-began,'limitations':['Two mixed diagnostic batches, not representative full-epoch throughput','Same-process model reload; fresh-process verification still required','No claim of semantic efficacy from nonzero gradients'],'training_released':False})
if __name__=='__main__': main()
