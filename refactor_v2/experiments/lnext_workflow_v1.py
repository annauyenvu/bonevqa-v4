"""Sequential L study, waiting for the audited V study before model execution."""
import copy,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from src.data import build_dataset,collate_fn
from src.models.bonevqa import BoneVQAModel
from src.train import build_split,seed_training_epoch
from src.utils import compute_metrics,set_seed
from experiments.lnext_spatial_v1 import SpatialSupervisedModel

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports'
P=R/'step4_localization_v3_1';V=R/'vnext_local_tokens_v1';T=R/'lnext_spatial_v1_targets2';O=R/'lnext_spatial_runs_v1'
ARMS=('L_real','L_placebo','L_uniform','L_off')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2)
def check(deps):
    for p,h in deps.items():assert sha(p)==h,p
def hashes(state):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in state.items()}
def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x.tolist() if hasattr(x,'tolist') else x

def prepare():
    assert read(V/'analysis_closure.json')['V_analysis_complete']
    check(read(V/'analysis_closure.json')['artifacts'])
    for deps in read(T/'target_seal.json').values():
        if isinstance(deps,dict):check(deps)
    audit=read(T/'target_audit.json');assert audit['passed'] and audit['modelval_component_disjoint']
    parent=torch.load(P/'B_PREtask.pt',map_location='cpu',weights_only=False);shared=hashes(parent['state']);head_reference=None
    for arm in ARMS:
        cfg=read(P/'B.json');cfg['model']=copy.deepcopy(parent['config'])
        cfg['model'].update(lnext_spatial_weight=0. if arm=='L_off' else .1,lnext_positive_weight=audit['spatial_positive_weight'])
        cfg['init_from']=str(O/(arm+'_PREtask.pt'));cfg['lnext']={'arm':arm,'targets':str(T/('targets_'+arm[2:]+'.json')),'epochs':4,'updates':292,'seed':42}
        set_seed(42);model=SpatialSupervisedModel(cfg['model'],parent['answer_vocab'],device='cpu');model.load_state_dict(parent['state'],strict=False)
        state=hashes(model.trainable_state_dict());assert all(state[k]==h for k,h in shared.items())
        head={k:h for k,h in state.items() if k.startswith('spatial_head.')}
        if head_reference is None:head_reference=head
        assert head==head_reference and len(head)==4
        model.save(cfg['init_from'],{'zero_task_updates':True,'parent_sha256':sha(P/'B_PREtask.pt'),'model_class':'SpatialSupervisedModel'})
        write(O/(arm+'.json'),cfg);write(O/(arm+'_initialization.json'),{'shared_tensors_exact':len(shared),'tensor_hashes':state,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        del model;gc.collect()
    files=list(O.glob('*.json'))+list(O.glob('*.pt'))+list((ROOT/'refactor_v2/src').rglob('*.py'))
    files += [Path(__file__).resolve(),ROOT/'refactor_v2/experiments/lnext_spatial_v1.py',ROOT/'refactor_v2/experiments/prepare_lnext_v1.py',ROOT/'refactor_v2/experiments/report_lnext_v1.py',R/'LNEXT_SPATIAL_PROTOCOL_20261025.md']
    files += list(T.glob('*.json'))+[P/n for n in ('B_PREtask.pt','train.json','model_val.json','data_seal.json','schedules.json','component_lookup.json')]+[V/'gate_indices.json',V/'analysis_closure.json']
    write(O/'gate_release.json',{'dependencies':{str(p):sha(p) for p in files},'quality_released':False})

def inputs(cfg):
    ds={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')}
    targets=read(cfg['lnext']['targets']);lookup=read(P/'component_lookup.json')
    def batch(role,indices,public=False):
        b=collate_fn([ds[role][i] for i in indices])
        if public:return {'images':b['images'],'questions':b['questions']}
        b['component_ids']=[lookup[i] for i in b['image_ids']]
        # Boxes/masks/annotation-derived regions never enter the image path.
        b['boxes']=[None]*len(indices);b['masks']=[None]*len(indices);b['regions']=[None]*len(indices)
        if role=='train':
            b['spatial_targets']=[targets[i]['target'] for i in b['image_ids']]
            b['spatial_weights']=[targets[i]['weight'] for i in b['image_ids']]
        return b
    return batch

def mechanics(model,batch,selected,cfg):
    model.eval();public=batch('train',selected,True)
    # The API echoes caller masks as diagnostics; compare answers/probabilities,
    # excluding that echoed input, when deliberately corrupting metadata.
    def endpoint(records):return [{k:plain(v) for k,v in r.items() if k!='masks'} for r in records]
    with torch.no_grad():
        original=endpoint(model.predict_batch(public))
        dirty=dict(public,boxes=[[[0,0,1,1]]]*4,masks=[np.ones((1,1,1),dtype=np.uint8)]*4,regions=['wrong']*4,answers=['wrong']*4,answer_types=['OPEN']*4,spatial_targets=[[1]*196]*4)
        assert endpoint(model.predict_batch(dirty))==original
        before={k:p.detach().clone() for k,p in model.spatial_head.named_parameters()}
        for p in model.spatial_head.parameters():p.fill_(42)
        assert endpoint(model.predict_batch(public))==original
        model.spatial_head.load_state_dict(before)
        open_batch=batch('train',selected[2:]);assert all(q!='Is there a fracture in this X-ray?' for q in open_batch['questions'])
        assert float(model(open_batch,stage=3)['spatial_loss'])==0
    return {'public_metadata_invariant':True,'spatial_head_intervention_invariant':True,'OPEN_spatial_loss_zero':True,'original_image_only':True}

def worker(arm,smoke):
    release=read(O/('gate_release.json' if smoke else 'training_release.json'));check(release['dependencies'])
    if not smoke:assert release['quality_released']
    cfg=read(O/(arm+'.json'));folder=O/(('gate_' if smoke else 'run_')+arm);assert not folder.exists();folder.mkdir()
    set_seed(42);model=SpatialSupervisedModel.load(cfg['init_from'],device='cpu');batch=inputs(cfg)
    selected=read(V/'gate_indices.json');assert len(selected)==4
    initial_check={}
    if smoke:
        # The auxiliary head cannot change zero-task B logits or fused answer features.
        public=batch('train',selected,True);model.eval()
        with torch.no_grad():
            masks=model._prepare_masks(public['images'],None,None,None)
            augmented=model._encode_core(public['images'],public['questions'],[None]*4,masks,'contour',False)
            base=BoneVQAModel._encode_core(model,public['images'],public['questions'],[None]*4,masks,'contour',False)
            for key in ('fracture_logit','logits_closed','type_logit'):assert torch.equal(augmented[key],base[key])
            assert torch.equal(augmented['fus']['x_f'],base['fus']['x_f'])
        initial_check={'initial_logits_and_fused_features_exact_B':True}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    names={k:p for k,p in model.named_parameters() if k.startswith(('spatial_head.','fracture_head.')) or ('visual.' in k and 'lora' in k.lower() and p.requires_grad)}
    initial={k:p.detach().clone() for k,p in names.items()};grads={k:0. for k in names};routes={};history=[];updates=0;started=time.perf_counter()
    schedules=[[selected]*8] if smoke else read(P/'schedules.json');expected=read(P/'model_val.json');expected=expected[:8] if smoke else expected
    for epoch,schedule in enumerate(schedules):
        seed_training_epoch(42,epoch);model.train();optimizer.zero_grad(set_to_none=True);losses=[];spatial=[];start=time.perf_counter()
        for i,indices in enumerate(schedule):
            output=model(batch('train',indices),stage=3);assert torch.isfinite(output['loss'])
            if smoke and i==0:
                heads=list(model.spatial_head.parameters())
                answer_grad=torch.autograd.grad(output['loss_answer'],heads,retain_graph=True,allow_unused=True)
                assert all(g is None for g in answer_grad)
                routes['answer_loss_to_spatial_head_absent']=True
                if arm!='L_off':
                    visual=[p for k,p in names.items() if k.startswith('visual.')]
                    values=torch.autograd.grad(output['spatial_loss'],visual,retain_graph=True,allow_unused=True)
                    maximum=max(float(g.abs().max()) for g in values if g is not None)
                    assert maximum>0 and np.isfinite(maximum);routes['spatial_loss_to_visual_lora_max']=maximum
            losses.append(float(output['loss'].detach()));spatial.append(float(output['spatial_loss'].detach()))
            (output['loss']/4).backward()
            for k,p in names.items():
                if p.grad is not None:assert torch.isfinite(p.grad).all();grads[k]=max(grads[k],float(p.grad.abs().max()))
            if (i+1)%4==0 or i+1==len(schedule):
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        train_seconds=time.perf_counter()-start;model.eval();start=time.perf_counter();rows=[]
        with torch.no_grad():
            for i in range(0,len(expected),4):
                pred=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
                assert model._spatial_logits.shape==(len(pred),196)
                for j,(e,p) in enumerate(zip(expected[i:i+4],pred)):
                    row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                    if 'p_yes' in p:row['p_yes']=float(p['p_yes'])
                    if e['question']=='Is there a fracture in this X-ray?':row['spatial_probability']=torch.sigmoid(model._spatial_logits[j]).tolist()
                    rows.append(row)
        metrics=compute_metrics(rows,fracture_threshold=.5);assert len(rows)==len(expected)
        write(folder/f'epoch{epoch+1}_evaluation.json',{'records':rows,'metrics':metrics})
        model.save(str(folder/f'epoch{epoch+1}.pt'),{'updates':updates,'disposable':smoke,'model_class':'SpatialSupervisedModel'})
        torch.save({'optimizer':optimizer.state_dict(),'updates':updates},folder/f'epoch{epoch+1}_optimizer.pt')
        history.append({'epoch':epoch+1,'updates':updates,'loss':sum(losses)/len(losses),'spatial_loss':sum(spatial)/len(spatial),'train_seconds':train_seconds,'eval_seconds':time.perf_counter()-start,'metrics':metrics});print(json.dumps(history[-1]),flush=True)
    assert updates==(2 if smoke else 292)
    changes={k:float((p-initial[k]).detach().abs().max()) for k,p in names.items()}
    assert any(v>0 for k,v in grads.items() if k.startswith('visual.'))
    assert any(v>0 for k,v in changes.items() if k.startswith('fracture_head.'))
    if arm!='L_off':assert grads['spatial_head.1.weight']>0 and changes['spatial_head.1.weight']>0
    tested=mechanics(model,batch,selected,cfg) if smoke else {}
    write(folder/'completion.json',{'history':history,'updates':updates,'elapsed_seconds':time.perf_counter()-started,'gradients':grads,'parameter_changes':changes,'routes':routes,'mechanics':tested,'initial_matching':initial_check,'fresh_replay_pending':True})

def replay(arm,smoke):
    cfg=read(O/(arm+'.json'));folder=O/(('gate_' if smoke else 'run_')+arm);c=read(folder/'completion.json');epoch=len(c['history']);path=folder/f'epoch{epoch}.pt';digest=sha(path)
    set_seed(90210);model=SpatialSupervisedModel.load(str(path),device='cpu').eval();batch=inputs(cfg);expected=read(folder/f'epoch{epoch}_evaluation.json')['records'];rows=[]
    with torch.no_grad():
        for i in range(0,len(expected),4):
            predictions=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
            assert model._spatial_logits.shape==(len(predictions),196)
            for j,(e,p) in enumerate(zip(expected[i:i+4],predictions)):
                row={k:e[k] for k in ('image_id','question','gt','answer_type')};row['pred']=p['answer']
                if 'p_yes' in p:row['p_yes']=float(p['p_yes'])
                if e['question']=='Is there a fracture in this X-ray?':row['spatial_probability']=torch.sigmoid(model._spatial_logits[j]).tolist()
                rows.append(row)
    assert rows==expected and sha(path)==digest
    write(folder/'fresh_replay.json',{'passed':True,'records_exact':len(rows),'checkpoint_sha256':digest})

def child(mode,arm):
    with (O/f'{mode}_{arm}.log').open('x') as log:subprocess.run([sys.executable,'-u',__file__,mode,arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def operate():
    assert not O.exists();O.mkdir();write(O/'coordinator.json',{'pid':os.getpid(),'waiting_for':str(V/'analysis_closure.json'),'started':time.time(),'arms':ARMS})
    while not (V/'analysis_closure.json').exists():
        errors=list(V.glob('failure_*.json'))+list(V.glob('analysis_failure.json'))
        if errors:raise RuntimeError('V failure before L release: '+str(errors))
        time.sleep(30)
    prepare()
    for arm in ARMS:child('smoke',arm);child('replay_smoke',arm)
    release=read(O/'gate_release.json');check(release['dependencies']);release['quality_released']=True
    release['gate_evidence']={a:{'completion':sha(O/f'gate_{a}/completion.json'),'replay':sha(O/f'gate_{a}/fresh_replay.json')} for a in ARMS};write(O/'training_release.json',release)
    for arm in ARMS:child('quality',arm);child('replay_quality',arm)
    write(O/'execution_completed.json',{'arms':ARMS,'full_replays_passed':True,'analysis_pending':True})

if __name__=='__main__':
    assert os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1';torch.set_num_threads(8)
    try:
        if len(sys.argv)==1:operate()
        elif sys.argv[1].startswith('replay'):replay(sys.argv[2],sys.argv[1]=='replay_smoke')
        else:worker(sys.argv[2],sys.argv[1]=='smoke')
    except Exception as e:
        if O.exists():
            path=O/('failure_'+str(os.getpid())+'.json')
            if not path.exists():write(path,{'error':repr(e),'argv':sys.argv,'preserve_all_outputs':True})
        raise
