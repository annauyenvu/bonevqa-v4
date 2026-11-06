"""Sequential, bounded new V study. No implicit overwrite or retry."""
import copy,gc,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import torch
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.models.bonevqa import BoneVQAModel
from src.train import build_split,seed_training_epoch
from src.utils import compute_metrics,set_seed
from experiments.vnext_model_v1 import VNextModel

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';P=R/'step4_localization_v3_1';O=R/'vnext_local_tokens_v1'
ARMS=('B','real_aux','placebo_aux','grid_aux','real_noaux')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2)
def plain(v):
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v.tolist() if hasattr(v,'tolist') else v
def hashes(state):return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in state.items()}
def check(release):
    for p,h in release['dependencies'].items():assert sha(p)==h,p

def prepare():
    assert not O.exists();O.mkdir()
    for path in (P/'step4_pilot_closure.json',R/'step6_confirmation_v3_1/steps56_closure.json'):
        for p,h in read(path)['artifacts'].items():assert sha(p)==h,p
    provenance=read(P/'proposal_provenance.json');real=read(P/'boxes.json');placebo=read(P/'placebo.json')
    assert sha(P/'boxes.json')==provenance['real_sha256'] and sha(P/'placebo.json')==provenance['placebo_sha256']
    grid={}
    for image,bs in real.items():
        n=len(bs);w=provenance['images'][image]['width'];h=provenance['images'][image]['height']
        grid[image]=[[round(j*w/n),0,round((j+1)*w/n),h] for j in range(n)] if n else []
        assert len(bs)==len(placebo[image])==len(grid[image])
    dump(O/'grid.json',grid)
    parent=torch.load(P/'B_PREtask.pt',map_location='cpu',weights_only=False);reference=hashes(parent['state']);local_reference=None
    for arm in ARMS:
        cfg=read(P/'B.json');cfg['model']=copy.deepcopy(parent['config'])
        cfg['model'].update(vnext_enabled=arm!='B',vnext_aux_weight=.1 if arm.endswith('_aux') else 0.,vnext_force_zero=False)
        cfg['init_from']=str(O/(arm+'_PREtask.pt'))
        cfg['vnext']={'arm':arm,'seed':42,'epochs':4,'updates':292,'aux_weight':cfg['model']['vnext_aux_weight'],'proposals':str(O/'grid.json' if arm=='grid_aux' else P/('placebo.json' if arm=='placebo_aux' else 'boxes.json'))}
        set_seed(42);model=VNextModel(cfg['model'],parent['answer_vocab'],device='cpu');model.load_state_dict(parent['state'],strict=False)
        current=hashes(model.trainable_state_dict());assert all(current[k]==h for k,h in reference.items())
        local={k:h for k,h in current.items() if k.startswith('local_tokens.')}
        if local:
            if local_reference is None:local_reference=local
            assert local==local_reference
        model.save(cfg['init_from'],{'zero_task_updates':True,'parent_sha256':sha(P/'B_PREtask.pt'),'model_class':'VNextModel'})
        dump(O/(arm+'.json'),cfg);dump(O/(arm+'_initialization.json'),{'shared_tensors_exact':len(reference),'local_tensors':len(local),'parameter_count':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'tensor_hashes':current})
        del model;gc.collect()
    rows=read(P/'train.json');lookup=read(P/'component_lookup.json');selected=[];used=set()
    for answer in ('yes','no'):
        for rowindex,row in enumerate(rows):
            if row['question']=='Is there a fracture in this X-ray?' and row['answer']==answer and real[row['image_id']] and lookup[row['image_id']] not in used:
                selected.append(rowindex);used.add(lookup[row['image_id']]);break
    for i,row in enumerate(rows):
        if len(selected)==4:break
        if row['answer_type']=='OPEN' and real[row['image_id']] and lookup[row['image_id']] not in used:selected.append(i);used.add(lookup[row['image_id']])
    assert len(selected)==4;dump(O/'gate_indices.json',selected)
    deps=list(O.glob('*.json'))+list(O.glob('*.pt'))+list((ROOT/'refactor_v2/src').rglob('*.py'))
    deps += [Path(__file__).resolve(),ROOT/'refactor_v2/experiments/vnext_model_v1.py',R/'ARCHITECTURE_STUDY_VNEXT_PROTOCOL_20261025.md']
    deps += [P/n for n in ('train.json','model_val.json','vocab.json','data_seal.json','schedules.json','component_lookup.json','boxes.json','placebo.json','proposal_provenance.json','B_PREtask.pt')]
    dump(O/'gate_release.json',{'dependencies':{str(p):sha(p) for p in deps},'quality_released':False})

def inputs(cfg,arm):
    ds={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')}
    boxes=read(cfg['vnext']['proposals']);lookup=read(P/'component_lookup.json')
    def batch(role,indices,public=False):
        b=collate_fn([ds[role][i] for i in indices]);b['component_ids']=[lookup[i] for i in b['image_ids']]
        if arm!='B':
            b['boxes']=[boxes[i] or None for i in b['image_ids']]
            b['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
        if public:return {k:b[k] for k in ('images','questions','boxes','masks') if k in b and (arm!='B' or k in ('images','questions'))}
        return b
    return batch

def core_fixture(model,b):
    masks=model._prepare_masks(b['images'],b.get('masks'),b.get('boxes'),None)
    return model._encode_core(b['images'],b['questions'],[None]*len(b['images']),masks,'contour',False)

def worker(arm,smoke):
    release=read(O/('gate_release.json' if smoke else 'training_release.json'));check(release)
    if not smoke:assert release['quality_released']
    cfg=read(O/(arm+'.json'));folder=O/(('gate_' if smoke else 'run_')+arm);assert not folder.exists();folder.mkdir()
    set_seed(42);model=VNextModel.load(cfg['init_from'],device='cpu');batch=inputs(cfg,arm)
    selected=read(O/'gate_indices.json');schedules=[selected]*8 if smoke else None
    schedules=[schedules] if smoke else read(P/'schedules.json')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    names={k:p for k,p in model.named_parameters() if k.startswith(('local_tokens.','fracture_head.')) or ('lora' in k.lower() and p.requires_grad)}
    before={k:p.detach().clone() for k,p in names.items()};grad={k:0. for k in names};answer_grad={};history=[];updates=0;started=time.perf_counter()
    expected=read(P/'model_val.json')
    if smoke:expected=expected[:8]
    for epoch,batches in enumerate(schedules):
        seed_training_epoch(42,epoch);model.train();opt.zero_grad(set_to_none=True);losses=[];aux=[];norm=[];epoch_start=time.perf_counter()
        for i,indices in enumerate(batches):
            output=model(batch('train',indices),stage=3);assert torch.isfinite(output['loss'])
            if smoke and i==0 and arm!='B':
                params={k:p for k,p in names.items() if k.startswith('local_tokens.attention') or k.startswith('local_tokens.question')}
                values=torch.autograd.grad(output['loss_answer'],list(params.values()),retain_graph=True,allow_unused=True)
                answer_grad={k:float(v.abs().max()) if v is not None else 0 for k,v in zip(params,values)}
                assert any(v>0 for v in answer_grad.values())
            losses.append(float(output['loss'].detach()));aux.append(float(output.get('local_aux_loss',torch.tensor(0))));norm.append(float(output.get('local_delta_norm',torch.tensor(0))))
            (output['loss']/4).backward()
            for k,p in names.items():
                if p.grad is not None:assert torch.isfinite(p.grad).all();grad[k]=max(grad[k],float(p.grad.abs().max()))
            if (i+1)%4==0 or i+1==len(batches):
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1));opt.step();opt.zero_grad(set_to_none=True);updates+=1
        train_seconds=time.perf_counter()-epoch_start;model.eval();rows=[];eval_start=time.perf_counter()
        with torch.no_grad():
            for i in range(0,len(expected),4):
                predictions=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
                for e,p in zip(expected[i:i+4],predictions):
                    row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                    if 'p_yes' in p:row['p_yes']=float(p['p_yes'])
                    rows.append(row)
        assert len(rows)==len(expected);metrics=compute_metrics(rows,fracture_threshold=.5)
        dump(folder/f'epoch{epoch+1}_evaluation.json',{'records':rows,'metrics':metrics})
        model.save(str(folder/f'epoch{epoch+1}.pt'),{'updates':updates,'disposable':smoke,'model_class':'VNextModel'})
        torch.save({'optimizer':opt.state_dict(),'updates':updates},folder/f'epoch{epoch+1}_optimizer.pt')
        history.append({'epoch':epoch+1,'updates':updates,'train_seconds':train_seconds,'eval_seconds':time.perf_counter()-eval_start,'loss':sum(losses)/len(losses),'aux':sum(aux)/len(aux),'delta_norm':sum(norm)/len(norm),'metrics':metrics});print(json.dumps(history[-1]),flush=True)
    assert updates==(2 if smoke else 292)
    changes={k:float((p-before[k]).detach().abs().max()) for k,p in names.items()}
    assert any(v>0 for k,v in grad.items() if k.startswith('fracture_head.'))
    mechanics={}
    if smoke and arm!='B':
        assert grad['local_tokens.attention.in_proj_weight']>0 and changes['local_tokens.attention.in_proj_weight']>0
        if arm.endswith('_aux'):assert grad['local_tokens.aux_head.weight']>0
        public=batch('train',selected,True);model.eval()
        with torch.no_grad():
            normal=core_fixture(model,public)
            empty=dict(public);empty['boxes']=[None]*4;empty['masks']=[None]*4
            no_local=core_fixture(model,empty)
            base=BoneVQAModel._encode_core(model,empty['images'],empty['questions'],[None]*4,model._prepare_masks(empty['images'],None,None,None),'contour',False)
            assert torch.equal(no_local['fracture_logit'],base['fracture_logit'])
            model.cfg['vnext_force_zero']=True;forced=core_fixture(model,public);model.cfg['vnext_force_zero']=False
            assert torch.equal(forced['fracture_logit'],base['fracture_logit'])
            change=float((normal['fracture_logit']-forced['fracture_logit']).abs().max());assert change>0
            pred=plain(model.predict_batch(public));contaminated=dict(public,answers=['wrong']*4,answer_types=['OPEN']*4,regions=['wrong']*4)
            assert plain(model.predict_batch(contaminated))==pred
            open_only=batch('train',[selected[2],selected[3]])
            test=model(open_only,stage=3);assert float(test['local_aux_loss'])==0
        mechanics={'empty_equals_global':True,'forced_zero_equals_global':True,'fracture_logit_intervention_max':change,'public_metadata_invariant':True,'aux_open_rows_zero':True}
    dump(folder/'completion.json',{'history':history,'updates':updates,'elapsed_seconds':time.perf_counter()-started,'gradients':grad,'parameter_changes':changes,'answer_route_gradients':answer_grad,'mechanics':mechanics,'fresh_replay_pending':True})

def replay(arm,smoke):
    cfg=read(O/(arm+'.json'));folder=O/(('gate_' if smoke else 'run_')+arm);completion=read(folder/'completion.json');epoch=len(completion['history']);path=folder/f'epoch{epoch}.pt';digest=sha(path)
    set_seed(90210);model=VNextModel.load(str(path),device='cpu').eval();batch=inputs(cfg,arm);expected=read(folder/f'epoch{epoch}_evaluation.json')['records'];rows=[]
    with torch.no_grad():
        for i in range(0,len(expected),4):
            predictions=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
            for e,p in zip(expected[i:i+4],predictions):
                row={k:e[k] for k in ('image_id','question','gt','answer_type')};row['pred']=p['answer']
                if 'p_yes' in p:row['p_yes']=float(p['p_yes'])
                rows.append(row)
    assert rows==expected and sha(path)==digest
    dump(folder/'fresh_replay.json',{'passed':True,'records_exact':len(rows),'checkpoint_sha256':digest})

def child(mode,arm):
    with (O/f'{mode}_{arm}.log').open('x') as log:subprocess.run([sys.executable,'-u',__file__,mode,arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def operate():
    prepare();dump(O/'coordinator.json',{'pid':os.getpid(),'arms':ARMS,'start_time':time.time()})
    for arm in ARMS:child('smoke',arm);child('replay_smoke',arm)
    release=read(O/'gate_release.json');check(release);release['quality_released']=True
    release['gate_evidence']={a:{'completion':sha(O/f'gate_{a}/completion.json'),'replay':sha(O/f'gate_{a}/fresh_replay.json')} for a in ARMS}
    dump(O/'training_release.json',release)
    for arm in ARMS:child('quality',arm);child('replay_quality',arm)
    dump(O/'execution_completed.json',{'arms':ARMS,'full_replays_passed':True,'analysis_pending':True})

if __name__=='__main__':
    assert os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1';torch.set_num_threads(8)
    try:
        if len(sys.argv)==1:operate()
        elif sys.argv[1].startswith('replay'):replay(sys.argv[2],sys.argv[1]=='replay_smoke')
        else:worker(sys.argv[2],sys.argv[1]=='smoke')
    except Exception as e:
        if O.exists():
            p=O/('failure_'+str(os.getpid())+'.json')
            if not p.exists():dump(p,{'error':repr(e),'argv':sys.argv,'preserve_outputs':True})
        raise
