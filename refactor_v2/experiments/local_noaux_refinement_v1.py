"""Bounded five-fold B/local study. Versioned outputs, no historical writes."""
import copy, gc, hashlib, json, math, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path('/Users/uyen/Downloads/BONEVQA')
R=ROOT/'refactor_v2/reports'
O=R/'local_noaux_refinement_v1'
CAN='Is there a fracture in this X-ray?'
ARMS=('B','local_real_noaux')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v,exclusive=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x') as f:json.dump(v,f,indent=2)
    else:
        tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def guards(start=None,cap=8*3600):
    import shutil
    assert shutil.disk_usage(ROOT).free>=8*1024**3,'disk reserve'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<20*1024**3,'RSS budget'
    if start is not None:assert time.time()-start<cap,'wall budget'
def verify():
    rel=read(O/'preparation.json')
    for p,h in rel['dependencies'].items():assert sha(p)==h,p
    return rel
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def frozen_hash(model):
    h=hashlib.sha256()
    for k,p in model.named_parameters():
        if not p.requires_grad:h.update(k.encode());h.update(bytes.fromhex(tensor_hash(p)))
    return h.hexdigest()
def prepare():
    assert not O.exists(),'Existing study; inspect before action'
    O.mkdir();bindings=read(R/'grouped_evaluation_bindings_v3_1/bindings.json')['folds']
    metadata={};paths=[Path(__file__),R/'LOCAL_NOAUX_REFINEMENT_PROTOCOL_20261027.md']
    paths+=list((ROOT/'refactor_v2/src').rglob('*.py'))
    paths += [ROOT/'refactor_v2/experiments'/n for n in ('vnext_model_v1.py','detector640_checkpoint_v3.py','private_TRAIN_sequence_loss_adapter_v3_1.py','sequence_answer_loss_candidate_v3_1.py')]
    paths += [ROOT/'refactor_v2/scripts/train_detector_v2.py']
    for fold,b in bindings.items():
        cfg=read(R/f'grouped_native_preparation_v3_2/fold_{fold}.json')
        roleids={role:{r['image_id'] for r in read(x['path'])} for role,x in b['roles'].items()}
        for role,x in b['roles'].items():assert sha(x['path'])==x['sha256'];paths.append(Path(x['path']))
        assert all(not roleids[a]&roleids[z] for a in roleids for z in roleids if a<z)
        paths += [Path(cfg['init_from']),Path(cfg['data']['sealed_answer_vocab']),Path(cfg['train']['component_batching']['manifest']),Path(cfg['train']['component_batching']['seal'])]
        paths += [Path(cfg['data']['loader_kwargs']['seal_path'])]
        schedule=read(R/f'grouped_native_preparation_v3_2/fold_{fold}_schedule.json')
        assert len(schedule)==2
        rows=read(b['roles']['train']['path'])
        for e in schedule:assert sorted(i for batch in e['ordered_indices'] for i in batch)==list(range(len(rows)))
        folder=O/f'fold_{fold}';folder.mkdir()
        for arm in ARMS+(('local_placebo_noaux',) if fold=='0' else ()):
            c=copy.deepcopy(cfg);c['run_name']=f'local_noaux_fold{fold}_{arm}_seed42'
            c['model'].update(vnext_enabled=arm!='B',vnext_aux_weight=0.0,vnext_force_zero=False)
            c['checkpoint_dir']=str(folder);c['recipe_status']='new_bounded_study_pending_gate'
            c['data']['loader_kwargs']['sealed_records_by_split']={role:b['roles'][role]['path'] if role in ('train','model_val') else None for role in b['roles']}
            dump(folder/(arm+'.json'),c,True);paths.append(folder/(arm+'.json'))
        dump(folder/'schedule.json',schedule,True);paths.append(folder/'schedule.json')
        metadata[fold]={'roles':b['roles'],'historical_B':b['checkpoint'],'historical_B_sha256':b['checkpoint_sha256'],'config_parent':b['config']}
        paths.append(Path(b['checkpoint']))
    dump(O/'metadata.json',metadata,True);paths.append(O/'metadata.json')
    dump(O/'preparation.json',{'dependencies':{str(p):sha(p) for p in sorted(set(paths))},'quality_released':False,'folds':list(metadata),'seed':42,'epochs':2,'created_wall':time.time()},True)

def detector(fold):
    from scripts.train_detector_v2 import BoPhatHienDataset,PositiveAnchoredBatchSampler,collate,tao_mo_hinh
    from src.utils import set_seed
    from torch.utils.data import DataLoader
    from src.preprocessing import RASTER_PREPROCESSING_VERSION
    verify();folder=O/f'fold_{fold}';dest=folder/'detector'
    assert not dest.exists();dest.mkdir();meta=read(O/'metadata.json')[fold]
    if fold=='0':
        p=ROOT/'refactor_v2/detector640_fold0_bounded_quality_v3_1/last.pt'
        expected=read(R/'detector640_fold0_bounded_quality_v3_1/completion.json')['last_sha256'];assert sha(p)==expected
        payload=torch.load(p,map_location='cpu',weights_only=False)
        assert set(payload['training_image_ids'])=={r['image_id'] for r in read(meta['roles']['train']['path'])}
        dump(dest/'receipt.json',{'checkpoint':str(p),'sha256':expected,'reused_fold0_only':True},True);return
    ids={r['image_id'] for r in read(meta['roles']['train']['path'])}
    ds=BoPhatHienDataset('train',canh_dai=640,tang_cuong=True,allowed_ids=ids)
    # Match source boxes against this fold's existing TRAIN annotations.
    from src.detector_protocol import verify_detector_annotation_parity
    verify_detector_annotation_parity(ds,meta['roles']['train']['path'])
    set_seed(42);model=tao_mo_hinh('mobilenet_320').cpu();model.transform.min_size=(640,);model.transform.max_size=640
    norm={k:{'type':type(v).__name__,'eps':v.eps} for k,v in model.named_modules() if hasattr(v,'running_mean') and hasattr(v,'eps')}
    assert all(v['type']=='FrozenBatchNorm2d' for v in norm.values())
    sampler=PositiveAnchoredBatchSampler(ds,4,42);loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=0)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2);updates=0;start=time.time();history=[]
    for epoch in range(2):
        sampler.epoch=epoch;model.train();losses=[]
        for images,targets,_ in loader:
            guards(start,7200);loss=sum(model(list(images),list(targets)).values());assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),5));opt.step();updates+=1;losses.append(float(loss.detach()))
            if updates%25==0:dump(dest/'progress.json',{'epoch':epoch+1,'updates':updates,'loss':losses[-1],'wall_seconds':time.time()-start})
        sched.step();history.append({'epoch':epoch+1,'updates':updates,'mean_loss':float(np.mean(losses))})
    p=dest/'LAST.pt';torch.save({'state':model.state_dict(),'architecture':'mobilenet_320','canh_dai':640,'detector_transform':{'min_size':[640],'max_size':640,'size_divisible':32},'normalization_spec':norm,'training_image_ids':sorted(ids),'preprocessing_version':RASTER_PREPROCESSING_VERSION,'initial_weight_provenance':model.initial_weight_provenance,'epoch':2,'optimizer_steps':updates},p)
    dump(dest/'receipt.json',{'checkpoint':str(p),'sha256':sha(p),'TRAIN_ids':sorted(ids),'TRAIN_manifest_sha256':sha(meta['roles']['train']['path']),'history':history,'wall_seconds':time.time()-start},True)

def proposals(fold):
    from experiments.detector640_checkpoint_v3 import load_bound_640
    from src.detector_runtime import FractureDetectorRuntime
    from src.data.fracatlas_vqa import FRACATLAS_ROOT,FracAtlasVQADataset
    from src.preprocessing import canonicalize_raster
    verify();folder=O/f'fold_{fold}';assert not (folder/'boxes.json').exists()
    receipt=read(folder/'detector/receipt.json');assert sha(receipt['checkpoint'])==receipt['sha256']
    runtime=object.__new__(FractureDetectorRuntime);runtime.model,payload=load_bound_640(receipt['checkpoint']);runtime.device=torch.device('cpu');runtime.image_size=640;runtime.score_threshold=.05;runtime.max_boxes=3
    meta=read(O/'metadata.json')[fold];unique={};roles={}
    for role,b in meta['roles'].items():
        for row in read(b['path']):unique[row['image_id']]=row['image'];roles[row['image_id']]=role
    assert set(payload['training_image_ids'])=={i for i in unique if roles[i]=='train'}
    real={};placebo={};info={};start=time.time()
    for iid,rel in sorted(unique.items()):
        guards(start,7200);p=FRACATLAS_ROOT/rel
        with Image.open(p) as im:
            im=canonicalize_raster(im);boxes=runtime.predict(im);size=im.size
        assert len(boxes)<=3
        real[iid]=boxes;placebo[iid]=FracAtlasVQADataset._matched_placebo_boxes(boxes,size) or []
        assert len(real[iid])==len(placebo[iid])
        info[iid]={'image_sha256':sha(p),'size':size,'role':roles[iid]}
        if len(real)%50==0:dump(folder/'proposal_progress.json',{'images':len(real),'total':len(unique),'elapsed':time.time()-start})
    dump(folder/'boxes.json',real,True);dump(folder/'placebo.json',placebo,True)
    dump(folder/'proposal_receipt.json',{'detector':receipt,'images':info,'boxes_sha256':sha(folder/'boxes.json'),'placebo_sha256':sha(folder/'placebo.json'),'nonempty':{r:sum(bool(real[i]) for i in real if roles[i]==r) for r in meta['roles']},'wall_seconds':time.time()-start},True)

def config(fold,arm):return read(O/f'fold_{fold}'/(arm+'.json'))
def construct(fold,arm):
    from experiments.vnext_model_v1 import VNextModel
    from src.utils import set_seed
    cfg=config(fold,arm);payload=torch.load(cfg['init_from'],map_location='cpu',weights_only=False)
    set_seed(42);m=VNextModel(cfg['model'],payload['answer_vocab'],device='cpu')
    result=m.load_state_dict(payload['state'],strict=False)
    assert not result.unexpected_keys
    current=m.trainable_state_dict()
    assert set(payload['state'])<=set(current)
    for k,t in payload['state'].items():assert torch.equal(t,current[k]),k
    assert {k for k in current if k not in payload['state']}=={k for k in current if k.startswith('local_tokens.')}
    m.to('cpu');return m,cfg
def datasets(fold):
    from src.data.fracatlas_vqa import FracAtlasVQADataset
    meta=read(O/'metadata.json')[fold]
    return {r:FracAtlasVQADataset(records_path=b['path'],with_masks=False,with_boxes=False) for r,b in meta['roles'].items()}
def batcher(fold,arm):
    from src.data import collate_fn
    from src.data.fracatlas_vqa import FracAtlasVQADataset
    from src.context_controls import load_component_lookup
    cfg=config(fold,arm);ds=datasets(fold)
    lookup,_=load_component_lookup(cfg['train']['component_batching']['manifest'],cfg['train']['component_batching']['seal'],ROOT)
    folder=O/f'fold_{fold}';receipt=read(folder/'proposal_receipt.json')
    p=folder/('placebo.json' if arm=='local_placebo_noaux' else 'boxes.json')
    assert sha(p)==receipt['placebo_sha256' if arm=='local_placebo_noaux' else 'boxes_sha256']
    boxes=read(p)
    def collate(samples):
        b=collate_fn(samples);b['component_ids']=[lookup[i] for i in b['image_ids']]
        b['boxes']=[None]*len(samples);b['masks']=[None]*len(samples)
        if arm!='B':
            b['boxes']=[boxes[i] or None for i in b['image_ids']]
            b['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
        return b
    return ds,collate
def public(b,arm):return {k:b[k] for k in (('images','questions') if arm=='B' else ('images','questions','boxes','masks'))}
def predict(model,fold,arm,role,canonical=False,limit=None,progress_path=None):
    ds,collate=batcher(fold,arm);expected=ds[role].records
    indices=[i for i,e in enumerate(expected) if not canonical or e['question']==CAN]
    if limit:indices=indices[:limit]
    rows=[];model.eval();start=time.time()
    with torch.no_grad():
        for offset in range(0,len(indices),4):
            guards(start);ids=indices[offset:offset+4];b=collate([ds[role][i] for i in ids]);out=model.predict_batch(public(b,arm))
            for i,p in zip(ids,out):
                e=expected[i];row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                if 'p_yes' in p:row['p_yes']=float(p['p_yes']);assert math.isfinite(row['p_yes']) and 0<=row['p_yes']<=1
                if e['question']==CAN:assert 'p_yes' in row
                rows.append(row)
            if progress_path and offset%80==0:dump(progress_path,{'role':role,'done':len(rows),'total':len(indices),'seconds':time.time()-start})
    return rows

def gate(fold,arm):
    from src.utils import set_seed
    from experiments.vnext_model_v1 import VNextModel
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    verify();folder=O/f'fold_{fold}'/('gate_'+arm);assert not folder.exists();folder.mkdir()
    model,cfg=construct(fold,arm);ds,collate=batcher(fold,arm);boxes=read(O/f'fold_{fold}/boxes.json')
    selected=[]
    for i,r in enumerate(ds['train'].records):
        if boxes[r['image_id']] and (r['question']==CAN or r['answer_type']=='OPEN'):selected.append(i)
        if len(selected)==4:break
    assert len(selected)==4
    b=collate([ds['train'][i] for i in selected]);before={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad};fh=frozen_hash(model)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.98),weight_decay=.01)
    model.train();start=time.time();grads={};table={(r['question'],str(r['answer']).strip().lower().strip(' .')):1. for r in ds['train'].records if r['answer_type']=='OPEN'}
    # Use the project normalizer, not a fixture-only key approximation.
    from src.utils import normalize_answer
    table={(r['question'],normalize_answer(r['answer'])):1. for r in ds['train'].records if r['answer_type']=='OPEN'}
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        for micro in range(4):
            guards(start,1800)
            with private_training_loss(model,b,3,'token_mean',table):out=model(b,stage=3)
            assert torch.isfinite(out['loss']);(out['loss']/4).backward()
        for k,p in model.named_parameters():
            if p.grad is not None:assert torch.isfinite(p.grad).all();grads[k]=max(grads.get(k,0),float(p.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    if arm!='B':assert grads['local_tokens.attention.in_proj_weight']>0
    assert fh==frozen_hash(model)
    expected=predict(model,fold,arm,'train',canonical=True,limit=8)
    model.save(str(folder/'LAST.pt'),{'updates':2,'disposable_gate':True})
    dump(folder/'expected.json',expected,True)
    changed=[k for k,p in model.named_parameters() if k in before and not torch.equal(p,before[k])]
    dump(folder/'completed.json',{'two_updates':True,'microbatches':8,'grads':grads,'changed':changed,'frozen_unchanged':True,'seconds':time.time()-start},True)
def gate_replay(fold,arm):
    from experiments.vnext_model_v1 import VNextModel
    verify();folder=O/f'fold_{fold}'/('gate_'+arm);m=VNextModel.load(str(folder/'LAST.pt'),device='cpu')
    assert predict(m,fold,arm,'train',canonical=True,limit=8)==read(folder/'expected.json')
    dump(folder/'replay.json',{'exact_records':8,'checkpoint_sha256':sha(folder/'LAST.pt')},True)

def train(fold,arm):
    from torch.utils.data import DataLoader
    from src.train import seed_training_epoch
    from src.utils import normalize_answer,compute_metrics
    from experiments.private_TRAIN_sequence_loss_adapter_v3_1 import private_training_loss
    verify();folder=O/f'fold_{fold}'/arm;assert not folder.exists();folder.mkdir()
    gatefolder=O/'fold_0'/('gate_'+('B' if arm=='B' else arm))
    assert (gatefolder/'replay.json').exists()
    model,cfg=construct(fold,arm);fh=frozen_hash(model);ds,collate=batcher(fold,arm)
    schedules=read(O/f'fold_{fold}/schedule.json');total=sum(e['updates'] for e in schedules);warmup=int(.05*total)
    table={(r['question'],normalize_answer(r['answer'])):1. for r in ds['train'].records if r['answer_type']=='OPEN'}
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg['train']['lr'],betas=(.9,.98),weight_decay=.01)
    def rate(step):
        if step<warmup:return (step+1)/max(1,warmup)
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total-warmup))))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,rate);updates=0;history=[];start=time.time();gradient_seen=set()
    initial={k:tensor_hash(p) for k,p in model.named_parameters() if p.requires_grad}
    dump(folder/'started.json',{'pid':os.getpid(),'wall':start,'expected_updates':total,'frozen_hash':fh,'initial_trainable_hashes':initial},True)
    for epoch,s in enumerate(schedules):
        seed_training_epoch(42,epoch);model.train();opt.zero_grad(set_to_none=True);losses=[];t0=time.time()
        loader=DataLoader(ds['train'],batch_sampler=s['ordered_indices'],collate_fn=collate,num_workers=0)
        for j,b in enumerate(loader):
            guards(start)
            with private_training_loss(model,b,3,'token_mean',table):out=model(b,stage=3)
            assert torch.isfinite(out['loss']);losses.append(float(out['loss'].detach()));(out['loss']/4).backward()
            if (j+1)%4==0 or j+1==len(loader):
                for k,p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all()
                        if torch.count_nonzero(p.grad):gradient_seen.add(k)
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(params,1));opt.step();opt.zero_grad(set_to_none=True);sched.step();updates+=1
                if updates%10==0:dump(folder/'progress.json',{'pid':os.getpid(),'epoch':epoch+1,'microbatch':j+1,'updates':updates,'total_updates':total,'loss':losses[-1],'lr':sched.get_last_lr()[0],'wall_seconds':time.time()-start})
        rows=predict(model,fold,arm,'model_val',progress_path=folder/'eval_progress.json');metrics=compute_metrics(rows,fracture_threshold=.5)
        dump(folder/f'epoch{epoch+1}_evaluation.json',{'records':rows,'metrics':metrics},True)
        history.append({'epoch':epoch+1,'updates':updates,'mean_loss':float(np.mean(losses)),'wall_seconds':time.time()-t0,'metrics':metrics})
        dump(folder/'history.json',history)
        print(json.dumps({'fold':fold,'arm':arm,**history[-1]}),flush=True)
    assert updates==total and frozen_hash(model)==fh
    if arm!='B':assert 'local_tokens.attention.in_proj_weight' in gradient_seen
    model.save(str(folder/'LAST.pt'),{'updates':updates,'epoch':2,'model_class':'VNextModel','fixed_LAST':True})
    changes=[k for k,p in model.named_parameters() if p.requires_grad and tensor_hash(p)!=initial[k]]
    dump(folder/'completed.json',{'updates':updates,'epochs':2,'frozen_unchanged':True,'gradients_seen':sorted(gradient_seen),'changed':changes,'checkpoint_sha256':sha(folder/'LAST.pt'),'wall_seconds':time.time()-start},True)

def evaluate(fold,arm,replay=False):
    from experiments.vnext_model_v1 import VNextModel
    from src.utils import compute_metrics
    verify();folder=O/f'fold_{fold}'/arm;path=folder/'LAST.pt';assert sha(path)==read(folder/'completed.json')['checkpoint_sha256']
    m=VNextModel.load(str(path),device='cpu').eval();state=frozen_hash(m);results={}
    roles=('model_val','calibration','test')
    for role in roles:
        rows=predict(m,fold,arm,role,canonical=(role=='calibration'),progress_path=folder/('replay_progress.json' if replay else 'assessment_progress.json'))
        if replay:
            expected=read(folder/(role+'_predictions.json'))['records'];assert rows==expected,(fold,arm,role)
            results[role]=len(rows)
        else:
            dump(folder/(role+'_predictions.json'),{'records':rows,'metrics':compute_metrics(rows,fracture_threshold=.5)},True)
    assert frozen_hash(m)==state
    dump(folder/('replay.json' if replay else 'evaluation.json'),{'passed':True,'checkpoint_sha256':sha(path),'roles':results if replay else list(roles),'frozen_unchanged':True},True)

def child(mode,fold,arm=None,timeout=8*3600):
    log=O/(mode+'_fold'+fold+('_'+arm if arm else '')+'.log')
    with log.open('x') as f:
        p=subprocess.Popen([sys.executable,'-u',__file__,mode,fold]+([arm] if arm else []),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(O/'active_child.json',{'pid':p.pid,'mode':mode,'fold':fold,'arm':arm,'started':time.time(),'cap':timeout})
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
        assert code==0,f'{mode}/{fold}/{arm} exit {code}; see {log}'
def operate():
    if not O.exists():prepare()
    assert not (O/'coordinator.json').exists(),'Existing coordinator; inspect no duplicate'
    os.close(os.open(O/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
    start=time.time();dump(O/'coordinator.json',{'pid':os.getpid(),'wall_start':start,'deadline':start+72*3600},True)
    state={'status':'preparing','completed':[]}
    try:
        for fold in map(str,range(5)):
            guards(start,72*3600)
            state.update(fold=fold,phase='detector');dump(O/'progress.json',state);child('detector',fold,timeout=7200);child('proposals',fold,timeout=7200)
            if fold=='0':
                for arm in ARMS+('local_placebo_noaux',):child('gate',fold,arm,1800);child('gate_replay',fold,arm,600)
                verify();dump(O/'quality_release.json',{'quality_released':True,'preparation_sha256':sha(O/'preparation.json'),'gate_replays':{a:sha(O/f'fold_0/gate_{a}/replay.json') for a in ARMS+('local_placebo_noaux',)},'wall':time.time()},True)
            for arm in ARMS+(('local_placebo_noaux',) if fold=='0' else ()):
                state.update(phase='training',arm=arm);dump(O/'progress.json',state)
                child('train',fold,arm);child('evaluate',fold,arm);child('replay',fold,arm)
                state['completed'].append([fold,arm]);dump(O/'progress.json',state)
        dump(O/'execution_completed.json',{'completed':state['completed'],'all_replays_passed':True,'head_operating_analysis_pending':True,'wall_seconds':time.time()-start},True)
        state.update(status='quality_completed_head_and_analysis_pending',phase='head_and_analysis_pending');dump(O/'progress.json',state)
    except Exception as e:
        state.update(status='failed_preserved',error=repr(e));dump(O/'progress.json',state);dump(O/'failure.json',{'error':repr(e),'wall':time.time(),'no_automatic_training_retry':True},True);raise

if __name__=='__main__':
    assert os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1'
    torch.set_num_threads(8)
    mode=sys.argv[1] if len(sys.argv)>1 else 'operate'
    if mode=='operate':operate()
    elif mode=='prepare':prepare()
    elif mode=='replay':evaluate(sys.argv[2],sys.argv[3],True)
    else:globals()[mode](*sys.argv[2:])
