from common import *
import csv,math,types,time,collections
import numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from src.models.bonevqa import BoneVQAModel,normalize_batch
from src.data import collate_fn
from src.utils import compute_metrics,normalize_answer,normalize_yes_no

def csvwrite(path,rows):
 if not rows:raise ValueError('Empty CSV not allowed '+str(path))
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def family(q):
 q=q.lower()
 if q.startswith('is there a fracture'):return 'fracture presence'
 if q.startswith('is this an x-ray of'):return 'body region'
 if q.startswith('is the ') and q.endswith('fractured?'):return 'region-specific fracture'
 if 'view x-ray' in q:return 'projection'
 if 'hardware' in q:return 'hardware'
 if 'multiple fractures' in q:return 'multiple fracture'
 if 'more than one scan' in q:return 'multiple scans'
 return 'open'

def richer_metrics(records):
 metrics=compute_metrics(records);by=collections.defaultdict(list)
 for r in records:
  if r['answer_type']=='CLOSED':by[family(r['question'])].append(r)
 fam={}
 for name,rs in by.items():
  cls=sorted({normalize_yes_no(r['gt']) for r in rs});acc=float(np.mean([normalize_yes_no(r['pred'])==normalize_yes_no(r['gt']) for r in rs]));bal=float(np.mean([np.mean([normalize_yes_no(r['pred'])==c for r in rs if normalize_yes_no(r['gt'])==c]) for c in cls]))
  fam[name]={'n':len(rs),'accuracy':acc,'balanced_accuracy':bal,'class_counts':dict(collections.Counter(normalize_yes_no(r['gt']) for r in rs))}
 metrics['macro_family_accuracy']=float(np.mean([r['accuracy'] for r in fam.values()]));metrics['family_metrics']=fam
 metrics['n_images']=len({r['image_id'] for r in records});metrics['n_closed_correct']=sum(normalize_yes_no(r['pred'])==normalize_yes_no(r['gt']) for r in records if r['answer_type']=='CLOSED')
 return metrics

def check_predictions(records,split):
 manifest=read(P/f'{split}_manifest.json');expected=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in manifest];actual=[(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in records]
 assert expected==actual,'Evaluator keys/order or answer mismatch'
 counts=(len(records),len({r['image_id'] for r in records}),sum(r['answer_type']=='CLOSED' for r in records));assert counts==SPECS[split],counts

class FrozenDataset(Dataset):
 def __init__(self,split,regime,audit):
  self.records=read(P/f'{split}_manifest.json');self.split=split;self.regime=regime;self.audit=audit
  self.index={r['image_id']:r for r in read(ROOT/'roi_bank'/regime/'index.json')};self.images={r['image_id']:r for r in read(P/'image_content_manifest.json')};self.checked=set()
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  r=self.records[i];iid=r['image_id'];a=self.index[iid];im=self.images[iid]
  if iid not in self.checked:
   assert sha(im['path'])==im['sha256'],'Image content changed';assert sha(a['path'])==a['mask_sha256'],'ROI corrupted'
   assert a['roi_failed']==0 and a['roi_success'];self.checked.add(iid)
  with Image.open(im['path']) as image:image=image.convert('RGB')
  with np.load(a['path']) as z:masks=z['masks'].copy()
  assert masks.shape==(a['mask_count'],image.height,image.width)
  self.audit.setdefault(self.split,set()).add(iid)
  return dict(image=image,question=r['question'],answer=r['answer'],answer_type=r['answer_type'],region=r.get('region'),image_id=iid,masks=masks,boxes=None)

def frozen_masks(self,images,masks,boxes,image_ids):
 assert masks is not None and len(masks)==len(images)
 for im,m in zip(images,masks):assert m is not None and m.ndim==3 and m.shape[1:]==(im.height,im.width)
 return masks

def safe_layernorm_forward(self, x):
 # PyTorch 2.2.2 MPS produces invalid affine gradients when the input is frozen.
 # A dummy input leaf requests the working backward path without changing values
 # or propagating gradients into the frozen encoder.
 if torch.is_grad_enabled() and x.device.type=='mps' and not x.requires_grad and ((self.weight is not None and self.weight.requires_grad) or (self.bias is not None and self.bias.requires_grad)):
  x=x.detach().requires_grad_(True)
 return torch.nn.functional.layer_norm(x,self.normalized_shape,self.weight,self.bias,self.eps)

def make_model(cfg,vocab,checkpoint=None):
 model=BoneVQAModel(cfg,vocab,device='mps');assert model.visual.kind=='open_clip','Refusing encoder fallback'
 if checkpoint:
  payload=torch.load(checkpoint,map_location='cpu',weights_only=False);own=model.state_dict()
  state={k:v for k,v in payload['state'].items() if k in own and own[k].shape==v.shape};model.load_state_dict(state,strict=False)
  print('INIT loaded',len(state),'skipped',len(payload['state'])-len(state),flush=True)
 model._prepare_masks=types.MethodType(frozen_masks,model)
 for module in model.modules():
  if isinstance(module,torch.nn.LayerNorm):module.forward=types.MethodType(safe_layernorm_forward,module)
 model.to('mps')
 return model

@torch.no_grad()
def evaluate(model,loader,split,with_loss=False):
 model.eval();records=[];losses=[];t=time.time()
 for n,batch in enumerate(loader):
  if with_loss:
   loss=model(batch,stage=3)['loss'];assert torch.isfinite(loss);losses.append(float(loss))
  preds=model.predict_batch(batch);nb=normalize_batch(batch)
  for p,a,typ,q,iid in zip(preds,nb['answers'],nb['answer_types'],nb['questions'],nb['image_ids']):
   confidence=float(p['confidence']);confidence_nonfinite=not np.isfinite(confidence)
   # Confidence is audit metadata, never an endpoint.  Some legacy Qwen
   # checkpoints yield NaN sequence probability on MPS although greedy tokens
   # are finite and repeatable.  Preserve the event and use a stable sentinel.
   if confidence_nonfinite:confidence=0.0
   records.append({'image_id':iid,'question':q,'gt':a,'pred':p['answer'],'answer_type':typ,'pred_type':p['answer_type'],'confidence':confidence,'confidence_nonfinite':confidence_nonfinite})
  if (n+1)%100==0:print('EVAL',split,'QA',len(records),'seconds',round(time.time()-t),flush=True)
 check_predictions(records,split);metrics=richer_metrics(records)
 metrics['nonfinite_confidence_count']=sum(r['confidence_nonfinite'] for r in records)
 if losses:metrics['val_loss']=float(np.mean(losses))
 return metrics,records

def tensor_hash(state):
 h=hashlib.sha256()
 for k in sorted(state):
  t=state[k].detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
 return h.hexdigest()

def run_config(regime,variant,seed):
 cfg=read(P/'base_config.json');cfg['seed']=seed;cfg['localization_regime']=regime;cfg['vlp_switch']=variant
 for k in ['use_visual_prompt','use_lens','use_latent_prompt']:cfg['model'][k]=variant=='vlp'
 return cfg

def finish_roi_audit(dest,regime,variant,seen,complete=True):
 source=read(ROOT/'roi_bank'/regime/'audit.json');source['roi_branch_enabled']=variant=='vlp';source['run_dataset_access']={s:{'expected_images':SPECS[s][1],'processed_images':len(seen.get(s,set()))} for s in SPECS}
 source['expected_images']=2023;source['processed_images']=len(set().union(*seen.values())) if seen else 0
 source['roi_success']=source['processed_images'];source['roi_failed']=0;source['bank_index_sha256']=sha(ROOT/'roi_bank'/regime/'index.json');source['complete']=complete
 if complete:
  assert source['processed_images']==2023
  for s in SPECS:assert len(seen.get(s,set()))==SPECS[s][1]
 write(dest/'roi_audit.json',source)


def train_run(regime,variant,seed,dest):
 setup(seed);lock=verify_lock();cfg=run_config(regime,variant,seed);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 if (dest/'QC.json').exists():raise RuntimeError('Completed run exists; refusing overwrite')
 write(dest/'config.json',cfg);(dest/'seed.txt').write_text(str(seed)+'\n');(dest/'environment.txt').write_text((P/'environment.txt').read_text())
 for split in SPECS:(dest/f'{split}_manifest_hash.txt').write_text(sha(P/f'{split}_manifest.json')+'\n')
 (dest/'protocol_hash.txt').write_text(sha(P/'LOCK.json')+'\n')
 vocab=read(P/'answer_vocab.json');write(dest/'answer_vocab.json',vocab);audit={};ds={s:FrozenDataset(s,regime,audit) for s in SPECS}
 # Independent generator prevents architecture-dependent initialization draws from changing sample order.
 generator=torch.Generator().manual_seed(seed)
 train_loader=DataLoader(ds['train'],batch_size=4,shuffle=True,generator=generator,num_workers=0,drop_last=True,collate_fn=collate_fn)
 val_loader=DataLoader(ds['val'],batch_size=4,shuffle=False,num_workers=0,collate_fn=collate_fn)
 test_loader=DataLoader(ds['test'],batch_size=4,shuffle=False,num_workers=0,collate_fn=collate_fn)
 model=make_model(cfg['model'],vocab,cfg['init_from']);params=[p for p in model.parameters() if p.requires_grad]
 tcfg=cfg['train'];opt=torch.optim.AdamW(params,lr=tcfg['lr'],weight_decay=tcfg['weight_decay'],betas=(.9,.98));epochs=tcfg['epochs'];accum=tcfg['grad_accum'];total_steps=math.ceil(len(train_loader)/accum)*epochs;warmup=int(tcfg['warmup_ratio']*total_steps);assert total_steps==750
 def lr(s):
  if s<warmup:return (s+1)/max(1,warmup)
  return .5*(1+math.cos(math.pi*min(1.,(s-warmup)/max(1,total_steps-warmup))))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,lr);best=None;history=[];step=0;best_epoch=None
 for epoch in range(epochs):
  model.train();opt.zero_grad(set_to_none=True);running=collections.defaultdict(float);t=time.time();order=[]
  for it,batch in enumerate(train_loader):
   order.extend(zip(batch['image_ids'],batch['questions']));out=model(batch,stage=3);loss=out['loss'];assert torch.isfinite(loss), 'NaN/Inf loss'
   (loss/accum).backward()
   for key in ['loss','loss_cls','loss_lm','loss_cs','loss_con','loss_type']:running[key]+=float(out[key].detach())
   if (it+1)%accum==0 or it+1==len(train_loader):
    norm=torch.nn.utils.clip_grad_norm_(params,tcfg['max_grad_norm'],error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);sched.step();step+=1
    if step%10==0:
     elapsed=time.time()-t;print('TRAIN',regime,variant,seed,'epoch',epoch+1,'step',step,'/750','loss',running['loss']/(it+1),'elapsed_s',round(elapsed),'lr',sched.get_last_lr()[0],flush=True);status('train',run=str(dest.relative_to(ROOT)),epoch=epoch+1,step=step,total_steps=750,elapsed_epoch_s=elapsed);finish_roi_audit(dest,regime,variant,audit,False)
  assert len(order)==6000
  write(dest/f'epoch_{epoch+1}_order_hash.json',{'sha256':jhash(order),'n_qa':len(order)})
  row={'epoch':epoch+1,'global_step':step,'train_seconds':time.time()-t,**{'train_'+k:v/len(train_loader) for k,v in running.items()}}
  metrics,records=evaluate(model,val_loader,'val',with_loss=True);row.update({'val_'+k:v for k,v in metrics.items() if isinstance(v,(int,float))});history.append(row)
  csvwrite(dest/'training_history.csv',history);write(dest/'training_history.json',history);csvwrite(dest/f'validation_epoch_{epoch+1}.csv',records)
  score=metrics['overall_acc']
  if best is None or score>best:
   best=score;best_epoch=epoch+1
   model.save(str(dest/'checkpoint_best.pt'),{'seed':seed,'regime':regime,'variant':variant,'epoch':best_epoch,'best_metric':best,'global_step':step,'protocol_sha256':sha(P/'LOCK.json')})
   csvwrite(dest/'validation_predictions.csv',records);write(dest/'validation_metrics.json',metrics)
  print('EPOCH COMPLETE',row,'selected',best_epoch,flush=True)
 assert step==750
 best_state=torch.load(dest/'checkpoint_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best_state['state'],strict=False);model.eval()
 (dest/'checkpoint_hash.txt').write_text(sha(dest/'checkpoint_best.pt')+'\n');(dest/'checkpoint_tensor_hash.txt').write_text(tensor_hash(best_state['state'])+'\n');del best_state
 metrics,records=evaluate(model,test_loader,'test');csvwrite(dest/'test_predictions.csv',records);write(dest/'test_predictions.json',records);write(dest/'metrics.json',metrics)
 finish_roi_audit(dest,regime,variant,audit,True)
 for name in SPECS:assert (dest/f'{name}_manifest_hash.txt').read_text().strip()==sha(P/f'{name}_manifest.json')
 for p in params:assert torch.isfinite(p).all(),'Nonfinite final parameters'
 verify_lock()
 write(dest/'QC.json',{'valid':True,'seed':seed,'regime':regime,'config':variant,'manifest_hashes':{s:sha(P/f'{s}_manifest.json') for s in SPECS},'roi_failed':0,'test_qa':2773,'closed_qa':1546,'test_images':612,'selected_epoch':best_epoch,'selected_val_overall_acc':best,'checkpoint_sha256':sha(dest/'checkpoint_best.pt'),'finished':datetime.datetime.now().isoformat(),'runtime_source_lock':sha(P/'LOCK.json')})
 print('RUN COMPLETE',dest,metrics,flush=True)
