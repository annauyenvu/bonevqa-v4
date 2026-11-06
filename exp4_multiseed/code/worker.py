from common import *
import argparse,traceback
from engine import *

def evaluate_checkpoint(checkpoint,dest,regime,seed):
 setup(seed);verify_lock();dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=payload['config'];cfg['llm_dtype']='float32';cfg['sam']['cache_dir']=str(ROOT/'unused_sam_cache');cfg['sam']['sam_med2d_ckpt']=None
 model=make_model(cfg,payload['answer_vocab'],checkpoint);model.eval();audit={};ds=FrozenDataset('test',regime,audit);loader=DataLoader(ds,batch_size=4,shuffle=False,num_workers=0,collate_fn=collate_fn)
 metrics,records=evaluate(model,loader,'test');csvwrite(dest/'test_predictions.csv',records);write(dest/'test_predictions.json',records);write(dest/'metrics.json',metrics)
 write(dest/'evaluation_audit.json',{'checkpoint_sha256':sha(checkpoint),'protocol_sha256':sha(P/'LOCK.json'),'seed':seed,'regime':regime,'test_images_processed':len(audit['test']),'n_qa':len(records),'n_closed':metrics['n_closed']})
 print('EVALUATION COMPLETE',dest,flush=True)

def smoke():
 setup(42);cfg=read(P/'base_config.json');bank=ROOT/'roi_bank/shared';meta={r['image_id']:r for r in read(P/'train_manifest.json')};sample=None
 for path in sorted(bank.glob('*.json')):
  a=read(path)
  if a['image_id_source'] in meta and a['mask_count']>0:sample=(a,meta[a['image_id_source']],path.with_suffix('.npz'));break
 assert sample,'No precomputed training ROI ready'
 a,r,path=sample
 with Image.open(LEGACY/'data/fracatlas/FracAtlas'/r['image']) as image:im=image.convert('RGB')
 m=np.load(path)['masks'];batch={'images':[im]*2,'questions':[r['question'],'Describe the finding.'],'answers':[r['answer'],'no fracture'],'answer_types':[r['answer_type'],'OPEN'],'regions':[None]*2,'masks':[m]*2,'boxes':[None]*2,'image_ids':[r['image_id']]*2}
 model=make_model(cfg['model'],read(P/'answer_vocab.json'),cfg['init_from']);model.train()
 out=model(batch,stage=3);print('SMOKE LOSSES',{k:float(v.detach()) for k,v in out.items() if k.startswith('loss')},flush=True);assert torch.isfinite(out['loss']);out['loss'].backward()
 bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()];print('BAD GRADIENTS',bad[:50],flush=True);print('CPU NORM',sum(float(p.grad.detach().cpu().double().square().sum()) for p in model.parameters() if p.grad is not None)**.5,flush=True)
 torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
 model.eval();model.zero_grad(set_to_none=True)
 with torch.no_grad():
  a=model.predict_batch(batch);b=model.predict_batch(batch)
 strip=lambda rs:[{k:r[k] for k in ['answer','answer_type','confidence']} for r in rs]
 print('REPEAT A',strip(a),'REPEAT B',strip(b),flush=True)
 assert [(x['answer'],x['answer_type']) for x in a]==[(x['answer'],x['answer_type']) for x in b],'Same model eval predictions not identical'
 write(P/'smoke_result.json',{'success':True,'loss_finite':True,'gradient_finite':True,'same_model_prediction_identity':True,'confidence_max_abs_diff':max(abs(x['confidence']-y['confidence']) for x,y in zip(a,b)),'example_image_id':r['image_id'],'predictions':strip(a)})
 print('SMOKE PASS',strip(a),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval','smoke']);p.add_argument('--regime',choices=REGIMES,default='auto');p.add_argument('--variant',choices=CONFIGS,default='vlp');p.add_argument('--seed',type=int,default=42);p.add_argument('--dest');p.add_argument('--checkpoint');a=p.parse_args()
 try:
  if a.mode=='train':train_run(a.regime,a.variant,a.seed,a.dest)
  elif a.mode=='eval':evaluate_checkpoint(a.checkpoint,a.dest,a.regime,a.seed)
  else:smoke()
 except Exception as exc:
  if a.dest:
   write(Path(a.dest)/'QC.json',{'valid':False,'technical_failure':repr(exc),'traceback':traceback.format_exc(),'seed':a.seed,'regime':a.regime,'variant':a.variant});
  status('technical_failure','error',mode=a.mode,run=a.dest,error=repr(exc));raise
