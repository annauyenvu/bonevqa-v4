"""Generate and audit immutable SAM output; no historical ROI cache is consumed."""
from common import *
import numpy as np,torch,time
from PIL import Image
from src.models.segment_prompt import SegmentPromptCreator
import torch.nn.functional as F

def main():
 setup(42);ms=manifests();cfg=read(P/'base_config.json')['model'];meta={r['image_id']:r for rows in ms.values() for r in rows};image_meta={r['image_id']:r for r in read(P/'image_content_manifest.json')};detector=read(LEGACY/'data/fracatlas/box_bo_phat_hien.json')
 assert all(iid in detector for iid in meta),'Missing detector entries'
 lock=read(P/'inputs_lock.json')
 for file,digest in {**lock['files'],**lock['pretrained']}.items():assert sha(ROOT/file)==digest, file
 bank=ROOT/'roi_bank';bank.mkdir(exist_ok=True);shared=bank/'shared';shared.mkdir(exist_ok=True)
 model=SegmentPromptCreator(model_id=cfg['sam']['model_id'],device='mps',max_masks=cfg['n_masks_max'],grid=cfg['sam']['grid'],default_prompt=cfg['sam']['default_prompt'],cache_dir=None,enabled=True,max_area_ratio=.6,min_mean_intensity=40.0)
 original=F.interpolate;counter={'calls':0}
 def interpolate(input,*args,**kwargs):
  mode=kwargs.get('mode',args[2] if len(args)>2 else 'nearest')
  if input.device.type=='mps' and input.ndim==3 and mode=='linear':
   counter['calls']+=1
   return original(input.cpu(),*args,**kwargs).to(input.device)
  return original(input,*args,**kwargs)
 F.interpolate=interpolate
 spec={'sam':cfg['sam']['model_id'],'pretrained_files':lock['pretrained'],'max_masks':3,'grid':2,'min_area':.002,'max_area':.6,'min_intensity':40,'iou_dedup':.85,'float32':True,'linear_interpolate_cpu':True}
 write(bank/'generation_spec.json',spec)
 for regime in REGIMES:
  folder=bank/regime;folder.mkdir(exist_ok=True);rows=[]
  for n,(iid,r) in enumerate(sorted(meta.items())):
   path=Path(image_meta[iid]['path']);assert sha(path)==image_meta[iid]['sha256']
   boxes=None if regime=='auto' else (detector[iid] if regime=='detector' else r.get('boxes'))
   if not boxes:boxes=None
   # Include actual box list and pixel hash, not just image ID.
   key=jhash({'image_sha256':image_meta[iid]['sha256'],'boxes':boxes,'spec':spec});mp=shared/(key+'.npz');jp=shared/(key+'.json')
   with Image.open(path) as im:im=im.convert('RGB');w,h=im.size
   if not mp.exists() or not jp.exists():
    count=counter['calls'];t=time.time()
    try:
     with torch.inference_mode():masks=model(im,boxes=boxes,cache_key=None)
     assert masks.ndim==3 and masks.shape[1:]==(h,w) and len(masks)<=3
     assert np.isin(masks,[0,1]).all()
     temp=mp.with_suffix('.tmp.npz');np.savez_compressed(temp,masks=masks);temp.replace(mp)
     audit={'image_id_source':iid,'mask_sha256':sha(mp),'mask_count':len(masks),'roi_success':True,'roi_failed':0,'sam_fallback_count':int(counter['calls']>count),'sam_fallback_operator_calls':counter['calls']-count,'seconds':time.time()-t,'key':key}
     write(jp,audit)
    except Exception as exc:
     write(bank/'FAILURE.json',{'regime':regime,'image_id':iid,'roi_failed':1,'error':repr(exc)});raise
   a=read(jp);assert a['roi_success'] and a['roi_failed']==0 and sha(mp)==a['mask_sha256']
   with np.load(mp) as z:assert z['masks'].shape[1:]==(h,w)
   rows.append({'image_id':iid,'path':str(mp),'boxes':boxes,'width':w,'height':h,**{k:a[k] for k in ['mask_sha256','mask_count','roi_success','roi_failed','sam_fallback_count','sam_fallback_operator_calls']}})
   if n%25==0:
    print('ROI',regime,n+1,'/',len(meta),'mask_count',a['mask_count'],flush=True);status('roi_preflight',regime=regime,processed_images=n+1,expected_images=len(meta));write(folder/'progress.json',{'processed_images':n+1,'expected_images':len(meta),'roi_failed':0})
  assert len(rows)==len(meta)==2023
  write(folder/'index.json',rows)
  split_audit={}
  for split,rs in ms.items():
   ids={r['image_id'] for r in rs};rr=[r for r in rows if r['image_id'] in ids]
   split_audit[split]={'expected_images':SPECS[split][1],'processed_images':len(rr),'roi_success':len(rr),'roi_failed':0,'roi_nonempty':sum(r['mask_count']>0 for r in rr),'roi_empty_valid_after_filters':sum(r['mask_count']==0 for r in rr),'mask_count':sum(r['mask_count'] for r in rr),'sam_fallback_count':sum(r['sam_fallback_count'] for r in rr),'sam_fallback_operator_calls':sum(r['sam_fallback_operator_calls'] for r in rr)}
   assert len(rr)==SPECS[split][1]
  write(folder/'audit.json',{'regime':regime,'expected_images':2023,'processed_images':len(rows),'roi_success':len(rows),'roi_failed':0,'sam_fallback_count':sum(r['sam_fallback_count'] for r in rows),'sam_fallback_count_definition':'images whose SAM generation used explicit CPU linear interpolation; cached reuse shares original provenance','splits':split_audit,'index_sha256':sha(folder/'index.json')})
  print('ROI REGIME COMPLETE',regime,split_audit,flush=True)
 write(bank/'COMPLETE.json',{'roi_failed':0,'regimes':{r:{'index_sha256':sha(bank/r/'index.json'),'audit_sha256':sha(bank/r/'audit.json')} for r in REGIMES}})
 status('roi_preflight','complete')
if __name__=='__main__':main()
