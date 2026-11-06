"""TRAIN-only weak spatial supervision; public prediction keeps the original path."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.models.bonevqa import BoneVQAModel,normalize_batch


def spatial_targets(size,boxes,mode='real',resolution=224,patch=16):
    """Map canonical continuous box edges to the exact rounded letterbox geometry.

    Raster membership uses pixel centers. Targets are fractional union-box
    coverage within each valid image patch, not segmentation diagnoses.
    Placebo translates the union cyclically inside the resized image rectangle;
    the pixel area and all valid-image weights remain identical.
    """
    assert mode in ('real','placebo','uniform','off')
    w,h=size;scale=min(resolution/w,resolution/h)
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    x=(np.arange(nw)+.5)*w/nw;y=(np.arange(nh)+.5)*h/nh
    union=np.zeros((nh,nw),dtype=bool)
    for box in boxes or []:
        x1,y1,x2,y2=map(float,box[:4]);assert 0<=x1<x2<=w and 0<=y1<y2<=h
        union|=((y[:,None]>=y1)&(y[:,None]<y2)&(x[None,:]>=x1)&(x[None,:]<x2))
    original_area=int(union.sum())
    if mode=='placebo':union=np.roll(union,(max(1,nh//2),max(1,nw//2)),axis=(0,1))
    assert int(union.sum())==original_area
    mask=union.astype(np.float32)
    if mode=='uniform':mask.fill(original_area/(nw*nh))
    if mode=='off':mask.fill(0)
    px=(resolution-nw)//2;py=(resolution-nh)//2
    target=np.zeros((resolution,resolution),dtype=np.float32);valid=np.zeros_like(target)
    target[py:py+nh,px:px+nw]=mask;valid[py:py+nh,px:px+nw]=1
    n=resolution//patch;assert n*patch==resolution
    def pool(v):return v.reshape(n,patch,n,patch).mean(axis=(1,3)).reshape(-1)
    mass=pool(target);weight=pool(valid);occupancy=mass/np.maximum(weight,1e-12)
    return occupancy,weight,{'image_pixels':nw*nh,'positive_pixels':original_area,'resized':[nw,nh],'padding':[px,py]}


class SpatialSupervisedModel(BoneVQAModel):
    def __init__(self,cfg=None,answer_vocab=None,device=None):
        super().__init__(cfg,answer_vocab,device)
        assert not any(self.cfg[x] for x in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt'))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261026)
            self.spatial_head=nn.Sequential(nn.LayerNorm(self.cfg['d_model']),nn.Linear(self.cfg['d_model'],1))
        self._spatial_logits=None

    def _encode_core(self,*args,**kwargs):
        core=super()._encode_core(*args,**kwargs)
        tokens=core['vis']['global_tokens'];assert tokens.shape[1]==197
        self._spatial_logits=self.spatial_head(tokens[:,1:]).squeeze(-1)
        return core

    def forward(self,batch,stage=3):
        output=super().forward(batch,stage);nb=normalize_batch(batch)
        indices=[i for i,(q,a) in enumerate(zip(nb['questions'],nb['answers'])) if q=='Is there a fracture in this X-ray?' and a in ('yes','no')]
        loss=self._spatial_logits.sum()*0
        if indices and self.cfg.get('lnext_spatial_weight',0)>0:
            target=torch.as_tensor(np.asarray(batch['spatial_targets']),device=self.device,dtype=torch.float32)[indices]
            weight=torch.as_tensor(np.asarray(batch['spatial_weights']),device=self.device,dtype=torch.float32)[indices]
            assert target.shape==weight.shape==self._spatial_logits[indices].shape
            assert ((target>=0)&(target<=1)).all() and ((weight>=0)&(weight<=1)).all()
            element=F.binary_cross_entropy_with_logits(self._spatial_logits[indices],target,reduction='none',pos_weight=target.new_tensor(self.cfg['lnext_positive_weight']))
            loss=((element*weight).sum(1)/weight.sum(1).clamp_min(1)).mean()
        output['loss']=output['loss']+self.cfg.get('lnext_spatial_weight',0)*loss
        output['spatial_loss']=loss
        output['spatial_canonical_rows']=len(indices)
        return output
