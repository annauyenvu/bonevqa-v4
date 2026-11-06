"""Versioned crop-token experiment; historical model sources remain sealed."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.models.bonevqa import BoneVQAModel, normalize_batch


class LocalTokens(nn.Module):
    def __init__(self,d,heads):
        super().__init__()
        self.norm=nn.LayerNorm(d)
        self.attention=nn.MultiheadAttention(d,heads,dropout=0,batch_first=True)
        self.question=nn.Linear(d,d,bias=False)
        self.gate=nn.Linear(d,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-4)
        self.mil_score=nn.Linear(d,1)
        self.aux_head=nn.Linear(d,1)

    def forward(self,global_tokens,text,text_mask,crops,valid):
        # Crops: B,K,4,D. No crop-specific diagnostic label is assigned.
        b,k,t,d=crops.shape
        active=valid.any(1)
        memory=crops.reshape(b,k*t,d)
        padding=~valid.repeat_interleave(t,dim=1)
        safe=padding.clone();safe[~active,0]=False
        q=(text*text_mask.unsqueeze(-1)).sum(1)/text_mask.sum(1,keepdim=True).clamp_min(1)
        delta=self.attention(global_tokens+self.question(q).unsqueeze(1),self.norm(memory),self.norm(memory),key_padding_mask=safe,need_weights=False)[0]
        delta=delta*torch.sigmoid(self.gate(global_tokens.mean(1))).unsqueeze(1)*active[:,None,None]
        crop_vectors=crops.mean(2)
        scores=self.mil_score(crop_vectors).squeeze(-1).masked_fill(~valid,-1e4)
        weights=scores.softmax(1)*active[:,None]
        local=(crop_vectors*weights.unsqueeze(-1)).sum(1)
        aux=self.aux_head(local).squeeze(-1)
        return global_tokens+delta,aux,active,delta


class VNextModel(BoneVQAModel):
    def __init__(self,cfg=None,answer_vocab=None,device=None):
        super().__init__(cfg,answer_vocab,device)
        assert not any(self.cfg[x] for x in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt'))
        if self.cfg.get('vnext_enabled',False):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261025)
                self.local_tokens=LocalTokens(self.cfg['d_model'],self.cfg['n_heads'])
        self._local_cache=None

    def _encode_core(self,images,questions,regions,mask_list,prompt_kind,training,include_lens_views=False,context_group_ids=None):
        core=super()._encode_core(images,questions,regions,mask_list,prompt_kind,training,include_lens_views,context_group_ids)
        if not self.cfg.get('vnext_enabled',False):return core
        crops=[];indices=[]
        for image,masks in zip(images,mask_list):
            slots=[]
            for mask in masks[:3]:
                yy,xx=np.where(np.asarray(mask)>0)
                if not len(xx):continue
                x1,x2=int(xx.min()),int(xx.max())+1;y1,y2=int(yy.min()),int(yy.max())+1
                slots.append(len(crops));crops.append(image.crop((x1,y1,x2,y2)))
            indices.append(slots)
        global_tokens=core['vis']['global_tokens'];b,_,d=global_tokens.shape
        features=global_tokens.new_zeros(b,3,4,d);valid=torch.zeros(b,3,dtype=torch.bool,device=global_tokens.device)
        if crops:
            raw=self.visual(self.visual.preprocess(crops,self.device))['tokens']
            # BiomedCLIP ViT: CLS followed by square patch grid.
            patches=raw[:,1:];side=int(patches.shape[1]**.5);assert side*side==patches.shape[1]
            pooled=F.adaptive_avg_pool2d(patches.transpose(1,2).reshape(len(crops),d,side,side),(2,2)).flatten(2).transpose(1,2)
            for i,slots in enumerate(indices):
                for j,slot in enumerate(slots):features[i,j]=pooled[slot];valid[i,j]=True
        fi,aux,active,delta=self.local_tokens(global_tokens,core['txt']['tokens'],core['txt']['mask'],features,valid)
        if self.cfg.get('vnext_force_zero',False):fi=global_tokens
        fus=self.fusion(fi,core['txt']['tokens'],None,core['txt']['mask'],None,context_group_ids=context_group_ids)
        core['vis']['f_i']=fi;core['fus']=fus
        core['logits_closed']=self.closed_head(fus['x_f']);core['type_logit']=self.type_head(fus['x_f'])
        core['fracture_logit']=self.fracture_head(fus['x_f'])
        self._local_cache=(aux,active,delta)
        return core

    def forward(self,batch,stage=3):
        output=super().forward(batch,stage)
        if not self.cfg.get('vnext_enabled',False):return output
        nb=normalize_batch(batch);aux,active,delta=self._local_cache
        indices=[i for i,(q,a) in enumerate(zip(nb['questions'],nb['answers'])) if q=='Is there a fracture in this X-ray?' and a in ('yes','no') and bool(active[i])]
        loss=aux.sum()*0
        if indices:
            target=torch.tensor([nb['answers'][i]=='yes' for i in indices],dtype=torch.float32,device=aux.device)
            loss=F.binary_cross_entropy_with_logits(aux[indices],target,pos_weight=aux.new_tensor(self.cfg['fracture_positive_weight']))
        output['loss']=output['loss']+self.cfg.get('vnext_aux_weight',0)*loss
        output['local_aux_loss']=loss.detach();output['local_delta_norm']=delta.detach().norm(dim=-1).mean()
        output['local_active_rows']=active.sum().detach()
        return output
