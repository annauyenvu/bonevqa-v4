"""Unreleased input-conditioned GM extension, prepared after P diagnostic design.

This component is not attached to the current P factorial or any trained model.
"""
import torch
from torch import nn
from src.models.latent_prompt import LatentPromptGenerator
from src.models.layers import CrossAttnBlock,invert_mask
from src.context_controls import component_disjoint_donors


class ContextualLatentGenerator(LatentPromptGenerator):
    def __init__(self,d_model=768,n_latent=32,n_heads=8,dropout=.1,prior_labels=None):
        super().__init__(d_model,n_latent,n_heads,dropout,prior_labels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261029)
            self.context_cross=CrossAttnBlock(d_model,n_heads,dropout)
            self.context_projection=nn.Linear(d_model,d_model)
            self.context_gate=nn.Parameter(torch.tensor(-4.))

    def forward(self,context_tokens,context_mask=None,context_mode='real',context_group_ids=None):
        assert context_mode in ('real','unconditioned','shuffled')
        if context_mask is not None:assert context_mask.any(1).all()
        memory=context_tokens;mask=context_mask
        if context_mode=='unconditioned':
            memory=torch.zeros_like(memory)
            mask=None if mask is None else torch.ones_like(mask,dtype=torch.bool)
        elif context_mode=='shuffled':
            assert context_group_ids is not None and len(context_group_ids)==len(memory)
            donors=torch.tensor(component_disjoint_donors(context_group_ids),device=memory.device)
            memory=memory.index_select(0,donors)
            mask=None if mask is None else mask.index_select(0,donors)
        base,prior=super().forward(context_tokens,context_mask)
        contextual=self.context_cross(base,memory,invert_mask(mask))
        delta=torch.sigmoid(self.context_gate)*torch.tanh(self.context_projection(contextual-base))
        return base+delta,prior
