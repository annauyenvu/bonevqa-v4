"""Unreleased full-model wrapper for the later context-conditioned P study.

There is no execution protocol or quality release for this wrapper yet.
Historical models and the running P consistency/bank factorial do not import it.
"""
import torch
from src.models.bonevqa import BoneVQAModel
from src.models.latent_prompt import LatentPromptGenerator
from experiments.pcontext_component_v1 import ContextualLatentGenerator


class RoutedContextGenerator(ContextualLatentGenerator):
    def forward(self,context_tokens,context_mask=None):
        if getattr(self,'runtime_force_zero',False):
            return LatentPromptGenerator.forward(self,context_tokens,context_mask)
        return super().forward(context_tokens,context_mask,
                               context_mode=getattr(self,'runtime_mode','real'),
                               context_group_ids=getattr(self,'runtime_group_ids',None))


class PContextModel(BoneVQAModel):
    def __init__(self,cfg=None,answer_vocab=None,device=None):
        super().__init__(cfg,answer_vocab,device)
        assert self.cfg['use_latent_prompt']
        assert not any(self.cfg[x] for x in ('use_visual_prompt','use_lens','use_textual_prompt'))
        # Keep all inherited GM tensors, then add identical context-module
        # initialization without advancing shared/B RNG state.
        previous=self.latent.state_dict()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261029)
            replacement=RoutedContextGenerator(self.cfg['d_model'],self.cfg['n_latent'],self.cfg['n_heads'],self.cfg['dropout'],self.cfg['prior_labels'])
        incompatible=replacement.load_state_dict(previous,strict=False)
        assert not incompatible.unexpected_keys
        assert all(k.startswith(('context_cross.','context_projection.','context_gate')) for k in incompatible.missing_keys)
        self.latent=replacement
        train_mode=self.cfg.get('pcontext_train_mode','real');eval_mode=self.cfg.get('pcontext_eval_mode','real')
        assert train_mode in ('real','unconditioned','shuffled')
        # Shuffling is a TRAIN donor-control intervention. Public evaluation
        # uses real or zero context and requires no annotation/group metadata.
        assert eval_mode in ('real','unconditioned')

    def _encode_core(self,images,questions,regions,mask_list,prompt_kind,training,
                     include_lens_views=False,context_group_ids=None):
        self.latent.runtime_mode=self.cfg.get('pcontext_train_mode' if training else 'pcontext_eval_mode','real')
        self.latent.runtime_group_ids=context_group_ids if training else None
        self.latent.runtime_force_zero=self.cfg.get('pcontext_force_zero',False)
        try:
            return super()._encode_core(images,questions,regions,mask_list,prompt_kind,training,
                                        include_lens_views,context_group_ids)
        finally:
            self.latent.runtime_mode='real';self.latent.runtime_group_ids=None;self.latent.runtime_force_zero=False
