"""Source-RAD optimizer groups with a prospectively bounded 292-update pilot."""
import torch
from transformers import get_polynomial_decay_schedule_with_warmup
from experiments.lapa_native_pilot_v1 import NativePilot,ARMS


class SourceOptimizerPilot(NativePilot):
    def optimizer(self):
        groups={}
        mm=('backbone.reference.text_projection.','backbone.reference.image_projection.',
            'backbone.reference.text_layers.','backbone.reference.image_layers.',
            'backbone.reference.text_pool.','backbone.reference.image_pool.')
        no_decay=('bias','LayerNorm.bias','LayerNorm.weight','norm.bias','norm.weight',
                  'norm1.bias','norm1.weight','norm2.bias','norm2.weight')
        for name,param in self.named_parameters():
            if not param.requires_grad:continue
            kind='head' if name.startswith('head.') else ('MM' if name.startswith(mm) else 'base')
            decay=0. if any(n in name for n in no_decay) else .01
            groups.setdefault((kind,decay),[]).append(param)
        multipliers={'base':1,'MM':5,'head':50}
        return torch.optim.AdamW([{'params':params,'lr':5e-6*multipliers[kind],
              'weight_decay':decay,'group_name':kind} for (kind,decay),params in groups.items()],
              lr=5e-6,eps=1e-8,betas=(.9,.98))

    @staticmethod
    def scheduler(optimizer):
        return get_polynomial_decay_schedule_with_warmup(optimizer,num_warmup_steps=29,
              num_training_steps=292,lr_end=0.,power=1.)
