"""Bounded native-pretrained backbone pilot routes with frozen encoders.

This is not full native paper fine-tuning: image/text encoders are frozen;
pretrained MM stack, new prompt modules and new TRAIN-vocabulary head train.
Graph409 reconstruction and optional v2 are explicitly adapted mechanisms.
"""
import torch
from torch import nn
from experiments.lapa_native_backbone_v1 import NativeBackbone
from experiments.lapa_reference_endpoint_v1 import ReferenceAnswerHead
from experiments.pf_slake_reference_v1 import SlakeReferencePF
from experiments.pf_masked_variant_v1 import configure_gat
from experiments.pf_reconstructed_v2 import GraphResidualAttention

ARMS=('carrier','gm_lf','pf_author','pf_residual','pf_residual_rewired')


class NativePilot(nn.Module):
    def __init__(self,arm):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm;self.eta=.1
        self.backbone=NativeBackbone()
        self.backbone.text.requires_grad_(False);self.backbone.visual.requires_grad_(False)
        self.vocab=self.backbone.vocab
        self.slake_pf=SlakeReferencePF(self.backbone.tokenizer)
        self.head=ReferenceAnswerHead(768,len(self.vocab))
        for module in (self.backbone.reference.prompt.PF,self.backbone.reference.prompt.graph_reduction,
                       self.backbone.reference.prompt.combined_reduction):module.requires_grad_(False)
        if arm!='gm_lf':self.backbone.reference.prompt.prompt_reduction.requires_grad_(False)
        if arm in ('carrier','gm_lf'):self.slake_pf.requires_grad_(False)
        if arm=='carrier':self.backbone.reference.prompt.requires_grad_(False);self.eta=0.
        if arm.startswith('pf_residual'):
            configure_gat(self.slake_pf.pf.GAN_layer,'masked_neighbors')
            self.slake_pf.pf.cross_attn_kw=GraphResidualAttention(self.slake_pf.pf.cross_attn_kw)

    def train(self,mode=True):
        super().train(mode);self.backbone.text.eval();self.backbone.visual.eval();return self

    def forward(self,images,questions):
        r=self.backbone(images,questions)
        if self.arm=='carrier':features=r['carrier']
        elif self.arm=='gm_lf':features=r['features']
        else:
            mode='rewired' if self.arm.endswith('rewired') else 'real'
            summary=self.slake_pf(r['lf_prompts'],self.backbone.text.get_input_embeddings(),mode)
            features=r['carrier']+summary
        return self.head(features),r['generated_mean']

    def optimizer(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
