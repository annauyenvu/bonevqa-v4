"""Native assets, source-sized LoRA branches and explicit CPU compute adaptation.

Technical route is bound to census-verified zero-proposal original images only.
It is not unrestricted automatic-SAM inference or verified stage initialization.
"""
import ast
import copy
import types
import hashlib
import json
import math
import torch
from torch import nn
from experiments.favp_visual_bridge_v2 import VisualBridge, render_original, ROOT, SOURCE, verify_source
from experiments.favp_vicuna_loader_v3 import load_vicuna
from experiments.check_favp_cpu_compute_v1 import FrozenLinearFP32, adapt
from experiments.check_favp_projector_update_v1 import inputs


class SourceSizedLoRA(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.B = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.dropout = nn.Dropout(.05)
        self.scale = alpha / rank

    def forward_with_bias(self, value, bias):
        base = torch.nn.functional.linear(value, self.base.weight, bias)
        branch = self.B(self.A(self.dropout(value.float()))) * self.scale
        return base + branch.to(base.dtype)

    def forward(self, value):
        base = self.base(value)
        branch = self.B(self.A(self.dropout(value.float()))) * self.scale
        return base + branch.to(base.dtype)


def attach(model, suffixes, rank, alpha):
    names = [n for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.split('.')[-1] in suffixes]
    for name in names:
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, SourceSizedLoRA(getattr(parent,leaf),rank,alpha))
    return names


def patch_eva_attention(visual):
    # Pinned attention bypasses qkv.forward via F.linear(...qkv.weight...).
    # Route that operation through a bias-aware low-rank wrapper explicitly.
    path = SOURCE / 'favp/models/eva_vit.py'
    verify_source(path)
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Attention')
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
    replacements = 0
    for node in ast.walk(fn):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='qkv' for t in node.targets):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='linear':
                node.value = ast.parse('self.qkv.forward_with_bias(x, qkv_bias)',mode='eval').body
                replacements += 1
    assert replacements==1
    count=0
    for block in visual.blocks:
        namespace=dict(block.attn.forward.__globals__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path),'exec'),namespace)
        block.attn.forward=types.MethodType(namespace['forward'],block.attn)
        count+=1
    assert count==39


class TrainableRoute(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisualBridge().requires_grad_(False)
        # Source EVA base stays FP32 on CPU; only low-rank parameters train.
        eva_names = attach(self.visual.visual, {'qkv'},4,8)
        assert len(eva_names)==39
        patch_eva_attention(self.visual.visual)
        self.visual.visual.use_checkpoint = True
        self.visual.qformer.requires_grad_(True)
        self.visual.query_tokens.requires_grad_(True)
        self.language, self.tokenizer, self.language_audit = load_vicuna()
        frozen_linears = adapt(self.language)
        lm_names = attach(self.language, {'q_proj','v_proj'},8,16)
        assert len(lm_names)==64 and frozen_linears==225
        self.projection = nn.Linear(768,4096).float()
        census = ROOT / 'reports/favp_train_proposal_census_v1'
        receipt = json.loads((census/'receipt.json').read_text())
        assert receipt['complete'] and receipt['images']==256 and receipt['nonempty']==0
        self.allowed = {json.loads(p.read_text())['rgb_sha256'] for p in census.glob('IMG*.json')}
        assert 0 < len(self.allowed) <= 256
        self.census_raster_count = len(self.allowed)
        self.audit = {'EVA_lora_modules':len(eva_names),'LM_lora_modules':len(lm_names),
            'EVA_qkv_route':'verified runtime AST correction: bias-aware LoRA forward replaces direct weight-only F.linear',
            'EVA_rank_alpha_dropout':[4,8,.05],'LM_rank_alpha_dropout':[8,16,.05],
            'Qformer_query_trainable':True,'projection':'new random; unverified stage handoff',
            'CPU_compute':'frozen LM base FP16 stored, temporary FP32 linear forward/backward; ordinary FP32 trainable LoRA branches',
            'SAM_policy':'cached complete TRAIN census verifies zero accepted proposals; reject uncensused raster',
            'census_unique_rasters':self.census_raster_count,'quality_released':False}

    def encode(self,image):
        import numpy as np
        raster = np.asarray(image.convert('RGB')).copy()
        assert hashlib.sha256(raster.tobytes()).hexdigest() in self.allowed, 'Input has no verified automatic-proposal route'
        views = render_original(image)
        # Source EVA uses reentrant checkpointing: a grad-bearing input prevents
        # the first checkpoint from dropping gradients to its internal LoRA.
        if self.training:
            views.requires_grad_(True)
        query, _ = self.visual(views)
        return query

    def forward(self,image,question,answer):
        # answer is present only in the training-loss API, not public generation.
        query = self.encode(image)
        return self.language(**inputs(self.language,self.tokenizer,self.projection,query,question,answer))

    def train(self,mode=True):
        super().train(mode)
        self.visual.ln_vision.eval()
        if mode:
            self.language.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:
            self.language.gradient_checkpointing_disable()
        return self

    def groups(self):
        names = {'EVA_LoRA':[], 'QFormer':[], 'LM_LoRA':[], 'projection':[]}
        for name,p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('visual.visual.'):
                names['EVA_LoRA'].append((name,p))
            elif name.startswith('visual.'):
                names['QFormer'].append((name,p))
            elif name.startswith('language.'):
                names['LM_LoRA'].append((name,p))
            elif name.startswith('projection.'):
                names['projection'].append((name,p))
            else:
                raise AssertionError(name)
        assert all(names.values())
        return names


