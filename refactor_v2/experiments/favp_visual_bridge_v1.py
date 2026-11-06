"""Pinned EVA/QFormer inference bridge without downloads or stage assumptions."""
import ast
import importlib
import json
import sys
import types
from functools import partial
from pathlib import Path
import torch
from torch import nn
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
SOURCE=ROOT/'references/favp_author_b43941ba'


def verify_source(path):
    manifest=json.loads((ROOT/'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['FAVP']['sha256']
    assert sha(path)==manifest[str(path)]


def namespace():
    for name,path in [('favp',SOURCE/'favp'),('favp.models',SOURCE/'favp/models'),('favp.segmed',SOURCE/'favp/segmed')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module


def eva_class():
    path=SOURCE/'favp/models/eva_vit.py';verify_source(path)
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    # Meta construction must not evaluate .item() on a meta linspace. In this
    # eval bridge drop_path_rate=0, so a literal zero list is exactly equivalent.
    for cls in nodes:
        if cls.name=='VisionTransformer':
            init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
            for i,n in enumerate(init.body):
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dpr' for t in n.targets):
                    init.body[i]=ast.parse('dpr = [0.0] * depth').body[0]
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) and
             not (isinstance(n,ast.ImportFrom) and n.module=='favp.common.dist_utils')]
    ns={'__name__':'favp_eva_primitives_v1'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=imports+nodes,type_ignores=[])),str(path),'exec'),ns)
    return ns['VisionTransformer']


def qformer_class():
    path=SOURCE/'favp/models/Qformer.py';verify_source(path)
    tree=ast.parse(path.read_text());moved={'apply_chunking_to_forward','find_pruneable_heads_and_indices','prune_linear_layer'}
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module=='transformers.modeling_utils':
            node.names=[n for n in node.names if n.name not in moved]
    tree.body.insert(0,ast.ImportFrom(module='transformers.pytorch_utils',names=[ast.alias(name=n) for n in sorted(moved)],level=0))
    name='favp_qformer_runtime_v1';module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    return module.BertConfig,module.BertLMHeadModel


class VisualBridge(nn.Module):
    def __init__(self):
        super().__init__();namespace()
        assets=ROOT/'references/native_assets_favp_v1'
        eva_path=assets/'eva_vit_g.pth';q_path=assets/'blip2_pretrained_flant5xxl.pth'
        assert sha(eva_path)=='99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f'
        assert sha(q_path)=='4b3839ea6c617f315ead9bf4036bbb0f0cf6bf62695ecfc14968ea626af03a29'
        visual_class=eva_class()
        with torch.device('meta'):
            self.visual=visual_class(img_size=224,patch_size=14,use_mean_pooling=False,embed_dim=1408,
                depth=39,num_heads=16,mlp_ratio=4.3637,qkv_bias=True,drop_path_rate=0.,norm_layer=partial(nn.LayerNorm,eps=1e-6))
        weights=torch.load(eva_path,map_location='cpu',weights_only=True)
        omitted={k for k in weights if k not in self.visual.state_dict()}
        assert omitted=={'norm.weight','norm.bias','head.weight','head.bias'},omitted
        filtered={k:v for k,v in weights.items() if k not in omitted}
        self.visual.load_state_dict(filtered,strict=True,assign=True)
        assert all(torch.equal(v,self.visual.state_dict()[k]) for k,v in filtered.items())
        self.visual.float();del weights,filtered
        config_cls,q_cls=qformer_class()
        cfg=config_cls(vocab_size=30522,hidden_size=768,num_hidden_layers=12,num_attention_heads=12,intermediate_size=3072)
        cfg.encoder_width=1408;cfg.add_cross_attention=True;cfg.cross_attention_freq=2;cfg.query_length=32
        self.qformer=q_cls(cfg)
        self.qformer.cls=None;self.qformer.bert.embeddings.word_embeddings=None;self.qformer.bert.embeddings.position_embeddings=None
        for layer in self.qformer.bert.encoder.layer:layer.output=None;layer.intermediate=None
        state=torch.load(q_path,map_location='cpu',weights_only=True)['model']
        qs={k[len('Qformer.'):]:v for k,v in state.items() if k.startswith('Qformer.')}
        self.qformer.load_state_dict(qs,strict=True)
        assert all(torch.equal(v,self.qformer.state_dict()[k]) for k,v in qs.items())
        self.query_tokens=nn.Parameter(state['query_tokens'].clone())
        self.ln_vision=nn.LayerNorm(1408)
        self.ln_vision.load_state_dict({k[len('ln_vision.'):]:v for k,v in state.items() if k.startswith('ln_vision.')},strict=True)
        self.audit={'EVA_loaded':len(self.visual.state_dict()),'EVA_unused_classifier_norm':sorted(omitted),
                    'Qformer_loaded':len(qs),'query_tokens':'pretrained','vision_norm':'pretrained',
                    'projection':'t5_proj unused; no verified FAVP-to-Vicuna stage projection',
                    'adaptations':['CPUfloat32 for EVA','meta constructor zero stochastic-depth list at eval drop_path0',
                                   'Transformers helper imports moved to pytorch_utils','base EVA inference; source visual LoRA not yet trained/attached']}

    def forward(self,views):
        # Native order: renderer original view0; local views1..N precede global
        # tokens in the QFormer memory. A zero-proposal case has original only.
        encoded=[self.ln_vision(self.visual(view.unsqueeze(0))) for view in views]
        memory=torch.cat(encoded[1:]+encoded[:1],dim=1) if len(encoded)>1 else encoded[0]
        mask=torch.ones(memory.shape[:2],dtype=torch.long)
        result=self.qformer.bert(query_embeds=self.query_tokens,encoder_hidden_states=memory,
                                 encoder_attention_mask=mask,return_dict=True).last_hidden_state
        return result,memory


def render_original(image):
    namespace()
    for relative in ['favp/models/fine_grained_visual_prompt.py','favp/models/transform.py','favp/segmed/utils/transforms.py']:
        verify_source(SOURCE/relative)
    renderer=importlib.import_module('favp.models.fine_grained_visual_prompt').FGVP_ENSEMBLE
    resize=importlib.import_module('favp.segmed.utils.transforms').ResizeLongestSide
    obj=renderer('red',2,'green',.5,'resize',224,resize(224),
        torch.tensor([.485,.456,.406]).view(3,1,1)*255,
        torch.tensor([.229,.224,.225]).view(3,1,1)*255,100,device='cpu')
    import numpy as np
    return obj('circle',np.asarray(image.convert('RGB')).copy(),[],[],[])
