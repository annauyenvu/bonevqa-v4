"""Author visual primitives + strict M3AE backbone/MM loading, no downloader.

Explicit adaptation: transfer pretrained type_embeddings to modality_types.
Missing original prompt/graph assets are not inferred from checkpoint weights.
"""
import ast
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize
from transformers import RobertaConfig,RobertaModel,AutoTokenizer
from experiments.lapa_interleaved_reference_v1 import InterleavedReference
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
SOURCE=ROOT/'references/lapa_author_677faf90'


def visual_primitives():
    path=SOURCE/'m3ae/modules/vision_encoders/clip_model.py'
    manifest=json.loads((ROOT/'reports/lapa_author_source_v1/source_and_fixture_audit.json').read_text())
    assert sha(path)==manifest['dependencies'][str(path)]
    names={'LayerNorm','QuickGELU','ResidualAttentionBlock','Transformer','VisualTransformer','adapt_position_encoding'}
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==names
    ns={'torch':torch,'nn':nn,'np':np,'OrderedDict':OrderedDict}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


class NativeBackbone(nn.Module):
    def __init__(self,resolution=384):
        super().__init__()
        torch.manual_seed(42)
        checkpoint=ROOT/'references/native_assets_lapa_v1/m3ae.ckpt'
        assert sha(checkpoint)=='469cf69b18cd75fbc3a010f202fb42c0cc074b34db75a0bfab442db982885442'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
        folder=ROOT/'references/native_assets_lapa_v1/roberta-base'
        self.tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True)
        cfg=RobertaConfig.from_pretrained(folder,local_files_only=True)
        self.text=RobertaModel(cfg)
        text={k[len('language_encoder.'):]:v for k,v in state.items() if k.startswith('language_encoder.')}
        legacy=text.pop('embeddings.position_ids')
        assert torch.equal(legacy,torch.arange(cfg.max_position_embeddings).unsqueeze(0))
        self.text.load_state_dict(text,strict=True)
        visual={k[len('vision_encoder.visual.'):]:v for k,v in state.items() if k.startswith('vision_encoder.visual.')}
        count=len([k for k in visual if k.endswith('attn.in_proj_weight')])
        assert count==11  # Author Transformer constructs layers-1 blocks; preserve it.
        primitives=visual_primitives()
        original_position=visual['positional_embedding'].clone()
        visual=primitives['adapt_position_encoding'](visual,patch_size=16,after=resolution,suffix='positional_embedding')
        assert torch.equal(visual['positional_embedding'][0],original_position[0])
        self.visual=primitives['VisualTransformer'](288,16,768,count+1,12,512,resolution)
        self.visual.load_state_dict(visual,strict=True)
        self.vocab=json.loads((ROOT/'reports/step4_localization_v3_1/vocab.json').read_text())
        bank=[i for a in self.vocab for i in self.tokenizer.encode(a,add_special_tokens=True)]
        self.register_buffer('bank_ids',torch.tensor(bank))
        self.reference=InterleavedReference(len(bank))
        mapping={'multi_modal_language_proj':'text_projection','multi_modal_vision_proj':'image_projection',
                 'multi_modal_language_layers':'text_layers','multi_modal_vision_layers':'image_layers',
                 'multi_modal_language_pooler':'text_pool','multi_modal_vision_pooler':'image_pool',
                 'type_embeddings':'modality_types'}
        transferred={}
        for source,dest in mapping.items():
            block={k[len(source)+1:]:v for k,v in state.items() if k.startswith(source+'.')}
            assert block,source
            target=getattr(self.reference,dest)
            target.load_state_dict(block,strict=True)
            assert all(torch.equal(v,target.state_dict()[k]) for k,v in block.items())
            transferred[source]={'destination':dest,'tensors':len(block)}
        assert all(torch.equal(v,self.text.state_dict()[k]) for k,v in text.items())
        assert all(torch.equal(v,self.visual.state_dict()[k]) for k,v in visual.items())
        self.audit={'visual_blocks':count,'visual_tensors':len(visual),'text_tensors':len(text),
                    'position_before':list(original_position.shape),'position_after':list(visual['positional_embedding'].shape),
                    'MM_transfer':transferred,'bank_tokens':len(bank),'modality_policy':'explicit pretrained tensor rename',
                    'native_prompt_weights':'new random parameters; no prompt weights in M3AE pretrain checkpoint'}
        self.preprocess=Compose([Resize(resolution,interpolation=Image.BICUBIC),CenterCrop(resolution),
            lambda image:image.convert('RGB'),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])

    def forward(self,images,questions):
        image=torch.stack([self.preprocess(i) for i in images])
        tokens=self.visual(image,None)
        text=self.tokenizer(questions,padding=True,truncation=True,max_length=32,return_tensors='pt')
        encoded=self.text(**text).last_hidden_state
        bank=self.text.get_input_embeddings()(self.bank_ids)
        result=self.reference(encoded,tokens,text['attention_mask'],bank)
        result['image_tokens_shape']=list(tokens.shape)
        result['text_tokens_shape']=list(encoded.shape)
        return result
