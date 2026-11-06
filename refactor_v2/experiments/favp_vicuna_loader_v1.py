"""Pinned Vicuna V0 with restricted mmap loading and explicit legacy RoPE binding."""
import importlib.util
import json
from pathlib import Path
import torch
from transformers import LlamaConfig,LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from experiments.favp_visual_bridge_v2 import ROOT,SOURCE,verify_source
from experiments.prepare_lapa_pilot_v1 import sha


def load_vicuna():
    folder=ROOT/'references/native_assets_favp_v1/vicuna-7b'
    receipt=json.loads((folder/'receipt.json').read_text());assert receipt['complete']
    for file in receipt['files']:assert sha(folder/file['name'])==file['sha256']
    path=SOURCE/'favp/models/modeling_llama.py';verify_source(path)
    spec=importlib.util.spec_from_file_location('favp_author_llama_v1',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=LlamaConfig.from_pretrained(folder,local_files_only=True);cfg._attn_implementation='eager'
    with torch.device('meta'):model=module.LlamaForCausalLM(cfg)
    index=json.loads((folder/'pytorch_model.bin.index.json').read_text())['weight_map']
    state={}
    for name in sorted(set(index.values())):
        shard=torch.load(folder/name,map_location='cpu',weights_only=True,mmap=True)
        assert not (state.keys() & shard.keys());state.update(shard);del shard
    assert set(state)==set(index)
    legacy={k:state.pop(k) for k in list(state) if k.endswith('.rotary_emb.inv_freq')}
    assert len(legacy)==32
    model.load_state_dict(state,strict=True,assign=True)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    for i,layer in enumerate(model.model.layers):
        old=legacy[f'model.layers.{i}.self_attn.rotary_emb.inv_freq']
        rotary=LlamaRotaryEmbedding(cfg.hidden_size//cfg.num_attention_heads,
                    max_position_embeddings=cfg.max_position_embeddings,base=cfg.rope_theta,device='cpu')
        assert torch.allclose(old.float(),rotary.inv_freq,rtol=1e-3,atol=5e-6)
        rotary.inv_freq=old
        layer.self_attn.rotary_emb=rotary
    assert all(p.device.type=='cpu' for p in model.parameters())
    assert all(b.device.type=='cpu' for b in model.buffers())
    model.requires_grad_(False);model.eval()
    tokenizer=LlamaTokenizer.from_pretrained(folder,use_fast=False,local_files_only=True)
    tokenizer.pad_token='$$'
    assert tokenizer.pad_token_id>=0 and tokenizer.pad_token_id<cfg.vocab_size
    return model,tokenizer,{'weight_tensors':len(state),'legacy_rope_vectors':32,'strict_state':True,
            'dtype':str(next(model.parameters()).dtype),'pad_token_id':tokenizer.pad_token_id,
            'checkpoint_revision':receipt['revision'],'runtime':'CPU float16/eager attention; author LM forward class on current Transformers',
            'loading':'restricted weights_only mmap; legacy RoPE bound explicitly; all runtime buffers materialized CPU'}
