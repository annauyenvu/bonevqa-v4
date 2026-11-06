"""Restricted metadata-only loading: no arbitrary pickle fallback or model execution."""
import collections
import hashlib
import json
import pickletools
import zipfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
OUT=ROOT/'reports/lapa_native_weight_inventory_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def inspect(path):
    result={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('data.pkl')]
            assert len(names)==1
            info=z.getinfo(names[0]);assert info.file_size<20*1024**2
            globals_=[]
            for op,arg,pos in pickletools.genops(z.read(names[0])):
                if op.name=='GLOBAL':globals_.append(arg)
            result['pickle_global_names']=sorted(set(globals_))
    try:
        obj=torch.load(path,map_location='meta',weights_only=True)
    except Exception as e:
        result.update(restricted_load_succeeded=False,error=repr(e),unsafe_fallback_used=False)
        return result
    assert isinstance(obj,dict)
    result['top_level_keys']=[str(k) for k in obj]
    state=obj.get('state_dict',obj)
    assert isinstance(state,dict)
    tensors={str(k):{'shape':list(v.shape),'dtype':str(v.dtype),'numel':v.numel()} for k,v in state.items() if isinstance(v,torch.Tensor)}
    result.update(restricted_load_succeeded=True,tensors=tensors,tensor_count=len(tensors),
                  total_tensor_elements=sum(v['numel'] for v in tensors.values()),
                  key_prefixes=dict(collections.Counter(k.split('.')[0] for k in tensors)))
    # Keep only literal primitive checkpoint configuration; do not execute objects.
    cfg=obj.get('hyper_parameters',{})
    if isinstance(cfg,dict):
        cfg=cfg.get('config',cfg)
        if isinstance(cfg,dict):result['primitive_config']={str(k):v for k,v in cfg.items() if v is None or type(v) in (str,int,float,bool)}
    return result


def main():
    OUT.mkdir(exist_ok=False)
    paths=[ROOT/'references/native_assets_lapa_v1/m3ae.ckpt',ROOT/'references/native_assets_lapa_v1/roberta-base/pytorch_model.bin']
    expected=['469cf69b18cd75fbc3a010f202fb42c0cc074b34db75a0bfab442db982885442','278b7a95739c4392fae9b818bb5343dde20be1b89318f37a6d939e1e1b9e461b']
    results=[]
    for path,h in zip(paths,expected):
        r=inspect(path);assert r['sha256']==h;results.append(r)
    (OUT/'inventory.json').write_text(json.dumps({'assets':results,'source_sha256':sha(Path(__file__)),'quality_released':False},indent=2))
    for r in results:print(json.dumps({k:v for k,v in r.items() if k not in ('tensors','primitive_config')}))


if __name__=='__main__':main()
