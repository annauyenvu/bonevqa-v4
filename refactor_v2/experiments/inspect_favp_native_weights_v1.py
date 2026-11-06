"""Restricted metadata-only inventory of author-linked FAVP assets."""
import collections
import json
from pathlib import Path
import torch
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
OUT=ROOT/'reports/favp_native_weight_inventory_v1'
ASSETS=[('QFormer','blip2_pretrained_flant5xxl.pth','4b3839ea6c617f315ead9bf4036bbb0f0cf6bf62695ecfc14968ea626af03a29'),
        ('EVA','eva_vit_g.pth','99d2bb36c6b52c94fe6e2e12373afb27de57ae81378c3d8c53bf0e83b0f4275f'),
        ('SAMMed2D','sam-med2d_b.pth','921d8c130cc4bcc4b7220c064e60f207f15ad59f17ab36de340e41c67a490d18')]


def main():
    OUT.mkdir(exist_ok=False);results=[]
    for label,name,expected in ASSETS:
        path=ROOT/'references/native_assets_favp_v1'/name
        actual=sha(path);assert actual==expected
        r={'asset':label,'path':str(path),'sha256':actual,'bytes':path.stat().st_size}
        try:
            obj=torch.load(path,map_location='meta',weights_only=True)
            assert isinstance(obj,dict)
            r['top_keys']=[str(k) for k in obj]
            state=obj
            for key in ('state_dict','model'):
                if key in obj and isinstance(obj[key],dict):state=obj[key];r['state_container']=key;break
            tensors={k:{'shape':list(v.shape),'dtype':str(v.dtype),'numel':v.numel()} for k,v in state.items() if isinstance(v,torch.Tensor)}
            assert tensors
            r.update(restricted_load_succeeded=True,tensors=tensors,tensor_count=len(tensors),
                     prefixes=dict(collections.Counter(k.split('.')[0] for k in tensors)))
        except Exception as e:r.update(restricted_load_succeeded=False,error=repr(e),unsafe_fallback_used=False)
        results.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('tensors','top_keys')}),flush=True)
    (OUT/'inventory.json').write_text(json.dumps({'assets':results,'source_sha256':sha(Path(__file__)),'quality_released':False},indent=2))


if __name__=='__main__':main()
