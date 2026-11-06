"""Restricted metadata acceptance of model tensors inside SAM training archive."""
import json
from pathlib import Path
import torch
from experiments.favp_sam_restricted_loader_v1 import load_archive
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]/'refactor_v2'
OUT=ROOT/'reports/favp_sam_archive_v2'


def main():
    OUT.mkdir(exist_ok=False)
    path=ROOT/'references/native_assets_favp_v1/sam-med2d_b.pth'
    archive=load_archive(path)
    assert isinstance(archive,dict) and isinstance(archive['model'],dict)
    tensors={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in archive['model'].items() if isinstance(v,torch.Tensor)}
    result={'top_keys':list(archive),'model_tensors':len(tensors),'tensors':tensors,'sha256':sha(path),
            'loader':'explicit global allowlist; optimizer substituted with inert local object; no optimizer methods execute',
            'quality_released':False,'sources':{str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('favp_sam_restricted_loader_v1.py')]}}
    (OUT/'inventory.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='tensors'}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
