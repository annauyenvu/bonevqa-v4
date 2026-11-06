"""Prospectively fixed adapted route with public-image automatic cache and AdamW."""
import hashlib
import json
import math
import numpy as np
import torch
from experiments.favp_trainable_route_v3 import TrainableRoute,ROOT
from experiments.favp_circle_cache_v1 import render_circle
from experiments.favp_generation_compat_v1 import enable_generation_cache_position
from experiments.prepare_lapa_pilot_v1 import sha
from src.data import build_dataset,collate_fn
from src.train import build_split
from src.utils import normalize_answer

INPUTS=ROOT/'reports/favp_pilot_inputs_v1'


class PilotRoute(TrainableRoute):
    def __init__(self):
        super().__init__()
        receipt=json.loads((INPUTS/'receipt.json').read_text())
        assert receipt['complete'] and receipt['model_val']==8
        path=INPUTS/'public_validation_cache.json'
        assert sha(path)==receipt['public_cache_sha256']
        rows=json.loads(path.read_text())
        assert len(rows)==8
        self.public_cache={r['rgb_sha256']:r for r in rows}
        enable_generation_cache_position(self.language)
        self.audit['public_cache']='frozen TRAIN census plus prospectively selected 8 model-validation automatic-SAM caches; no annotation ROI'
        self.audit['generation_compatibility']='cache_position forwarded; tokenizer semantic BOS/EOS/PAD explicit'

    def encode(self,image):
        raster=np.asarray(image.convert('RGB')).copy()
        key=hashlib.sha256(raster.tobytes()).hexdigest()
        if key in self.public_cache:
            views=render_circle(image,self.public_cache[key]['boxes_xywh'])
            if self.training:views.requires_grad_(True)
            query,_=self.visual(views)
            return query
        return super().encode(image)


def optimizer_for(model):
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5,
        weight_decay=.05,betas=(.9,.999),eps=1e-8)
    def factor(step):
        if step<6:
            lr=1e-6+(3e-5-1e-6)*step/6
        else:
            progress=min(1.,max(0.,(step-6)/58))
            lr=1e-5+.5*(3e-5-1e-5)*(1+math.cos(math.pi*progress))
        return lr/3e-5
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def pilot_batch():
    selection=json.loads((INPUTS/'selection_and_schedules.json').read_text())
    cfg=json.loads((ROOT/'reports/step4_localization_v3_1/B.json').read_text())
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    records=[ds[r['record_index']] for r in selection['subset']['train'][:2]]
    return collate_fn(records),[normalize_answer(r['answer']) for r in records]
