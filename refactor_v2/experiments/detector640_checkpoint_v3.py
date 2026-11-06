"""Scoped640checkpoint loader;not a silent change to production detector runtime."""
import json
import sys
from pathlib import Path

import torch
from torchvision.ops.misc import FrozenBatchNorm2d
from scripts.train_detector_v2 import BoPhatHienDataset,tao_mo_hinh


def load_bound_640(path):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    assert payload["architecture"]=="mobilenet_320" and payload["canh_dai"]==640
    transform=payload["detector_transform"]
    assert transform=={"min_size":[640],"max_size":640,"size_divisible":32}
    model=tao_mo_hinh("mobilenet_320",weights=None).to("cpu")
    # Module classes are NOT encoded in state_dict. Restore training-time
    # FrozenBatchNorm explicitly rather than accepting scratch BatchNorm.
    spec=payload["normalization_spec"]
    assert spec and all(v["type"]=="FrozenBatchNorm2d" for v in spec.values())
    for name,value in spec.items():
        parent_name,child_name=name.rsplit(".",1)
        parent=model.get_submodule(parent_name)
        old=model.get_submodule(name)
        assert isinstance(old,torch.nn.BatchNorm2d)
        parent._modules[child_name]=FrozenBatchNorm2d(old.num_features,eps=value["eps"])
    assert {k:{"type":type(v).__name__,"eps":v.eps} for k,v in model.named_modules() if isinstance(v,(torch.nn.BatchNorm2d,FrozenBatchNorm2d))}==spec
    model.transform.min_size=tuple(transform["min_size"])
    model.transform.max_size=transform["max_size"]
    model.transform.size_divisible=transform["size_divisible"]
    model.load_state_dict(payload["state"],strict=True)
    model.eval()
    return model,payload


@torch.no_grad()
def reload_worker(checkpoint,fixture,output):
    assert not Path(output).exists()
    torch.set_num_threads(8)
    spec=json.loads(Path(fixture).read_text())
    ds=BoPhatHienDataset("train",canh_dai=640,tang_cuong=False,allowed_ids=set(spec["TRAIN_image_ids"]))
    images=[ds[i][0] for i in range(len(ds))]
    assert [iid for iid,_ in ds.muc]==sorted(spec["TRAIN_image_ids"])
    model,_=load_bound_640(checkpoint)
    transformed,_=model.transform(images,None)
    assert all(tuple(s)==(640,640) for s in transformed.image_sizes)
    torch.save({"predictions":model(images),"actual_image_sizes":transformed.image_sizes,"tensor_shape":list(transformed.tensors.shape)},output)


if __name__=="__main__":
    reload_worker(*sys.argv[1:])
