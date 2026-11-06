"""Source circle renderer; mask pixels are unused by this specific prompt."""
import importlib
import numpy as np
import torch
from experiments.favp_visual_bridge_v2 import SOURCE,namespace,verify_source


def render_circle(image,boxes_xywh,masks=None):
    namespace()
    for name in ['favp/models/fine_grained_visual_prompt.py','favp/models/transform.py','favp/segmed/utils/transforms.py']:
        verify_source(SOURCE/name)
    cls=importlib.import_module('favp.models.fine_grained_visual_prompt').FGVP_ENSEMBLE
    resize=importlib.import_module('favp.segmed.utils.transforms').ResizeLongestSide
    renderer=cls('red',2,'green',.5,'resize',224,resize(224),
        torch.tensor([.485,.456,.406]).view(3,1,1)*255,
        torch.tensor([.229,.224,.225]).view(3,1,1)*255,100,device='cpu')
    if boxes_xywh:
        boxes=torch.tensor(boxes_xywh,dtype=torch.float32)
        boxes[:,2:]+=boxes[:,:2]
        centers=torch.stack([boxes[:,0::2].mean(1),boxes[:,1::2].mean(1)],1)
        if masks is None:
            # Circle rendering uses box geometry only. Verify against real mask
            # arrays during cache preparation if any nonempty case is observed.
            masks=torch.zeros(len(boxes_xywh),1,1,1)
    else:boxes,centers,masks=[],[],[]
    return renderer('circle',np.asarray(image.convert('RGB')).copy(),centers,boxes,masks)
