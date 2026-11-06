"""Small invariants for paired hierarchical resampling and seed-controlled augmentation."""
from common import *
import numpy as np
from analyze import hierarchical
from src.data.prompts import build_lens
from PIL import Image
n=np.zeros((5,3,2,4));n[:,:,0,:]=1;den=np.ones(4)
b=hierarchical(n,den,reps=200);assert np.array_equal(b[:,:,0]-b[:,:,1],np.full((200,3),100.))
n[:,:,:,:]=np.arange(4)[None,None,None,:];den=np.array([1,2,3,4.]);b=hierarchical(n,den,reps=200);assert np.array_equal(b[:,:,0],b[:,:,1]);assert np.array_equal(b[:,0],b[:,2])
a=np.ones((1,12,12),dtype=np.uint8);im=Image.new('RGB',(12,12),(130,130,130))
np.random.seed(42);p=np.asarray(build_lens(im,a));np.random.seed(42);q=np.asarray(build_lens(im,a));assert np.array_equal(p,q)
np.random.seed(123);p=np.asarray(build_lens(im,a,alpha=.5));np.random.seed(456);q=np.asarray(build_lens(im,a,alpha=.5));assert np.array_equal(p,q)
print('PASS: hierarchical pairing, constant known effect, shared regime draws, seeded augmentation, deterministic eval alpha')
