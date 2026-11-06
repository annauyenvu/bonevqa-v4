from common import *
import torch
setup(42)
for req in [False,True]:
 for shape in [(2,48,768),(2,784,768)]:
  torch.manual_seed(42);x=torch.randn(shape,device='mps',requires_grad=req);m=torch.nn.LayerNorm(768).to('mps');(m(x).square().mean()).backward()
  print('MPS input_grad',req,'shape',shape,'weight_finite',bool(torch.isfinite(m.weight.grad).all()),'bias_finite',bool(torch.isfinite(m.bias.grad).all()),'weight_max',float(m.weight.grad.abs().max()),flush=True)
import types
from engine import safe_layernorm_forward
x=torch.randn((2,48,768),device='mps');m=torch.nn.LayerNorm(768).to('mps');m.forward=types.MethodType(safe_layernorm_forward,m);m(x).square().mean().backward()
c=torch.nn.LayerNorm(768);c(x.cpu()).square().mean().backward()
assert torch.isfinite(m.weight.grad).all() and torch.isfinite(m.bias.grad).all()
assert torch.allclose(m.weight.grad.cpu(),c.weight.grad,rtol=1e-3,atol=1e-6)
assert torch.allclose(m.bias.grad.cpu(),c.bias.grad,rtol=1e-3,atol=1e-6)
print('WORKAROUND PASS, affine gradients agree with CPU',flush=True)
