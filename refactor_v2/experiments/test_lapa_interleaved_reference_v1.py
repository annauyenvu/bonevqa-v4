"""Bounded toy-interface test of actual author cross-layer/LF ordering."""
import hashlib
import json
from pathlib import Path

import torch
from experiments.lapa_interleaved_reference_v1 import InterleavedReference


def main():
    torch.set_num_threads(1); torch.manual_seed(20261027)
    model = InterleavedReference(41, d_model=32, mm_heads=8, graph_nodes=5).eval()
    text = torch.randn(2, 5, 32, requires_grad=True)
    image = torch.randn(2, 7, 32, requires_grad=True)
    mask = torch.tensor([[1,1,1,0,0], [1,1,1,1,1]], dtype=torch.bool)
    bank = torch.randn(41, 32, requires_grad=True)
    out = model(text, image, mask, bank)
    assert out['trace'] == [name for i in range(1,7) for name in (f'MM{i}',f'LF{i}')]
    baseline = model(text, image, mask, bank, include_prompt=False)
    assert torch.equal(out['carrier'], baseline['features'])
    out['features'].square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0 for x in (text,image,bank))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for name,p in model.named_parameters() if name.startswith(('text_layers.','image_layers.','prompt.LF.')))
    path = Path(__file__).resolve().parents[2] / 'refactor_v2/reports/lapa_interleaved_reference_v1'
    path.mkdir(exist_ok=False)
    result = {'scope':'Toy32dim synthetic tokens; exact author cross-modal/LF primitives, not backbone or VQA gate.',
              'trace':out['trace'], 'text_image_bank_gradients_finite_nonzero':True,
              'all_MM_LF_parameter_gradients_finite':True, 'compute_then_zero_control_carrier_exact_eval':True,
              'native_backbone_verified':False,'PF_integrated':False,'quality_released':False,
              'dependencies':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('lapa_interleaved_reference_v1.py')]}}
    with (path/'interface_fixture.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
