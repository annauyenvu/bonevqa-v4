"""Full native frozen LM diagnostic with temporary FP32 linear compute.

Stored weights stay FP16. This is an explicit numerical adaptation, not bitwise
equivalence to CPU FP16, nor an accepted quality-training configuration.
"""
import json
from pathlib import Path
import resource
import time
import types
import torch
import torch.nn.functional as F
from experiments.favp_vicuna_loader_v3 import load_vicuna
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.check_favp_projector_update_v1 import inputs
from experiments.lapa_integrated_gate_v1 import batch_data
from experiments.prepare_lapa_pilot_v1 import sha

OUT = ROOT / 'reports/favp_cpu_compute_v1'


class FrozenLinearFP32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias):
        ctx.save_for_backward(weight)
        ctx.input_dtype = value.dtype
        return F.linear(value.float(), weight.float(), None if bias is None else bias.float()).to(value.dtype)

    @staticmethod
    def backward(ctx, grad):
        weight, = ctx.saved_tensors
        return F.linear(grad.float(), weight.t().float()).to(ctx.input_dtype), None, None


def adapt(model):
    count = 0
    def forward(module, value):
        assert not module.weight.requires_grad
        assert module.bias is None or not module.bias.requires_grad
        return FrozenLinearFP32.apply(value, module.weight, module.bias)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(forward, module)
            count += 1
    return count


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    start = time.perf_counter()
    prior = ROOT / 'reports/favp_projector_update_v1/projector.pt'
    state = torch.load(prior, map_location='cpu', weights_only=True)
    feature = ROOT / 'reports/favp_visual_bridge_v2/features.pt'
    assert sha(feature) == state['feature_sha256']
    saved = torch.load(feature, map_location='cpu', weights_only=True)
    batch, answers = batch_data()
    model, tokenizer, audit = load_vicuna()
    count = adapt(model)
    projection = torch.nn.Linear(768, 4096).float()
    projection.load_state_dict(state['projection'], strict=True)
    model.eval()
    tick = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs(model, tokenizer, projection, saved['queries'], batch['questions'][0], answers[0]))
        repeat = model(**inputs(model, tokenizer, projection, saved['queries'], batch['questions'][0], answers[0]))
    seconds = time.perf_counter()-tick
    assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
    assert torch.equal(output.logits, repeat.logits) and torch.equal(output.loss, repeat.loss)
    delta = output.logits.float()-state['logits'].float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    tick = time.perf_counter()
    result = model(**inputs(model, tokenizer, projection, saved['queries'], batch['questions'][0], answers[0]))
    result.loss.backward()
    backward_seconds = time.perf_counter()-tick
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in projection.parameters())
    assert all(p.grad is None for p in model.parameters())
    receipt = {'linear_modules': count, 'two_forward_seconds': seconds,
        'forward_backward_seconds': backward_seconds, 'loss': float(output.loss), 'fp16_reference_loss': float(state['loss']),
        'max_logit_difference': float(delta.abs().max()), 'mean_logit_difference': float(delta.abs().mean()),
        'all_position_argmax_agreement': float((output.logits.argmax(-1)==state['logits'].argmax(-1)).float().mean()),
        'repeat_exact': True, 'gradient_norms': {k:float(p.grad.norm()) for k,p in projection.named_parameters()},
        'peak_rss_raw': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'total_seconds': time.perf_counter()-start,
        'reference_checkpoint_sha256': sha(prior), 'scope': 'actual same TRAIN image/query/projector; FP16 stored weights and activations, temporary FP32 linear forward/backward; diagnostic not bitwise equivalence or quality release',
        'quality_released': False, 'audit': audit}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT / 'failure.json').open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
