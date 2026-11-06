"""Measured native Vicuna linear-kernel precision diagnostic, not quality tuning."""
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.prepare_lapa_pilot_v1 import sha

OUT = ROOT / 'reports/favp_cpu_linear_benchmark_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    folder = ROOT / 'references/native_assets_favp_v1/vicuna-7b'
    index = json.loads((folder / 'pytorch_model.bin.index.json').read_text())['weight_map']
    receipt = json.loads((folder / 'receipt.json').read_text())
    keys = ['model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.down_proj.weight']
    results = []
    for key in keys:
        shard_path = folder / index[key]
        bound = next(x for x in receipt['files'] if x['name'] == shard_path.name)
        assert sha(shard_path) == bound['sha256']
        state = torch.load(str(shard_path), map_location='cpu', weights_only=True, mmap=True)
        weight = state[key]
        assert weight.dtype == torch.float16
        x = torch.randn(1, 82, weight.shape[1], dtype=torch.float32).half()
        times = {'fp16': [], 'fp32_temporary_including_cast': []}
        with torch.no_grad():
            for _ in range(3):
                tick = time.perf_counter()
                half = F.linear(x, weight)
                times['fp16'].append(time.perf_counter()-tick)
                tick = time.perf_counter()
                single = F.linear(x.float(), weight.float()).half()
                times['fp32_temporary_including_cast'].append(time.perf_counter()-tick)
        assert torch.isfinite(half).all() and torch.isfinite(single).all()
        results.append({'key': key, 'shape': list(weight.shape), 'times_seconds': times,
                        'max_abs_difference_after_half': float((half.float()-single.float()).abs().max()),
                        'mean_abs_difference_after_half': float((half.float()-single.float()).abs().mean()),
                        'exact': torch.equal(half, single)})
        del state, weight
    (OUT / 'receipt.json').write_text(json.dumps({'sequence_tokens': 82, 'threads': 8,
        'results': results, 'scope': 'two actual pretrained layer weights, deterministic random activations; CPU kernel diagnostic only, not full model speed/equivalence proof',
        'quality_released': False}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT / 'failure.json').open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
