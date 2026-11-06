"""Independent fixed-budget/source and disposable-update check for head-only pilot."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
OUT = ROOT/'refactor_v2/reports/frozen_fracture_head_pilot_v3_1'
CACHE = ROOT/'refactor_v2/reports/frozen_head_TRAIN_cache_v3_1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import torch
    import torch.nn.functional as F
    from src.models.heads import FracturePresenceHead
    torch.set_num_threads(8)
    assert not (OUT/'execution_release.json').exists()
    p = read(OUT/'protocol.json')
    for path, digest in p['dependencies_sha256'].items():
        assert sha(path) == digest
    assert (p['epochs'], p['batch_size'], p['total_updates'], p['seed']) == (10, 64, 410, 42)
    assert p['lr'] == .0001 and p['betas'] == [.9, .98] and p['weight_decay'] == .01
    assert p['clip'] == 1 and p['pos_weight'] == 2110/458 and p['scheduler'] is None
    audit = read(CACHE/'independent_artifact_audit.json')
    assert audit['passed'] and audit['cache_sha256'] == sha(CACHE/'TRAIN_features.pt')
    tree = ast.parse((ROOT/'refactor_v2/experiments/fit_frozen_fracture_head_v3_1.py').read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('predict_batch', 'build_split', 'load_state_dict_from_url') for n in ast.walk(tree))
    optimizers = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'AdamW']
    assert len(optimizers) == 1 and ast.unparse(optimizers[0].args[0]) == 'head.parameters()'
    assert read(OUT/'smoke_fresh_process_replay.json')['passed']
    data = torch.load(CACHE/'TRAIN_features.pt', map_location='cpu')
    assert data['role'] == 'TRAIN_ONLY' and data['labels'].sum().item() == 458
    head = FracturePresenceHead(data['features'].shape[1], data['head_dropout'])
    head.load_state_dict(data['head_state'], strict=True)
    hashes = lambda: {k: hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k, v in head.state_dict().items()}
    smoke = read(OUT/'smoke.json')
    assert hashes() == smoke['initial']
    torch.manual_seed(42)
    opt = torch.optim.AdamW(head.parameters(), lr=.0001, betas=(.9, .98), weight_decay=.01)
    head.train()
    opt.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(head(data['features'][:64]).float(), data['labels'][:64], pos_weight=torch.tensor(2110/458))
    loss.backward()
    grads = {k: float(v.grad.abs().max()) for k, v in head.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
    opt.step()
    assert {'loss': float(loss), 'gradient_max': grads, 'unclipped_norm': float(norm)} == smoke['step']
    assert hashes() == smoke['final']
    result = {'passed': True, 'protocol_sha256': sha(OUT/'protocol.json'), 'auditor_sha256': sha(__file__),
        'smoke_sha256': sha(OUT/'smoke.json'), 'smoke_replay_sha256': sha(OUT/'smoke_fresh_process_replay.json'),
        'independent_one_update_gradient_and_final_hash_reconstruction_exact': True,
        'disposable_updates_including_this_audit': 3, 'disposable_weights_discarded': True,
        'scoped_release': 'One seed42 TRAIN-only frozen-feature head LAST410; no longer training, seed/epoch/threshold search or backbone update',
        'quality_gain_established': False, 'public_evaluation_required': True, 'production_released': False}
    (OUT/'execution_release.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Independent head-only LAST410 scoped release PASS; all disposable weights discarded')


if __name__ == '__main__':
    main()
