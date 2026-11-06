"""Fixed 410-step TRAIN-only head refit. No backbone optimizer or modelval access.

Requires separately verified feature cache, technical gate and scoped release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
CACHE = ROOT/'refactor_v2/reports/frozen_head_TRAIN_cache_v3_1'
OUT = ROOT/'refactor_v2/reports/frozen_fracture_head_pilot_v3_1'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists()
    audit = read(CACHE/'independent_artifact_audit.json')
    assert audit['passed'] and audit['cache_sha256'] == sha(CACHE/'TRAIN_features.pt')
    paths = [Path(__file__).resolve(), CACHE/'TRAIN_features.pt', CACHE/'protocol.json',
             CACHE/'report.json', CACHE/'fresh_process_replay.json', CACHE/'independent_artifact_audit.json',
             ROOT/'refactor_v2/reports/FROZEN_FEATURE_HEAD_PILOT_DRAFT_V3.json',
             ROOT/'refactor_v2/src/models/heads.py']
    OUT.mkdir()
    dump(OUT/'protocol.json', {'scope': 'One fixed frozen-feature head optimization pilot, not clinical confirmation',
        'dependencies_sha256': {str(p): sha(p) for p in paths}, 'parent_checkpoint_sha256': read(CACHE/'protocol.json')['target']['sha256'],
        'TRAIN_images': 2568, 'positives': 458, 'negatives': 2110,
        'seed': 42, 'epochs': 10, 'batch_size': 64, 'updates_per_epoch': 41, 'total_updates': 410,
        'optimizer': 'AdamW', 'lr': .0001, 'betas': [.9, .98], 'weight_decay': .01,
        'clip': 1., 'pos_weight': 2110/458, 'scheduler': None,
        'architecture': 'Unmodified existing FracturePresenceHead; four original tensors warmstarted',
        'features': 'Frozen eval-mode original CPU x_f; head train-mode dropout remains enabled',
        'sampler': 'torch.randperm each epoch from dedicated seed42 CPU generator; all2568 exactly once/epoch, lastbatch8',
        'control': 'Same original head and same frozen TRAIN features, zero additional updates',
        'checkpoint': 'LAST410 only; no epoch/seed/threshold selection',
        'modelval_use': 'None during fitting; fixed subsequent321-image public inference required, primary BA difference at0.5',
        'calibration_test_inference': False, 'production_released': False,
        'candidate_requires': 'Fresh-process public evaluation/replay, known-component paired intervals and independent audit before benefit claim',
        'no_automatic_retry': True})


def validate():
    p = read(OUT/'protocol.json')
    for path, digest in p['dependencies_sha256'].items():
        assert sha(path) == digest, path
    return p


def create():
    import torch
    from src.models.heads import FracturePresenceHead
    torch.set_num_threads(8)
    p = validate()
    data = torch.load(CACHE/'TRAIN_features.pt', map_location='cpu')
    assert data['role'] == 'TRAIN_ONLY' and data['optimizer_updates'] == 0
    head = FracturePresenceHead(data['features'].shape[1], data['head_dropout'])
    head.load_state_dict(data['head_state'], strict=True)
    assert len(dict(head.named_parameters())) == 4
    torch.manual_seed(42)
    optimizer = torch.optim.AdamW(head.parameters(), lr=p['lr'], betas=tuple(p['betas']), weight_decay=p['weight_decay'])
    return p, data, head, optimizer


def tensor_hashes(head):
    return {k: hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k, v in head.state_dict().items()}


def step(head, optimizer, x, y, p):
    import torch
    import torch.nn.functional as F
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(head(x).float(), y, pos_weight=torch.tensor(p['pos_weight']))
    assert torch.isfinite(loss)
    loss.backward()
    grads = {k: float(v.grad.abs().max()) for k, v in head.named_parameters()}
    assert all(torch.isfinite(v.grad).all() for v in head.parameters()) and all(v > 0 for v in grads.values())
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), p['clip'])
    assert torch.isfinite(norm)
    optimizer.step()
    return {'loss': float(loss), 'gradient_max': grads, 'unclipped_norm': float(norm)}


def smoke_measure():
    import torch
    p, data, head, optimizer = create()
    assert all(torch.equal(head.state_dict()[k], v) for k, v in data['head_state'].items())
    initial = tensor_hashes(head)
    result = step(head, optimizer, data['features'][:64], data['labels'][:64], p)
    final = tensor_hashes(head)
    assert all(initial[k] != final[k] for k in initial)
    clone = type(head)(data['features'].shape[1], data['head_dropout']).eval()
    clone.load_state_dict(head.state_dict(), strict=True)
    head.eval()
    with torch.no_grad():
        assert torch.equal(head(data['features'][:64]), clone(data['features'][:64]))
    return {'initial': initial, 'final': final, 'step': result, 'roundtrip_exact': True,
            'updates': 1, 'disposable_not_quality_initialization': True}


def smoke():
    assert not (OUT/'smoke.json').exists()
    dump(OUT/'smoke.json', smoke_measure())
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'smoke_replay'], check=True)


def smoke_replay():
    assert smoke_measure() == read(OUT/'smoke.json')
    dump(OUT/'smoke_fresh_process_replay.json', {'passed': True, 'two_actual_disposable_updates_total': 2,
        'same_initial_final_hashes_loss_gradients_roundtrip_exact': True})


def run():
    import torch
    p = validate()
    release = read(OUT/'execution_release.json')
    assert release['passed'] and release['protocol_sha256'] == sha(OUT/'protocol.json')
    assert not (OUT/'progress.json').exists()
    start = time.perf_counter()
    dump(OUT/'progress.json', {'status': 'running', 'updates': 0})
    try:
        p, data, head, optimizer = create()
        assert tensor_hashes(head) == read(OUT/'smoke.json')['initial']
        rng = torch.Generator().manual_seed(42)
        history, updates = [], 0
        for epoch in range(10):
            order = torch.randperm(2568, generator=rng)
            assert sorted(order.tolist()) == list(range(2568))
            loss_sum = 0.
            for offset in range(0, 2568, 64):
                assert updates < 410
                idx = order[offset:offset+64]
                result = step(head, optimizer, data['features'][idx], data['labels'][idx], p)
                loss_sum += result['loss']*len(idx)
                updates += 1
            assert updates == (epoch+1)*41
            history.append({'epoch': epoch+1, 'updates': updates, 'TRAIN_weighted_BCE': loss_sum/2568,
                            'last_batch_gradient_max': result['gradient_max'], 'last_batch_unclipped_norm': result['unclipped_norm']})
            print(history[-1], flush=True)
            dump(OUT/'progress.json', {'status': 'running', 'updates': updates, 'history': history})
        assert updates == 410
        torch.save({'head_state': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'protocol_sha256': sha(OUT/'protocol.json'), 'parent_checkpoint_sha256': p['parent_checkpoint_sha256'],
            'cache_sha256': sha(CACHE/'TRAIN_features.pt'), 'updates': 410, 'seed': 42,
            'quality_candidate_only': True, 'production_released': False}, OUT/'HEAD_LAST410.pt')
        exported = torch.load(OUT/'HEAD_LAST410.pt', map_location='cpu')
        assert all(torch.equal(v, exported['head_state'][k]) for k, v in head.state_dict().items())
        assert all(int(v['step']) == 410 for v in exported['optimizer']['state'].values())
        update_max = {k: float((v-data['head_state'][k]).abs().max()) for k, v in head.state_dict().items()}
        assert len(update_max) == 4 and all(v > 0 for v in update_max.values())
        dump(OUT/'report.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'candidate_sha256': sha(OUT/'HEAD_LAST410.pt'),
            'elapsed_seconds': time.perf_counter()-start, 'history': history, 'head_final_hashes': tensor_hashes(head),
            'head_update_max_from_original': update_max, 'export_reload_tensors_exact': True,
            'optimizer_updates': 410, 'backbone_updates': 0, 'modelval_calibration_test_inference': False,
            'quality_gain_established': False, 'public_evaluation_pending': True, 'production_released': False})
        dump(OUT/'progress.json', {'status': 'LAST410_saved_public_evaluation_pending', 'updates': 410})
    except Exception as error:
        dump(OUT/'progress.json', {'status': 'failed_preserved_no_automatic_retry', 'failure': repr(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'smoke_replay', 'run'])
    {'prepare': prepare, 'smoke': smoke, 'smoke_replay': smoke_replay, 'run': run}[parser.parse_args().mode]()
