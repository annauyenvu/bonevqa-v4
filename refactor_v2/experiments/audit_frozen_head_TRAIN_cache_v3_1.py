"""Separate cache contract and saved-feature head-score audit; no backbone load."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
OUT = ROOT/'refactor_v2/reports/frozen_head_TRAIN_cache_v3_1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    assert not p.exists(), 'Preserve previous audit'
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def contract():
    p = read(OUT/'protocol.json')
    for path, digest in p['dependencies_sha256'].items():
        assert sha(path) == digest, path
    assert sha(p['target']['path']) == p['target']['sha256']
    records = read(p['TRAIN_manifest'])
    chosen = [records[i] for i in p['indices']]
    assert len(chosen) == len({r['image_id'] for r in chosen}) == 2568
    assert [r['image_id'] for r in chosen] == p['image_ids']
    assert sum(r['answer'] == 'yes' for r in chosen) == 458
    assert all(r['answer'] in ('yes', 'no') and r['question'] == 'Is there a fracture in this X-ray?' for r in chosen)
    assert not p['quality_training_released'] and p['optimizer_updates'] == 0
    source = ROOT/'refactor_v2/experiments/cache_frozen_head_TRAIN_v3_1.py'
    tree = ast.parse(source.read_text())
    public = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'predict_batch']
    assert len(public) == 1 and isinstance(public[0].args[0], ast.Dict)
    assert {k.value for k in public[0].args[0].keys} == {'images', 'questions'}
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('AdamW', 'backward', 'step') for n in ast.walk(tree))
    return p, chosen


def preflight():
    contract()
    write(OUT/'preflight.json', {'passed': True, 'protocol_sha256': sha(OUT/'protocol.json'),
        'auditor_sha256': sha(__file__), 'TRAIN2568_existing_labels_public_inputs_only': True,
        'no_optimizer_or_reserved_role_inference': True, 'head_training_released': False})
    print('TRAIN-only cache contract PASS')


def audit():
    import torch
    from src.models.heads import FracturePresenceHead
    torch.set_num_threads(8)
    p, chosen = contract()
    report = read(OUT/'report.json')
    assert report['protocol_sha256'] == sha(OUT/'protocol.json')
    assert report['cache_sha256'] == sha(OUT/'TRAIN_features.pt')
    assert report['replay_sha256'] == sha(OUT/'fresh_process_replay.json')
    assert read(OUT/'fresh_process_replay.json')['passed']
    data = torch.load(OUT/'TRAIN_features.pt', map_location='cpu')
    assert data['role'] == 'TRAIN_ONLY' and data['optimizer_updates'] == 0
    assert data['checkpoint_sha256'] == p['target']['sha256'] and data['protocol_sha256'] == sha(OUT/'protocol.json')
    x = data['features']
    assert x.shape[0] == 2568 and x.ndim == 2 and x.dtype == torch.float32 and torch.isfinite(x).all()
    assert data['image_ids'] == p['image_ids'] and data['indices'] == p['indices']
    assert data['labels'].tolist() == [float(r['answer'] == 'yes') for r in chosen]
    assert len(data['all263_trainable_hashes']) == 263 and len(data['decoded_image_sha256']) == 2568
    head = FracturePresenceHead(x.shape[1], data['head_dropout']).eval()
    head.load_state_dict(data['head_state'], strict=True)
    saved = read(ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1/token_mean.json')['records']
    probabilities = []
    with torch.no_grad():
        for i in range(0, len(x), 4):
            probabilities.extend(head(x[i:i+4]).sigmoid().tolist())
    assert probabilities == data['scores'] == [r['p_yes'] for r in saved]
    counts = {'TP': 0, 'FN': 0, 'TN': 0, 'FP': 0}
    for prob, r in zip(probabilities, chosen):
        counts[('TP' if prob >= .5 else 'FN') if r['answer'] == 'yes' else ('FP' if prob >= .5 else 'TN')] += 1
    assert counts == {'TP': 314, 'FN': 144, 'TN': 2025, 'FP': 85}
    write(OUT/'independent_artifact_audit.json', {'passed': True,
        'auditor_sha256': sha(__file__), 'report_sha256': sha(OUT/'report.json'),
        'cache_sha256': sha(OUT/'TRAIN_features.pt'), 'feature_shape': list(x.shape),
        'all2568_cached_head_public_scores_exact': True, 'original_TRAIN_counts': counts,
        'fresh_backbone_feature_replay_bound': True, 'optimizer_updates': 0,
        'modelval_calibration_test_inference': False, 'head_training_released': False,
        'limit': 'Separate head-only score reconstruction and artifact audit, not a third backbone extraction or quality gain'})
    print('Cache independent head-score/artifact audit PASS', counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'audit'])
    {'preflight': preflight, 'audit': audit}[parser.parse_args().mode]()
