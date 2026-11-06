"""Pin author source and exercise only its isolated GAT class on synthetic data.

No models, checkpoints, network, clinical labels or experiment configs are loaded.
"""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'refactor_v2/references/lapa_author_677faf90'
OUTPUT = ROOT / 'refactor_v2/reports/lapa_author_source_v1'
COMMIT = '677faf90c010dac133dcbde516bb7c96448b29d1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == COMMIT
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    files = sorted(p for p in SOURCE.rglob('*') if p.is_file() and '.git' not in p.relative_to(SOURCE).parts)
    hashes = {str(p): sha(p) for p in files}
    path = SOURCE / 'm3ae/modules/language_encoders/bert_model.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GraphAttentionLayer')
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    # Execute the pinned, isolated author class verbatim, not the full module.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    layer = namespace['GraphAttentionLayer'](2, 2, dropout=0.0, alpha=0.2, concat=False)
    with torch.no_grad():
        layer.W.copy_(torch.eye(2))
        layer.a.zero_()
    features = torch.tensor([[[1., 2.], [3., 4.], [8., 9.]]], requires_grad=True)
    adj = torch.eye(3).unsqueeze(0)
    output = layer(features, adj)
    # With zero attention logits, released code averages all nodes despite self-only adjacency.
    expected = features.mean(dim=1, keepdim=True).expand_as(features)
    assert torch.allclose(output, expected)
    assert not torch.allclose(output, features)
    changed = features.detach().clone()
    changed[0, 2] += 6
    intervened = layer(changed, adj)
    assert not torch.equal(output[0, 0], intervened[0, 0])
    output.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    sample_losses = torch.tensor([0.2, 0.4, 0.6, 0.8])
    result = {
        'commit': actual, 'source_files': len(files), 'source_unchanged': True,
        'fixture_scope': 'Exact isolated author GraphAttentionLayer; synthetic3nodes2features; not a full LaPA execution or clinical experiment.',
        'self_only_graph_preserves_node_independence': False,
        'self_only_zero_logits_equal_all_node_mean': True,
        'nonadjacent_node_intervention_changes_node0': True,
        'all_gradients_finite': True,
        'sum_over_mean_consistency_scale_at_batch4': float(sample_losses.sum() / sample_losses.mean()),
        'loss_fixture_scope': 'Reduction arithmetic only; complete author objective not executed.',
        'top_level_license_files': [p.name for p in SOURCE.iterdir() if p.is_file() and p.name.lower().startswith(('license', 'copying'))],
        'license_status': 'No top-level repository license file found; embedded per-file notices do not establish a license for all LaPA additions. Redistribution review pending.',
        'no_training_release': True,
        'dependencies': hashes,
        'audit_source_sha256': sha(Path(__file__)),
    }
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / 'source_and_fixture_audit.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'dependencies'}, indent=2))


if __name__ == '__main__':
    main()
