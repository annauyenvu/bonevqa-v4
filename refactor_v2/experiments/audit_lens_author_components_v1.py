"""Synthetic checks on pinned Lens helpers and static notebook routes only."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'refactor_v2/references/lens_author_22a94007'
OUT = ROOT / 'refactor_v2/reports/lens_author_components_v1'


def main():
    torch.set_num_threads(1)
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '22a940079c8a6b9d94994c3b950ad4b28e7b253c'
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    path = SOURCE / 'src/localization_lens/modeling.py'
    spec = importlib.util.spec_from_file_location('pinned_lens_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(20261025)
    first = torch.randn(4, 8, requires_grad=True)
    second = torch.randn(4, 8, requires_grad=True)
    losses = {}
    for name in ('DecoupledContrastiveLoss', 'InfoNCELoss'):
        criterion = getattr(module, name)(temperature=0.1)
        loss = criterion(first, second)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
        assert torch.isfinite(first.grad).all() and torch.isfinite(second.grad).all()
        assert first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0
        singleton = criterion(first[:1], second[:1])
        assert singleton.item() == 0
        losses[name] = {'loss': loss.item(), 'finite_nonzero_gradients': True, 'batch1_zero': True}
    tokens = torch.arange(32.).reshape(1, 16, 2)
    result = module.PixelShuffleTokens(factor=2)(tokens, 4, 4)
    assert result.shape == (1, 4, 8)
    assert torch.equal(tokens.flatten().sort().values, result.flatten().sort().values)
    nb_path = SOURCE / 'notebooks/02_Train_Eval.ipynb'
    cells = json.loads(nb_path.read_text())['cells']
    code = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'code')
    assert 'run_stage' in code and 'SMOKE_TEST = True' in code
    assert 'epochs=5' in code and 'epochs=3' in code
    paths = [path, nb_path, SOURCE / 'notebooks/01_SAM_Augmentations.ipynb', SOURCE / 'src/localization_lens/training.py', SOURCE / 'src/localization_lens/inference.py']
    record = {
        'commit': commit, 'source_clean': True,
        'scope': 'Synthetic helper checks and notebook source inspection; no notebooks executed, models loaded, GT invented or training launched.',
        'contrastive_fixtures': losses,
        'pixel_unshuffle_shape': list(result.shape), 'pixel_values_preserved': True,
        'notebook_routes_to_inspected_run_stage': True,
        'notebook_default_smoke_test': True,
        'notebook_phase_epochs_arguments': [5, 3],
        'notebook_smoke_max_steps_per_phase': 2,
        'full_paper_reproduction_verified': False,
        'warning': 'Passing helper tests does not prove PixelShuffleTokens executes in the native training route. Batch1 yields no contrastive signal; accumulation does not enlarge negatives in this implementation.',
        'dependencies': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    OUT.mkdir(exist_ok=False)
    with (OUT / 'fixture_and_notebook_audit.json').open('x') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != 'dependencies'}, indent=2))


if __name__ == '__main__':
    main()
