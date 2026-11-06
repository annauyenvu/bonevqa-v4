"""Synthetic acceptance boundaries of pinned Lens tile pooling; no model."""
import ast
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/lens_author_22a94007/src/localization_lens/modeling.py'
OUT = ROOT / 'reports/lens_tile_pooling_fixture_v1'


def main():
    OUT.mkdir(exist_ok=False)
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if getattr(n, 'name', '') == 'pool_semantic_views']
    assert len(nodes) == 1
    ns = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    pool = ns['pool_semantic_views']
    pixels = torch.ones(1, 6, 3, 2, 2)
    features = torch.tensor([1., 1., 2., 2., 3., 3.]).reshape(6, 1, 1)
    assert torch.equal(pool(features, pixels).flatten(), torch.tensor([1., 2., 3.]))
    # Three views with actual tile counts 1,2,3: total divisible by three,
    # but source groups by equal-size chunks, with no per-view metadata.
    uneven = torch.tensor([1., 2., 2., 3., 3., 3.]).reshape(6, 1, 1)
    mixed = pool(uneven, pixels).flatten()
    assert torch.equal(mixed, torch.tensor([1.5, 2.5, 3.]))
    rejected = []
    for name, f, p in [
        ('nondivisible', features[:5], pixels[:, :5]),
        ('zero_pixels', features, torch.zeros_like(pixels)),
        ('feature_count_mismatch', features[:5], pixels),
    ]:
        try:
            pool(f, p)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(name)
    receipt = {'passed': True, 'equal_tiles_correct': True,
               'unequal_tiles_accepted_but_mixed': mixed.tolist(),
               'rejected': rejected,
               'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'limit': 'Synthetic helper only. Does not show native processor emits unequal tile counts. Require real processor per-view count/order parity before release; all-zero pixels are treated as padding.'}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
