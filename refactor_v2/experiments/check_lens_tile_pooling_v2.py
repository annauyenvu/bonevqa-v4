"""Corrected exception expectations; preserves v1 failure and author source."""
import ast
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/lens_author_22a94007/src/localization_lens/modeling.py'
OUT = ROOT / 'reports/lens_tile_pooling_fixture_v2'


def main():
    OUT.mkdir(exist_ok=False)
    namespace = {'torch': torch}
    nodes = [n for n in ast.parse(SOURCE.read_text()).body
             if getattr(n, 'name', '') == 'pool_semantic_views']
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    pool = namespace['pool_semantic_views']
    pixels = torch.ones(1, 6, 3, 2, 2)
    features = torch.tensor([1., 1., 2., 2., 3., 3.]).reshape(6, 1, 1)
    assert torch.equal(pool(features, pixels).flatten(), torch.tensor([1., 2., 3.]))
    uneven = torch.tensor([1., 2., 2., 3., 3., 3.]).reshape(6, 1, 1)
    assert torch.equal(pool(uneven, pixels).flatten(), torch.tensor([1.5, 2.5, 3.]))
    outcomes = {}
    for name, f, p, expected in [
        ('nondivisible', features[:5], pixels[:, :5], ValueError),
        ('zero_pixels', features, torch.zeros_like(pixels), ValueError),
        ('feature_count_mismatch', features[:5], pixels, RuntimeError),
    ]:
        try:
            pool(f, p)
        except expected as error:
            outcomes[name] = {'type': type(error).__name__, 'message': str(error)}
        else:
            raise AssertionError(name)
    # Two samples with different, but internally equal, tile counts plus padding.
    padded = torch.ones(2, 6, 3, 2, 2)
    padded[1, 3:] = 0
    combined = torch.cat([features, torch.tensor([4., 5., 6.]).reshape(3, 1, 1)])
    assert torch.equal(pool(combined, padded).squeeze(-1), torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    receipt = dict(passed=True, exception_checks=outcomes,
                   equal_tile_and_padded_batch_checks=True,
                   unequal_view_counts_accepted_with_mixing=[1.5, 2.5, 3.],
                   source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   limitation='Synthetic helper only; native processor tile-count/order contract unverified. No quality release. V1 preserved.')
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
