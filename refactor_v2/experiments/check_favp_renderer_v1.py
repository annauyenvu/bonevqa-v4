"""Bounded source-extracted renderer fixture; no models or clinical data."""
import ast
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/favp_author_b43941ba/favp/models'
OUT = ROOT / 'reports/favp_renderer_fixture_v1'


def main():
    OUT.mkdir(exist_ok=False)
    ns = dict(torch=torch, np=np, cv2=cv2, F=F)
    hashes = {}
    for name, selected in [('transform.py', {'boxes_to_circles', 'str2rgb'}),
                           ('fine_grained_visual_prompt.py', {'FGVP_ENSEMBLE'})]:
        p = SOURCE / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        tree = ast.parse(p.read_text())
        nodes = [n for n in tree.body if getattr(n, 'name', None) in selected]
        assert len(nodes) == len(selected)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), ns)

    class IdentityResize:
        def apply_image(self, image):
            return image.copy()

        def apply_boxes_torch(self, boxes, original_size):
            return boxes.clone()

    renderer = ns['FGVP_ENSEMBLE'](
        'red', 1, 'green', .5, 'resize', 8, IdentityResize(),
        torch.zeros(3, 1, 1), torch.ones(3, 1, 1), 1, device='cpu')
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = (20, 40, 80)
    original = torch.from_numpy(image).permute(2, 0, 1).float()
    shapes = {}
    for count in (0, 1, 2):
        masks = torch.zeros(count, 1, 8, 8, dtype=torch.bool)
        boxes = torch.tensor([[1., 1., 3., 3.], [4., 4., 6., 6.]])[:count]
        centers = torch.stack((boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)), 1)
        for i in range(count):
            start = 1 + 3 * i
            masks[i, 0, start:start+2, start:start+2] = True
        result = renderer('mask', image, centers, boxes, masks)
        assert result.shape == (count+1, 3, 8, 8)
        assert torch.equal(result[0], original)
        for i in range(count):
            diff = (result[i+1] != original).any(0)
            assert torch.equal(diff, masks[i, 0])
        shapes[str(count)] = list(result.shape)
    receipt = dict(passed=True, shapes=shapes, source_sha256=hashes,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   author_red_tuple=ns['str2rgb']('red'),
                   scope='Exact extracted mask renderer and helpers; identity resize, synthetic RGB pixels. No SAM, encoder, Q-Former, weights, clinical targets or quality release.')
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
