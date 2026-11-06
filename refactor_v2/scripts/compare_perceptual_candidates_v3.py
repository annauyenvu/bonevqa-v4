"""Measure pixel agreement within equal-pHash candidate groups."""
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def compare(group):
    arrays = []
    for row in group['images']:
        with Image.open(row['path']) as source:
            pixels = np.asarray(ImageOps.exif_transpose(source).convert('L').resize((128, 128)), dtype=float)
        arrays.append((row['path'], pixels))
    pairs = []
    for (left, first), (right, second) in itertools.combinations(arrays, 2):
        difference = float(np.mean(np.abs(first - second)) / 255)
        pairs.append({'left': left, 'right': right, 'normalized_mae': difference,
                      'candidate_only': True})
    return pairs


def main():
    root = Path(__file__).resolve().parents[2] / 'refactor_v2/reports'
    groups = json.loads((root / 'perceptual_candidates_v3.json').read_text())['candidate_groups']
    with (root / 'perceptual_pixel_comparisons_v3.jsonl').open('w') as output:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for index, pairs in enumerate(pool.map(compare, groups), 1):
                for pair in pairs:
                    output.write(json.dumps(pair) + '\n')
                if index % 100 == 0 or index == len(groups):
                    (root / 'perceptual_comparison_progress_v3.json').write_text(json.dumps({
                        'groups_processed': index, 'total_groups': len(groups),
                        'status': 'complete' if index == len(groups) else 'running'}))
                    print(index, len(groups), flush=True)


if __name__ == '__main__':
    main()
