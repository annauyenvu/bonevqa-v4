"""Score cross-dataset perceptual candidates without changing source data."""
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load(path):
    with Image.open(path) as source:
        array = np.asarray(ImageOps.exif_transpose(source).convert('L').resize((64, 64)), dtype=float)
    return path, array.flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--within-pmc', action='store_true')
    parser.add_argument('--within-roco', action='store_true')
    args = parser.parse_args()
    reports = Path(__file__).resolve().parents[2] / 'refactor_v2/reports'
    stem = 'within_roco' if args.within_roco else 'within_pmc' if args.within_pmc else 'cross_dataset'
    pairs = [json.loads(line) for line in (reports / (stem + '_hamming_candidates_v3.jsonl')).open()]
    paths = sorted({pair[key] for pair in pairs for key in ('pmc', 'roco')})
    with ThreadPoolExecutor(max_workers=10) as pool:
        arrays = dict(pool.map(load, paths))
    selected = []
    for pair in pairs:
        first, second = arrays[pair['pmc']], arrays[pair['roco']]
        correlation = float(np.corrcoef(first, second)[0, 1]) if first.std() and second.std() else 0.0
        if correlation >= 0.985:
            selected.append({**pair, 'correlation': correlation})
    report = {'status': 'complete', 'candidate_pairs': len(pairs), 'correlation_min': 0.985,
              'selected_edges': selected, 'interpretation': 'Conservative grouping candidates, not clinical identity proof.'}
    (reports / (stem + '_pixel_edges_v3.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps({'candidate_pairs': len(pairs), 'selected_edges': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
