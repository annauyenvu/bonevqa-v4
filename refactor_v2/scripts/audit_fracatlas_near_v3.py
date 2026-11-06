"""Enumerate FracAtlas pHash neighbours, including unequal fingerprints."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def neighbours(item, rows):
    index, row = item
    first = int(row['phash'], 16)
    return [{'left': row['path'], 'right': other['path'],
             'hamming': distance}
            for other in rows[index + 1:]
            if (distance := (first ^ int(other['phash'], 16)).bit_count()) <= 8]


def main():
    root = Path(__file__).resolve().parents[2] / 'refactor_v2/reports'
    rows = [row for line in (root / 'perceptual_fingerprints_v3.jsonl').open()
            if 'phash' in (row := json.loads(line)) and '\\fracatlas\\' in row['path']]
    pairs = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        for result in pool.map(lambda item: neighbours(item, rows), enumerate(rows)):
            pairs.extend(result)
    output = {'status': 'candidate_pixels_pending', 'images': len(rows),
              'threshold': 8, 'pairs': pairs,
              'limitation': 'Strict decode failures excluded; pHash proximity is not proof of duplication.'}
    (root / 'fracatlas_hamming_candidates_v3.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({'images': len(rows), 'candidate_pairs': len(pairs)}), flush=True)


if __name__ == '__main__':
    main()
