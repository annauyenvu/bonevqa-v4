"""Build reproducible perceptual fingerprints for subsequent candidate review."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


def fingerprint(path):
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('L')
            size = image.size
            pixels = np.asarray(image.resize((32, 32)), dtype=float)
        coefficients = dctn(pixels, norm='ortho')[:8, :8].flatten()
        bits = coefficients > np.median(coefficients[1:])
        return {'path': str(path), 'size': size,
                'phash': np.packbits(bits).tobytes().hex()}
    except Exception as error:
        return {'path': str(path), 'error': str(error)}


def main():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(Path(entry.path) for entry in os.scandir(root / 'data')
                   if entry.name.lower().endswith(('.jpg', '.jpeg', '.png')))
    output = root / 'refactor_v2/reports/perceptual_fingerprints_v3.jsonl'
    progress = root / 'refactor_v2/reports/perceptual_audit_progress_v3.json'
    errors = 0
    with output.open('w') as stream, ThreadPoolExecutor(max_workers=6) as pool:
        for count, record in enumerate(pool.map(fingerprint, paths), 1):
            stream.write(json.dumps(record) + '\n')
            errors += int('error' in record)
            if count % 1000 == 0 or count == len(paths):
                progress.write_text(json.dumps({'status': 'complete' if count == len(paths) else 'running',
                                               'processed': count, 'total': len(paths), 'errors': errors}))
                print(count, len(paths), errors, flush=True)


if __name__ == '__main__':
    main()
