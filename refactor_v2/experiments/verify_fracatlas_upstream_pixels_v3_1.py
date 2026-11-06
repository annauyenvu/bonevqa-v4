"""Fresh pixel/byte diagnostics for radius8 candidates, not duplicate labels."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from experiments import audit_fracatlas_upstream_neighbours_v3_1 as core
from scripts.audit_perceptual_images_v3 import fingerprint

OUT = core.ROOT/'refactor_v2/reports/fracatlas_upstream_neighbours_v3_2'


def load(path):
    p = Path(path)
    before = core.sha(p)
    fp = fingerprint(p)
    assert 'phash' in fp, fp
    with Image.open(p) as im:
        arr = np.asarray(ImageOps.exif_transpose(im).convert('L').resize((128, 128)), dtype=float)
    assert core.sha(p) == before
    return path, (before, fp['phash'], arr)


def main():
    started = time.perf_counter()
    output = OUT/'fresh_candidate_pixel_audit.json'
    assert not output.exists()
    independent = json.loads((OUT/'independent_exhaustive_verification.json').read_text())
    assert independent['passed']
    assert independent['report_sha256'] == core.sha(OUT/'report.json')
    report = json.loads((OUT/'report.json').read_text())
    assert core.sha(OUT/'candidate_paths.json') == report['candidate_paths_sha256']
    pairs = json.loads((OUT/'candidate_paths.json').read_text())
    cached = {r['path']: r['phash'] for s in core.SOURCE.read_text().splitlines() if 'phash' in (r := json.loads(s))}
    paths = sorted({r[k] for r in pairs for k in ('fracatlas', 'upstream')})
    with ThreadPoolExecutor(max_workers=6) as pool:
        loaded = dict(pool.map(load, paths))
    assert all(v[1] == cached[p] for p, v in loaded.items())
    scored = []
    for r in pairs:
        a, b = loaded[r['fracatlas']], loaded[r['upstream']]
        assert (int(a[1], 16)^int(b[1], 16)).bit_count() == r['hamming']
        corr = float(np.corrcoef(a[2].ravel(), b[2].ravel())[0, 1]) if a[2].std() and b[2].std() else 0.0
        scored.append({**r, 'exact_bytes_equal': a[0] == b[0],
            'grayscale128_normalized_MAE': float(np.abs(a[2]-b[2]).mean()/255),
            'grayscale128_correlation': corr, 'clinical_duplicate_label': None})
    result = {'status': 'complete', 'script_sha256': core.sha(Path(__file__).resolve()),
        'report_sha256': core.sha(OUT/'report.json'), 'candidate_paths_sha256': core.sha(OUT/'candidate_paths.json'),
        'independent_exhaustive_verification_sha256': core.sha(OUT/'independent_exhaustive_verification.json'),
        'fingerprints_sha256': core.sha(core.SOURCE), 'fresh_candidate_images': len(paths),
        'fresh_phash_matches_cached_all_candidates': True,
        'image_sha256': {p: v[0] for p, v in loaded.items()},
        'candidate_pairs': len(pairs), 'exact_byte_equal_pairs': sum(r['exact_bytes_equal'] for r in scored),
        'correlation_ge_0_985_pairs': sum(r['grayscale128_correlation'] >= .985 for r in scored),
        'normalized_MAE_le_0_01_pairs': sum(r['grayscale128_normalized_MAE'] <= .01 for r in scored),
        'scores': scored, 'elapsed_seconds': time.perf_counter()-started,
        'limits': ['Fresh raw pixels/bytes verified for candidates only, not all cached images',
                  'Radius8 pHash is not exhaustive for crops, rotations, composites or all transformations',
                  'Pixel similarity alone is not patient or clinical identity proof',
                  'Absence of candidate edges is not proof of independent evaluation'],
        'model_calls': 0, 'training_updates': 0, 'manifest_changes': 0,
        'clinical_labels_invented': False, 'independence_certified': False}
    core.dump(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('scores', 'image_sha256')}, indent=2))


if __name__ == '__main__':
    main()
