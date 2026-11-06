"""Read-only cached-fingerprint FracAtlas/upstream radius8 audit, no model."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'refactor_v2/reports/perceptual_fingerprints_v3.jsonl'
OUT = ROOT/'refactor_v2/reports/fracatlas_upstream_neighbours_v3_1'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def rows():
    records = [json.loads(s) for s in SOURCE.read_text().splitlines()]
    frac = [r for r in records if '\\fracatlas\\' in r['path']]
    upstream = [r for r in records if any(x in r['path'] for x in ('\\roco\\', '\\pmc_vqa\\'))]
    assert frac and upstream
    assert all('phash' in r and len(r['phash']) == 16 for r in frac+upstream)
    assert len({r['path'] for r in frac+upstream}) == len(frac+upstream)
    return frac, upstream


def search(frac, upstream):
    tree = None
    for i, r in enumerate(upstream):
        v = int(r['phash'], 16)
        if tree is None:
            tree = [v, [i], {}]
            continue
        node = tree
        while True:
            d = (v ^ node[0]).bit_count()
            if d == 0:
                node[1].append(i)
                break
            if d not in node[2]:
                node[2][d] = [v, [i], {}]
                break
            node = node[2][d]
    pairs = []
    for i, r in enumerate(frac):
        v = int(r['phash'], 16)
        pending = [tree]
        while pending:
            node = pending.pop()
            d = (v ^ node[0]).bit_count()
            if d <= 8:
                pairs.extend((i, j, d) for j in node[1])
            pending.extend(c for edge, c in node[2].items() if d-8 <= edge <= d+8)
        if (i+1) % 500 == 0:
            print(json.dumps({'processed': i+1, 'total': len(frac), 'pairs': len(pairs)}), flush=True)
    return sorted(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    frac, upstream = rows()
    binding = {'fingerprints_sha256': sha(SOURCE), 'script_sha256': sha(Path(__file__).resolve())}
    if args.verify:
        import numpy as np
        report = json.loads((OUT/'report.json').read_text())
        assert all(report[k] == v for k, v in binding.items())
        assert sha(OUT/'candidates.json') == report['candidates_sha256']
        # Independent exhaustive vectorized XOR/popcount, no BK-tree pruning.
        values = np.array([int(r['phash'], 16) for r in upstream], dtype=np.uint64)
        lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
        found = []
        for i, r in enumerate(frac):
            xor = np.bitwise_xor(values, np.uint64(int(r['phash'], 16)))
            distances = lut[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
            found.extend([i, int(j), int(distances[j])] for j in np.flatnonzero(distances <= 8))
        assert found == json.loads((OUT/'candidates.json').read_text())
        dump(OUT/'independent_exhaustive_verification.json', {
            'passed': True, **binding, 'report_sha256': sha(OUT/'report.json'),
            'all_pairs_checked': len(frac)*len(upstream), 'candidate_pairs': len(found),
            'method': 'Independent exhaustive vectorized XOR/byte-popcount, not BK-tree',
            'elapsed_seconds': time.perf_counter()-start, 'model_calls': 0,
            'fresh_all_image_fingerprints_recomputed': False,
            'clinical_duplicate_identity_certified': False, 'manifest_changes': 0})
        print(json.dumps({'verification_passed': True, 'candidate_pairs': len(found)}))
        return
    OUT.mkdir(exist_ok=False)
    dump(OUT/'protocol.json', {**binding, 'radius': 8,
        'query': 'all cached FracAtlas fingerprints', 'index': 'all cached ROCO/PMC fingerprints',
        'scope': 'Pixel-derived candidate search only, no QA labels, model or split changes',
        'limitations': 'No rotation/crop invariant proof; cached fingerprint search is not fresh byte or pixel verification'})
    pairs = search(frac, upstream)
    dump(OUT/'candidates.json', pairs)
    paths = [{'fracatlas': frac[i]['path'], 'upstream': upstream[j]['path'], 'hamming': d}
             for i, j, d in pairs]
    dump(OUT/'candidate_paths.json', paths)
    report = {**binding, 'status': 'search_complete_pending_exhaustive_verification',
        'fracatlas_images': len(frac), 'upstream_images': len(upstream),
        'all_possible_path_pairs': len(frac)*len(upstream), 'radius': 8,
        'candidate_pairs': len(pairs), 'equal_phash_pairs': sum(d == 0 for _, _, d in pairs),
        'candidate_upstream_dataset': dict(Counter('roco' if '\\roco\\' in upstream[j]['path'] else 'pmc_vqa' for _, j, _ in pairs)),
        'candidates_sha256': sha(OUT/'candidates.json'),
        'candidate_paths_sha256': sha(OUT/'candidate_paths.json'),
        'elapsed_seconds': time.perf_counter()-start,
        'fresh_all_image_fingerprints_recomputed': False,
        'clinical_duplicate_identity_certified': False,
        'no_candidates_would_prove_independence': False,
        'source_dataset_or_sealed_manifest_changes': 0, 'model_calls': 0, 'training_updates': 0,
        'long_training_authorized': False, 'step3_complete': False}
    dump(OUT/'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
