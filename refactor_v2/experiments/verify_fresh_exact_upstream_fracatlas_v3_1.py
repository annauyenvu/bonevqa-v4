"""Separate-process rehash and independent manifest/overlap arithmetic."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'refactor_v2/reports/fresh_exact_upstream_fracatlas_v3_1'


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while chunk := f.read(2**20):
            h.update(chunk)
    return h.hexdigest()


def check_bytes(row):
    p = Path(row['path'])
    assert p.stat().st_size == row['bytes']
    assert digest(p) == row['sha256'], row['path']


def main():
    started = time.perf_counter()
    output = OUT/'independent_verification.json'
    assert not output.exists()
    r = json.loads((OUT/'report.json').read_text())
    assert digest(OUT/'image_hashes.jsonl') == r['image_hashes_sha256']
    assert digest(ROOT/'refactor_v2/experiments/audit_fresh_exact_upstream_fracatlas_v3_1.py') == r['script_sha256']
    cache = ROOT/'refactor_v2/reports/perceptual_fingerprints_v3.jsonl'
    assert digest(cache) == r['source_path_cache_sha256']
    rows = [json.loads(s) for s in (OUT/'image_hashes.jsonl').read_text().splitlines()]
    cached_paths = [json.loads(s)['path'] for s in cache.read_text().splitlines()]
    assert sorted(cached_paths) == [v['path'] for v in rows]
    assert len(rows) == len(set(cached_paths)) == r['source_path_count'] == 100219
    assert sum(v['bytes'] for v in rows) == r['bytes_freshly_read']
    with ThreadPoolExecutor(max_workers=6) as pool:
        for _ in pool.map(check_bytes, rows):
            pass
    hashes = {v['path']: v['sha256'] for v in rows}
    groups = {}
    for key, summary in r['datasets'].items():
        if key == 'fracatlas/all_available_source':
            paths = {p for p in hashes if '\\fracatlas\\' in p}
        else:
            p = Path(summary['manifest_path'])
            assert digest(p) == summary['manifest_sha256']
            entries = json.loads(p.read_text()); assert len(entries) == summary['rows']
            dataset = 'fracatlas' if key.startswith('fracatlas/') else ('pmc_vqa' if 'pmc_vqa' in key else 'roco' if 'roco' in key else 'vqa_rad')
            prefix = 'BONE_VQA\\BaoYen_work\\data\\'+dataset+'\\'+('FracAtlas\\' if dataset == 'fracatlas' else '')
            paths = {str(ROOT/'data'/(prefix+e['image'].replace('/', '\\'))) for e in entries}
        assert len(paths) == summary['image_paths']
        groups[key] = {hashes[p] for p in paths}
        assert len(groups[key]) == summary['unique_image_hashes']
    for pair in r['fracatlas_upstream_exact_overlaps']:
        actual = sorted(groups[pair['left']].intersection(groups[pair['right']]))
        assert actual == pair['shared_byte_hashes'] and len(actual) == pair['shared_sha256_images']
    oldpath = ROOT/'refactor_v2/reports/data_inventory_exact_v3.json'
    assert digest(oldpath) == r['old_inventory_sha256']
    old = json.loads(oldpath.read_text())
    for pair in old['overlaps']:
        assert len(groups[pair['left']] & groups[pair['right']]) == pair['shared_sha256_images']
    for key, prior in old['datasets'].items():
        current = r['datasets'][key]
        for a,b in [('rows','qa_or_caption_rows'),('image_paths','image_paths'),('unique_image_hashes','unique_image_hashes'),('manifest_sha256','manifest_sha256')]:
            assert current[a] == prior[b]
    assert len(groups['pmc_vqa/train_lon'] & groups['pmc_vqa/test']) == r['PMC_train_lon_test_shared_byte_hashes'] == 33252
    result = {'passed':True, 'script_sha256':digest(Path(__file__).resolve()),
        'report_sha256':digest(OUT/'report.json'), 'protocol_sha256':digest(OUT/'protocol.json'),
        'image_hashes_sha256':digest(OUT/'image_hashes.jsonl'),
        'all_source_paths_independently_rehashed':len(rows),
        'all_manifest_hashes_membership_unique_counts_recomputed':True,
        'all_reported_fracatlas_overlap_hashes_recomputed':True,
        'old_inventory_dataset_counts_and_reported_overlaps_reproduced':True,
        'elapsed_seconds':time.perf_counter()-started, 'model_calls':0, 'training_updates':0,
        'clinical_identity_or_independence_certified':False, 'manifest_changes':0,
        'long_training_authorized':False, 'step3_complete':False}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
