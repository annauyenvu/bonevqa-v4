"""Fresh SHA256 all cached source paths plus manifest-bound overlap counts."""
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'refactor_v2/reports/fresh_exact_upstream_fracatlas_v3_1'
CACHE = ROOT/'refactor_v2/reports/perceptual_fingerprints_v3.jsonl'
PREFIX = 'BONE_VQA\\BaoYen_work\\data\\'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def dump(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def hash_image(p):
    path = Path(p)
    before = path.stat()
    h = sha(path)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), p
    return {'path': p, 'sha256': h, 'bytes': after.st_size}


def source_image(dataset, image):
    folder = dataset+'\\'+('FracAtlas\\' if dataset == 'fracatlas' else '')
    return str(ROOT/'data'/(PREFIX+folder+image.replace('/', '\\')))


def manifests():
    result = {}
    for dataset, splits in {'roco': ('train','val','test'), 'pmc_vqa': ('train','train_lon','test'), 'vqa_rad': ('train','test')}.items():
        for split in splits:
            result[dataset+'/'+split] = (ROOT/'data'/(PREFIX+dataset+'\\'+split+'.json'), dataset)
    base = ROOT/'.evaluation_runtime/data/pretraining_joint_groups_v3'
    for name, dataset in [('roco_train.json','roco'),('roco_validation_caption_valid.json','roco'),('pmc_vqa_train_label_filtered.json','pmc_vqa')]:
        result['clean_pretraining/'+name] = (base/name, dataset)
    base = ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0'
    for role in ('train','model_val','calibration','test'):
        result['fracatlas/fold0/'+role] = (base/f'vqa_{role}.json', 'fracatlas')
    return result


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    cached = [json.loads(s) for s in CACHE.read_text().splitlines()]
    paths = sorted({r['path'] for r in cached})
    assert len(paths) == len(cached)
    bindings = {'script_sha256': sha(Path(__file__).resolve()), 'source_path_cache_sha256': sha(CACHE)}
    dump(OUT/'protocol.json', {**bindings, 'path_count': len(paths),
        'method': 'Fresh streaming SHA256, six I/O workers, stat stable before/after per path',
        'scope': 'Source bytes and existing manifest membership only, no clinical labels or model calls',
        'limitations': 'Byte equality only, no patient/clinical identity or independent evaluation certification'})
    hashes = {}; total_bytes = 0
    with (OUT/'image_hashes.jsonl').open('x') as stream, ThreadPoolExecutor(max_workers=6) as pool:
        for n, record in enumerate(pool.map(hash_image, paths), 1):
            stream.write(json.dumps(record)+'\n')
            hashes[record['path']] = record['sha256']; total_bytes += record['bytes']
            if n % 2000 == 0 or n == len(paths):
                stream.flush()
                progress = {'processed': n, 'total': len(paths), 'bytes_read': total_bytes,
                    'elapsed_seconds': time.perf_counter()-started, 'status': 'hashing' if n < len(paths) else 'summarizing'}
                dump(OUT/'progress.json', progress); print(json.dumps(progress), flush=True)
    summaries = {}; sets = {}; members = {}
    for key, (p, dataset) in manifests().items():
        rows = json.loads(p.read_text())
        selected = sorted({source_image(dataset, r['image']) for r in rows})
        assert all(x in hashes for x in selected), (key, [x for x in selected if x not in hashes][:3])
        sets[key] = {hashes[x] for x in selected}; members[key] = selected
        summaries[key] = {'manifest_path': str(p), 'manifest_sha256': sha(p),
            'rows': len(rows), 'image_paths': len(selected), 'unique_image_hashes': len(sets[key])}
    fracall = [p for p in paths if '\\fracatlas\\' in p]
    sets['fracatlas/all_available_source'] = {hashes[p] for p in fracall}; members['fracatlas/all_available_source'] = fracall
    summaries['fracatlas/all_available_source'] = {'image_paths': len(fracall), 'unique_image_hashes': len(sets['fracatlas/all_available_source'])}
    overlaps = []
    for left in sorted(k for k in sets if k.startswith('fracatlas/')):
        for right in sorted(k for k in sets if not k.startswith('fracatlas/')):
            shared = sets[left] & sets[right]
            overlaps.append({'left': left, 'right': right, 'shared_sha256_images': len(shared),
                'shared_byte_hashes': sorted(shared)})
    oldpath = ROOT/'refactor_v2/reports/data_inventory_exact_v3.json'
    old = json.loads(oldpath.read_text())
    old_checks = {}
    for key, prior in old['datasets'].items():
        current = summaries[key]
        old_checks[key] = all(current[new] == prior[before] for new,before in
            [('rows','qa_or_caption_rows'),('image_paths','image_paths'),('unique_image_hashes','unique_image_hashes'),('manifest_sha256','manifest_sha256')])
    pmc_overlap = len(sets['pmc_vqa/train_lon'] & sets['pmc_vqa/test'])
    report = {**bindings, 'status': 'complete_pending_separate_verification',
        'source_path_count': len(paths), 'bytes_freshly_read': total_bytes,
        'image_hashes_sha256': sha(OUT/'image_hashes.jsonl'),
        'old_inventory_sha256': sha(oldpath), 'old_inventory_datasets_reproduced': old_checks,
        'datasets': summaries, 'fracatlas_upstream_exact_overlaps': overlaps,
        'PMC_train_lon_test_shared_byte_hashes': pmc_overlap,
        'elapsed_seconds': time.perf_counter()-started,
        'clinical_labels_created': False, 'model_calls': 0, 'training_updates': 0,
        'manifest_changes': 0, 'independence_certified': False,
        'long_training_authorized': False, 'step3_complete': False}
    dump(OUT/'report.json', report)
    dump(OUT/'progress.json', {'status':'complete_pending_separate_verification', 'report_sha256':sha(OUT/'report.json')})
    print(json.dumps({'source_paths':len(paths),'all_old_counts_reproduced':all(old_checks.values()),
        'fracatlas_upstream_overlap_counts':[v['shared_sha256_images'] for v in overlaps],
        'PMC_train_lon_test_overlap':pmc_overlap,'elapsed_seconds':report['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
