"""Record cached base revisions and hashes without network access."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return str(path), value.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    cache = root / '.evaluation_runtime/hf_cache/hub'
    paths = []
    revisions = {}
    for repository in cache.glob('models--*'):
        if not any(name in repository.name for name in ('Qwen', 'BiomedCLIP', 'BiomedBERT')):
            continue
        for snapshot in (repository / 'snapshots').glob('*'):
            revisions.setdefault(repository.name, []).append(snapshot.name)
            paths.extend(path for path in snapshot.rglob('*') if path.is_file())
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(digest, paths))
    report = {'status': 'complete', 'snapshot_revisions': revisions, 'file_sha256': hashes,
              'limitation': 'Cache hashes establish local artifact identity, not original training-data independence.'}
    (root / 'refactor_v2/reports/base_cache_hashes_v3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'files_hashed': len(hashes), 'repositories': len(revisions)}), flush=True)


if __name__ == '__main__':
    main()
