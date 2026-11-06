"""Bind locally available backbone/tokenizer bytes; no model or network work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.evaluation_runtime/hf_cache/hub'
OUT = ROOT / 'refactor_v2/reports/lapa_port_backbone_binding_v1'
MODELS = (
    'microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    'microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract',
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    records = []
    for name in MODELS:
        root = CACHE / ('models--' + name)
        revision = (root / 'refs/main').read_text().strip()
        snapshot = root / 'snapshots' / revision
        assert snapshot.is_dir(), snapshot
        files = []
        for path in sorted(snapshot.rglob('*')):
            if path.is_file():
                resolved = path.resolve(strict=True)
                assert resolved.is_relative_to(root.resolve()), resolved
                files.append({'name': str(path.relative_to(snapshot)),
                              'resolved_path': str(resolved),
                              'bytes': resolved.stat().st_size,
                              'sha256': sha(resolved)})
        assert files
        records.append({'model': name, 'revision': revision, 'files': files})
    receipt = {'models': records, 'scope': 'Available files in the two selected offline cache snapshots. This binds current bytes, not author-native pretraining provenance or a complete model inventory.',
               'runtime_file_access_traced': False,
               'quality_release': False,
               'script_sha256': sha(Path(__file__))}
    with (OUT / 'cache_binding.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'models': len(records), 'files': sum(len(r['files']) for r in records),
                      'receipt': str(OUT / 'cache_binding.json')}))


if __name__ == '__main__':
    main()
