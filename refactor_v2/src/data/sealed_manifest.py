"""Read only manifests explicitly recorded in the audited data seal."""
import hashlib
import json
from pathlib import Path


def load_sealed_records(manifest_path, seal_path, project_root):
    root = Path(project_root).resolve()
    path = Path(manifest_path).resolve()
    seal = json.loads(Path(seal_path).read_text(encoding='utf-8'))
    if seal.get('status') != 'sealed_development_data':
        raise RuntimeError('Unsupported or unsealed data snapshot')
    relative = path.relative_to(root).as_posix()
    expected = seal.get('artifacts', {}).get(relative)
    if expected is None:
        raise RuntimeError('Manifest is absent from the data seal: ' + relative)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise RuntimeError('Manifest hash mismatch: ' + relative)
    records = json.loads(content)
    if not isinstance(records, list):
        raise ValueError('Expected a list of dataset records')
    return records
