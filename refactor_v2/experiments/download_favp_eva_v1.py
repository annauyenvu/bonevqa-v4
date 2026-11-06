"""Approved EVA byte download, with bounded size and persistent receipt."""
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1] / 'references/native_assets_favp_v1'
root.mkdir(exist_ok=True)
name = 'eva_vit_g.pth'
url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/' + name
size = 2025249237
part, target, receipt = root / (name + '.partial'), root / name, root / 'eva_receipt.json'
assert not any(p.exists() for p in (part, target, receipt)), 'Refuse duplicate or overwrite'
(root / 'eva_download_started.json').write_text(json.dumps({'pid': os.getpid(), 'source': url, 'expected_bytes': size}))
try:
    digest, count = hashlib.sha256(), 0
    with urlopen(url, timeout=60) as response, part.open('xb') as out:
        assert 'text/html' not in response.headers.get('Content-Type', '')
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            assert count <= size
            out.write(chunk)
            digest.update(chunk)
    assert count == size, (count, size)
    part.rename(target)
    data = dict(complete=True, source=url, bytes=count, sha256=digest.hexdigest(), deserialized=False,
                limitation='Local checksum only; full FAVP checkpoint compatibility and pretraining provenance not established.')
    receipt.write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)
except Exception as error:
    (root / 'eva_failure.json').write_text(json.dumps({'error': repr(error), 'source': url}, indent=2))
    raise
