"""Approved official SAM-Med2D bytes only; no deserialization."""
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from download_lapa_roberta_assets_v1 import Form

root = Path(__file__).resolve().parents[1] / 'references/native_assets_favp_v1'
fid = '1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl'
url = 'https://drive.google.com/uc?' + urlencode({'export': 'download', 'id': fid})
size = 2561829701
target = root / 'sam-med2d_b.pth'
part = root / 'sam-med2d_b.pth.partial'
receipt = root / 'sammed_receipt.json'
assert root.is_dir() and not any(p.exists() for p in (target, part, receipt))
(root / 'sammed_download_started.json').write_text(json.dumps({'pid': os.getpid(), 'source': url, 'expected_bytes': size}))
try:
    response = urlopen(url, timeout=60)
    if 'text/html' in response.headers.get('Content-Type', ''):
        form = Form()
        form.feed(response.read(1000000).decode())
        response.close()
        assert form.action == 'https://drive.usercontent.google.com/download'
        assert form.fields.get('id') == fid
        response = urlopen(form.action + '?' + urlencode(form.fields), timeout=60)
    assert 'text/html' not in response.headers.get('Content-Type', '')
    count, digest = 0, hashlib.sha256()
    with response, part.open('xb') as out:
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            assert count <= size
            out.write(chunk)
            digest.update(chunk)
    assert count == size
    part.rename(target)
    result = dict(complete=True, source=url, publisher='https://github.com/OpenGVLab/SAM-Med2D',
                  bytes=count, sha256=digest.hexdigest(), deserialized=False,
                  limitation='Local checksum and publisher link, not independently published checksum or FAVP compatibility verification.')
    receipt.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
except Exception as error:
    (root / 'sammed_failure.json').write_text(json.dumps({'error': repr(error), 'source': url}, indent=2))
    raise
