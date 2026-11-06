"""Approved byte-only retrieval from the URL in pinned FAVP source."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1] / 'references/native_assets_favp_v1'
root.mkdir(exist_ok=True)
name = 'blip2_pretrained_flant5xxl.pth'
url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/' + name
size = 433481467
part, target, receipt = root / (name + '.partial'), root / name, root / 'qformer_receipt.json'
assert not any(p.exists() for p in (part, target, receipt)), 'Do not overwrite or duplicate'
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
    data = dict(complete=True, source=url, bytes=count, sha256=digest.hexdigest(),
                deserialized=False, author_revision='b43941bac4774b796fd8650a263ea32c220e6aba',
                limitation='URL and size match; local checksum is not an independent author checksum. Does not establish full FAVP stage checkpoint provenance or compatibility.')
    receipt.write_text(json.dumps(data, indent=2))
    print(json.dumps(data), flush=True)
except Exception as error:
    (root / 'qformer_failure.json').write_text(json.dumps({'error': repr(error), 'source': url}, indent=2))
    raise
