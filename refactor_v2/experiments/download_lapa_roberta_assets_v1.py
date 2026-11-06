"""Download approved author-linked bytes only; never deserialize weights."""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / 'references/native_assets_lapa_v1/roberta-base'
FILES = [
 ('config.json', '1jy_4Nx7SvkSiYOxQTLMAF6GHkE7u97nm', 481),
 ('dict.txt', '1uGup0k4DslpEO6GUXRbSuF0OQcIhzbbb', 603290),
 ('merges.txt', '1GG23QZ8LpFRzuBLtRq4vwi1hqIsi6yqv', 456318),
 ('pytorch_model.bin', '1WKr_S8dgP0uFJbV4qTvnV2ZCgYC9DJNm', 501200538),
 ('README.md', '1dNDnNWRlEYsQD3oZV6qs27GffK7GoU6U', 9078),
 ('tokenizer.json', '1SBqSK7Md7R4jgsjB7Ue2JawjVXbTN2f4', 1355863),
 ('vocab.json', '1gk8rU2PJttl8F6k5-LlT8HozscZjygK9', 898823),
]

class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = {}
        self.action = None
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.action = a.get('action')
        if tag == 'input' and a.get('type') == 'hidden':
            self.fields[a['name']] = a.get('value', '')

def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    receipts = []
    try:
        for name, fid, size in FILES:
            url = 'https://drive.google.com/uc?' + urlencode({'export': 'download', 'id': fid})
            response = urlopen(url, timeout=60)
            if 'text/html' in response.headers.get('Content-Type', ''):
                form = Form()
                form.feed(response.read(1000000).decode())
                response.close()
                assert form.action == 'https://drive.usercontent.google.com/download', 'Unexpected confirmation page'
                assert form.fields.get('id') == fid
                response = urlopen(form.action + '?' + urlencode(form.fields), timeout=60)
            assert 'text/html' not in response.headers.get('Content-Type', ''), 'HTML is not an asset'
            part = ROOT / (name + '.partial')
            digest, count = hashlib.sha256(), 0
            with response, part.open('xb') as out:
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    assert count <= size, 'Exceeds listed size'
                    out.write(chunk)
                    digest.update(chunk)
            assert count == size, (name, count, size)
            part.rename(ROOT / name)
            receipts.append(dict(name=name, drive_id=fid, bytes=count, sha256=digest.hexdigest(), source=url))
            (ROOT / 'receipt.json').write_text(json.dumps({'files': receipts, 'complete': False}, indent=2))
            print(name, count, flush=True)
        (ROOT / 'receipt.json').write_text(json.dumps({'files': receipts, 'complete': True, 'weights_deserialized': False}, indent=2))
    except Exception as error:
        (ROOT / 'failure.json').write_text(json.dumps({'error': repr(error), 'completed_files': receipts}, indent=2))
        raise

if __name__ == '__main__':
    main()
