"""Download an explicit native-model allowlist; no model imports or loads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

PACKAGES = {
    'vicuna': dict(repo='Vision-CAIR/vicuna-7b', revision='0abeb8872d75421e58ec146c39b31f5dc78c7613',
                   directory='native_assets_favp_v1/vicuna-7b', files=[
                       'config.json', 'generation_config.json', 'pytorch_model.bin.index.json',
                       'special_tokens_map.json', 'tokenizer.model', 'tokenizer_config.json',
                       'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin']),
}


def dump(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', choices=PACKAGES)
    args = parser.parse_args()
    spec = PACKAGES[args.package]
    root = Path(__file__).resolve().parents[1] / 'references' / spec['directory']
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    dump(root / 'started.json', dict(pid=os.getpid(), started_unix=started, package=args.package, **spec))
    completed = []
    try:
        api = f"https://huggingface.co/api/models/{spec['repo']}/revision/{spec['revision']}?blobs=true"
        with urlopen(api, timeout=60) as response:
            metadata = json.load(response)
        assert metadata['sha'] == spec['revision']
        indexed = {f['rfilename']: f for f in metadata['siblings']}
        selected = [indexed[name] for name in spec['files']]
        dump(root / 'source_manifest.json', dict(repo=spec['repo'], revision=spec['revision'], files=selected))
        total = sum(f['size'] for f in selected)
        assert shutil.disk_usage(root).free > total + 20 * 1024**3, 'Insufficient 20GiB reserve'
        for info in selected:
            name, size = info['rfilename'], info['size']
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{quote(name)}?download=true"
            part = root / (name + '.partial')
            target = root / name
            assert not target.exists() and not part.exists()
            digest, count = hashlib.sha256(), 0
            progress_time = 0
            with urlopen(Request(url, headers={'User-Agent': 'BoneVQA-approved-asset-acquisition'}), timeout=90) as response, part.open('xb') as out:
                assert 'text/html' not in response.headers.get('Content-Type', '')
                while chunk := response.read(4 * 1024 * 1024):
                    count += len(chunk)
                    assert count <= size, 'Size exceeds manifest'
                    out.write(chunk)
                    digest.update(chunk)
                    if time.time() - progress_time > 30:
                        dump(root / 'progress.json', dict(file=name, received_bytes=count, expected_bytes=size, updated_unix=time.time()))
                        progress_time = time.time()
            assert count == size, (name, count, size)
            sha = digest.hexdigest()
            if 'lfs' in info:
                assert sha == info['lfs']['sha256'], 'Publisher SHA-256 mismatch'
                verification = 'publisher_lfs_sha256'
            else:
                content = part.read_bytes()
                git_sha = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
                assert git_sha == info['blobId'], 'Publisher Git blob mismatch'
                verification = 'publisher_git_blob_sha1'
            part.rename(target)
            completed.append(dict(name=name, bytes=count, sha256=sha, source=url, verification=verification))
            dump(root / 'receipt.json', dict(complete=False, repo=spec['repo'], revision=spec['revision'], files=completed, deserialized=False))
            print(name, count, verification, flush=True)
        dump(root / 'receipt.json', dict(complete=True, repo=spec['repo'], revision=spec['revision'], files=completed,
                                       total_bytes=total, deserialized=False, elapsed_seconds=time.time()-started))
        print('PACKAGE_COMPLETE', args.package, total, flush=True)
    except Exception as error:
        dump(root / 'failure.json', dict(error=repr(error), completed=completed, failed_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
