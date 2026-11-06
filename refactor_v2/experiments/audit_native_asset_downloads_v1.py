"""Final byte audit across approved asset packages; never deserialize weights."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'references'
OUT = ROOT / 'reports/native_asset_acquisition_closure_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    packages = [
        ('LaPA_M3AE', REF/'native_assets_lapa_v1/m3ae_download_receipt.json', 'm3ae.ckpt'),
        ('LaPA_RoBERTa', REF/'native_assets_lapa_v1/roberta-base/receipt.json', None),
        ('FAVP_QFormer', REF/'native_assets_favp_v1/qformer_receipt.json', 'blip2_pretrained_flant5xxl.pth'),
        ('FAVP_EVA', REF/'native_assets_favp_v1/eva_receipt.json', 'eva_vit_g.pth'),
        ('FAVP_SAMMed2D', REF/'native_assets_favp_v1/sammed_receipt.json', 'sam-med2d_b.pth'),
        ('FAVP_Vicuna', REF/'native_assets_favp_v1/vicuna-7b/receipt.json', None),
        ('Lens_SmolVLM', REF/'native_assets_lens_v1/SmolVLM-256M-Instruct/receipt.json', None),
        ('Lens_SAMHuge', REF/'native_assets_lens_v1/sam-vit-huge/receipt.json', None),
    ]
    results = []
    for package, receipt_path, singleton in packages:
        receipt = json.loads(receipt_path.read_text())
        assert receipt['complete'] is True, package
        files = [dict(name=singleton, bytes=receipt['bytes'], sha256=receipt['sha256'])] if singleton else receipt['files']
        manifest_path = receipt_path.parent/'source_manifest.json'
        publisher = {}
        if manifest_path.exists():
            publisher = {f['rfilename']: f for f in json.loads(manifest_path.read_text())['files']}
        checked = []
        for entry in files:
            path = receipt_path.parent/entry['name']
            assert path.stat().st_size == entry['bytes'], str(path)
            actual = sha(path)
            assert actual == entry['sha256'], str(path)
            reference = publisher.get(entry['name'])
            evidence = 'local_receipt_sha256'
            if reference:
                assert entry['bytes'] == reference['size']
                if 'lfs' in reference:
                    assert actual == reference['lfs']['sha256']
                    evidence = 'publisher_lfs_sha256'
                else:
                    data = path.read_bytes()
                    git = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                    assert git == reference['blobId']
                    evidence = 'publisher_git_blob_sha1_and_local_sha256'
            checked.append(dict(path=str(path), bytes=entry['bytes'], sha256=actual, evidence=evidence))
        assert not list(receipt_path.parent.glob('*.partial')), package
        results.append(dict(package=package, receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
                            files=checked, total_bytes=sum(f['bytes'] for f in checked)))
        print(package, 'verified', flush=True)
    result = dict(byte_audit_passed=True, checked_unix=time.time(), packages=results,
                  total_files=sum(len(p['files']) for p in results), total_bytes=sum(p['total_bytes'] for p in results),
                  free_disk_bytes=shutil.disk_usage(ROOT).free, weights_deserialized=False,
                  native_reproduction_complete=False,
                  unresolved=['LaPA graph577/nodeorder/answerassets: author link expired',
                              'FAVP pretrained stage handoff provenance/effective config',
                              'Runtime/model gates and Lens paper-route/supervision verification'])
    target = OUT/'inventory.json'
    target.write_text(json.dumps(result, indent=2))
    (OUT/'seal.json').write_text(json.dumps({'inventory_sha256':sha(target), 'audit_script_sha256':sha(Path(__file__))}, indent=2))
    print(json.dumps({k:result[k] for k in ['byte_audit_passed','total_files','total_bytes','free_disk_bytes']}))


if __name__ == '__main__':
    main()
