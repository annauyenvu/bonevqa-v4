"""Small isolated-runtime helpers; no imports from the historical environment."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'references/lens_author_22a94007'
ASSETS=ROOT/'references/native_assets_lens_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_source(relative):
    path=SOURCE/relative
    hashes=json.loads((ROOT/'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['LocalizationLens']['sha256']
    assert sha(path)==hashes[str(path)],path
    return path


def verify_package(name):
    path=ASSETS/name
    receipt=json.loads((path/'receipt.json').read_text());assert receipt['complete']
    for entry in receipt['files']:assert sha(path/entry['name'])==entry['sha256']
    return path


def fixed_train_image(index=8):
    from PIL import Image
    project=ROOT.parent
    manifest=ROOT/'reports/step4_localization_v3_1/train.json'
    seal=json.loads((ROOT/'reports/step4_localization_v3_1/data_seal.json').read_text())
    assert sha(manifest)==seal['artifacts']['refactor_v2/reports/step4_localization_v3_1/train.json']
    row=json.loads(manifest.read_text())[index]
    image_path=project/'.evaluation_runtime/data/fracatlas/FracAtlas'/row['image']
    return Image.open(image_path).convert('RGB'),{'image_id':row['image_id'],'path':str(image_path),
        'file_sha256':sha(image_path),'manifest_sha256':sha(manifest),'record_index':index}
