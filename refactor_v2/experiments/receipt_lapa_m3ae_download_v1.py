"""Byte-level receipt after curl success; no pickle or torch load."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'references/native_assets_lapa_v1'
source = root / 'm3ae.ckpt.partial'
target = root / 'm3ae.ckpt'
receipt = root / 'm3ae_download_receipt.json'
assert not target.exists() and not receipt.exists()
assert source.stat().st_size == 1566863121
with source.open('rb') as stream:
    prefix = stream.read(16)
    stream.seek(0)
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
assert prefix.startswith(b'PK'), 'Expected checkpoint zip container, not HTML'
source.rename(target)
data = dict(file=str(target), bytes=target.stat().st_size, sha256=digest,
            drive_id='1nRXsdt8MaM-pSFZMTp0ARwTrdut8FUpF',
            author_folder='https://drive.google.com/drive/folders/1b3_kiSHH8khOQaa7pPiX_ZQnUIBxeWWn',
            complete=True, deserialized=False,
            limitation='Local SHA-256, not independently published author checksum. Training provenance and tensor compatibility not established.')
receipt.write_text(json.dumps(data, indent=2))
print(json.dumps(data))
