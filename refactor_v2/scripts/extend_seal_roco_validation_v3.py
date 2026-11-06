"""Pin historical ROCO validation as an alignment-selection cohort."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    previous = root / 'refactor_v2/reports/STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json'
    seal = json.loads(previous.read_text())
    source = root / 'data' / 'BONE_VQA\\BaoYen_work\\data\\roco\\val.json'
    rows = json.loads(source.read_text())
    excluded = [index for index, row in enumerate(rows) if not str(row.get('caption') or '').strip()]
    selected = [row for index, row in enumerate(rows) if index not in set(excluded)]
    target = root / '.evaluation_runtime/data/pretraining_joint_groups_v3/roco_validation_caption_valid.json'
    target.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    seal['artifacts'][target.relative_to(root).as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    seal['roco_validation_filter'] = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                     'excluded_source_indices': excluded, 'retained': len(selected)}
    seal['parent_seal_sha256'] = hashlib.sha256(previous.read_bytes()).hexdigest()
    seal['roco_validation_role'] = 'Alignment selection only, not an independent final VQA test'
    destination = root / 'refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json'
    destination.write_text(json.dumps(seal, indent=2))
    print(json.dumps({'validation_captions': len(selected), 'excluded_empty': len(excluded), 'artifacts': len(seal['artifacts'])}))


if __name__ == '__main__':
    main()
