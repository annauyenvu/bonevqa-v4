"""Quarantine malformed choice labels without inventing replacement answers."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / '.evaluation_runtime/data/pretraining_joint_groups_v3/pmc_vqa_train.json'
    rows = json.loads(source.read_text())
    accepted, rejected = [], []
    for row in rows:
        label = str(row.get('answer_label', '')).strip().upper().strip(' .:()')
        choices = row.get('choices') or {}
        if label not in choices or not str(choices.get(label, '')).strip():
            rejected.append(row)
        else:
            accepted.append(row)
    output = root / '.evaluation_runtime/data/pretraining_joint_groups_v3/pmc_vqa_train_label_filtered.json'
    output.write_text(json.dumps(accepted, ensure_ascii=False, indent=2))
    report = {'status': 'label_integrity_filtered', 'accepted': len(accepted),
              'quarantined': rejected, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'policy': 'Exclude malformed labels from new train artifact; preserve source answers.'}
    (root / 'refactor_v2/reports/pmc_label_integrity_v3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'accepted': len(accepted), 'quarantined': len(rejected)}))


if __name__ == '__main__':
    main()
