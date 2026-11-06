"""Preserve source PMC test and materialize a label-valid audit cohort."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'data' / 'BONE_VQA\\BaoYen_work\\data\\pmc_vqa\\test.json'
    rows = json.loads(source.read_text())
    valid, excluded = [], []
    for index, row in enumerate(rows):
        label = str(row.get('answer_label', '')).strip().upper().strip(' .:()')
        choices = row.get('choices') or {}
        if label in choices and str(choices[label]).strip():
            valid.append(row)
        else:
            excluded.append(index)
    directory = root / '.evaluation_runtime/data/pretraining_joint_groups_v3'
    output = directory / 'pmc_vqa_test_label_valid.json'
    output.write_text(json.dumps(valid, ensure_ascii=False, indent=2))
    report = {'status': 'label_valid_evaluation_cohort', 'source_rows': len(rows),
              'retained_rows': len(valid), 'excluded_source_indices': excluded,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'reporting': 'Report filtered-cohort denominator and exclusions; not directly identical to original benchmark.',
              'metric_contract': 'Choice-label selection accuracy with provided choices; free-text answer accuracy must be reported separately.',
              'independence': 'Historical checkpoint prohibited; revised upstream train filtered for known article/perceptual overlap.'}
    (root / 'refactor_v2/reports/pmc_evaluation_cohort_v3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'source': len(rows), 'retained': len(valid), 'excluded': len(excluded)}))


if __name__ == '__main__':
    main()
