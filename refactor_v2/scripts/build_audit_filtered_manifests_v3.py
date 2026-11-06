"""Create development manifests while preserving historical source artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict-decoder', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / '.evaluation_runtime/data/fracatlas_component_clean'
    output = root / '.evaluation_runtime/data/fracatlas_audit_filtered_v3'
    if args.strict_decoder:
        output = root / '.evaluation_runtime/data/fracatlas_strict_development_v3'
    output.mkdir(exist_ok=True)
    manifest = json.loads((source / 'component_split_manifest.json').read_text())
    excluded = {image for group in manifest['contradictory_label_components'] for image in group}
    if args.strict_decoder:
        decoder = json.loads((root / 'refactor_v2/reports/strict_decoder_quarantine_impact_v3.json').read_text())
        excluded.update(decoder['strict_decode_failed_ids'])
    report = {'status': 'development_only_not_final_seal',
              'excluded_images': sorted(excluded),
              'reason': 'Unresolved fracture labels; strict decoder failures additionally excluded when requested',
              'strict_decoder': args.strict_decoder,
              'splits': {}, 'remaining_gates': ['near-duplicate audit completeness',
              'pretraining provenance', 'nested evaluation design', 'decoder policy']}
    for split in ('train', 'model_val', 'calibration', 'test'):
        path = source / ('vqa_' + split + '.json')
        rows = json.loads(path.read_text())
        selected = [row for row in rows if row['image_id'] not in excluded]
        target = output / path.name
        target.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
        report['splits'][split] = {'images': len({row['image_id'] for row in selected}),
                                  'qa': len(selected), 'removed_qa': len(rows) - len(selected),
                                  'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (output / 'audit_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
