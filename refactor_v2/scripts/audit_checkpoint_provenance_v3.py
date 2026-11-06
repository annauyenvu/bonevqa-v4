"""Extract checkpoint provenance without constructing a model."""
import hashlib
import json
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / '.evaluation_runtime/checkpoints/stage2_pmcvqa_full/best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    report = {
        'checkpoint': str(checkpoint),
        'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'payload_keys': list(payload),
        'metadata': {key: value for key, value in payload.items()
                     if key in ('config', 'cfg', 'epoch', 'metrics', 'data_manifest', 'answer_vocab')},
        'status': 'requires_manifest_and_source_verification',
    }
    output = root / 'refactor_v2/reports/checkpoint_provenance_v3.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({key: value for key, value in report.items() if key != 'metadata'}, indent=2))
    for key in ('config', 'cfg'):
        if key in report['metadata']:
            print(json.dumps(report['metadata'][key], indent=2, default=str))


if __name__ == '__main__':
    main()
