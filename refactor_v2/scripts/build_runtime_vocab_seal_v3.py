"""Extend the immutable historical seal with runtime-normalized vocabularies."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data.sealed_manifest import load_sealed_records
from src.utils import normalize_answer


def main():
    root = Path(__file__).resolve().parents[2]
    original = root / 'refactor_v2/reports/STEP1_DATA_SEAL_V3.json'
    seal = json.loads(original.read_text())
    counts = {}
    for fold in range(5):
        directory = root / '.evaluation_runtime/data/fracatlas_component_cv_v3' / ('fold_' + str(fold))
        rows = load_sealed_records(directory / 'vqa_train.json', original, root)
        frequency = Counter(normalize_answer(row['answer']) for row in rows)
        if '' in frequency:
            raise ValueError('Empty answer after runtime normalization')
        vocabulary = sorted(frequency, key=lambda answer: (-frequency[answer], answer))
        path = directory / 'answer_vocab_runtime_normalized_v3.json'
        path.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2))
        assert set(vocabulary) == {normalize_answer(row['answer']) for row in rows}
        seal['artifacts'][path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[str(fold)] = len(vocabulary)
    seal['supersedes_for_training'] = str(original.relative_to(root))
    seal['parent_seal_sha256'] = hashlib.sha256(original.read_bytes()).hexdigest()
    seal['vocabulary_normalization'] = 'src.utils.normalize_answer'
    seal['runtime_vocab_counts'] = counts
    destination = root / 'refactor_v2/reports/STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json'
    destination.write_text(json.dumps(seal, indent=2))
    for relative, expected in seal['artifacts'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Seal verification failed: ' + relative)
    print(json.dumps({'runtime_vocab_counts': counts, 'artifacts_verified': len(seal['artifacts']),
                      'historical_seal_preserved': True}))


if __name__ == '__main__':
    main()
