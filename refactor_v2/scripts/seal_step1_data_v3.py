"""Seal audited development data, explicitly scoped apart from runtime readiness."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    reports = root / 'refactor_v2/reports'
    checks = {}
    for name in ('component_cv_verification_v3', 'independent_transitive_check_v3',
                 'final_vocabulary_verification_v3', 'manifest_reproduction_v3'):
        checks[name] = json.loads((reports / (name + '.json')).read_text())['status']
    if any(value != 'pass' for value in checks.values()):
        raise RuntimeError('Required data check failed')
    train = root / '.evaluation_runtime/data/pretraining_joint_groups_v3'
    candidates = root / '.evaluation_runtime/data/pretraining_candidates_v3'
    paths = [train / name for name in ('pmc_vqa_train_label_filtered.json',
                                      'pmc_vqa_test_label_valid.json', 'roco_train.json')]
    paths.append(candidates / 'pmc_vqa_model_val_article_candidate.json')
    paths.extend((root / '.evaluation_runtime/data/fracatlas_component_cv_v3').rglob('*.json'))
    artifacts = {}
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        if 'pmc_vqa' in path.name:
            for row in payload:
                label = str(row.get('answer_label', '')).strip().upper().strip(' .:()')
                if label not in (row.get('choices') or {}):
                    raise RuntimeError('Invalid choice label in ' + str(path))
        artifacts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    seal = {'status': 'sealed_development_data', 'step': 1, 'checks': checks,
            'artifacts': artifacts,
            'scope': 'Data inventory, conservative known duplicate grouping, quarantine, split manifests and train-only fold vocabularies.',
            'initialization': 'Historical PMC checkpoint excluded; fresh audited base initialization and upstream training required.',
            'runtime_training_ready': False,
            'next_step_gates': ['Bind loaders to these exact artifacts and verify source hashes at runtime.',
                               'Load base weights offline and verify architecture, module routes and losses.',
                               'Train detectors separately within each fold training partition.'],
            'limitations': ['Development-informed grouped evaluation, not untouched external validation.',
                           'Heuristic duplicate search cannot certify patient independence or exhaustive duplicate removal.',
                           'Base-model training data independence is unknown.',
                           'PMC test is a filtered 49990-QA cohort; original benchmark denominator differs.',
                           'VQA-RAD remains supplementary and unsealed for image-independent evaluation.']}
    destination = reports / 'STEP1_DATA_SEAL_V3.json'
    destination.write_text(json.dumps(seal, indent=2))
    for relative, expected in artifacts.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Artifact changed during seal verification')
    print(json.dumps({'status': seal['status'], 'verified_artifacts': len(artifacts),
                      'runtime_training_ready': False}))


if __name__ == '__main__':
    main()
