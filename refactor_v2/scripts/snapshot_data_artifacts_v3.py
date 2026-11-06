"""Hash the current audited data artifacts without implying training readiness."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / '.evaluation_runtime/data/fracatlas_component_cv_v3').rglob('*.json'))
    paths += list((root / '.evaluation_runtime/data/pretraining_joint_groups_v3').glob('*.json'))
    paths += [root / '.evaluation_runtime/data/pretraining_candidates_v3/pmc_vqa_model_val_article_candidate.json']
    checks = {}
    for name in ('component_cv_verification_v3', 'independent_transitive_check_v3',
                 'final_vocabulary_verification_v3', 'manifest_reproduction_v3'):
        path = root / 'refactor_v2/reports' / (name + '.json')
        checks[name] = json.loads(path.read_text())['status']
        paths.append(path)
    if any(status != 'pass' for status in checks.values()):
        raise RuntimeError('Required artifact checks have not passed')
    report = {'status': 'artifact_snapshot_verified_step1_closure_pending', 'checks': checks,
              'artifacts': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(paths)},
              'remaining': ['PMC validation/test label integrity and metric contract',
                            'verify actual cached base resources load offline in step 2',
                            'bind final loader paths to audited manifests before training'],
              'limitations': ['Development-informed evaluation; no untouched external clinical validation.',
                              'No guarantee of exhaustive duplicate detection or base-pretraining independence.']}
    (root / 'refactor_v2/reports/DATA_ARTIFACT_SNAPSHOT_V3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'artifacts': len(paths), 'checks': checks, 'status': report['status']}))


if __name__ == '__main__':
    main()
