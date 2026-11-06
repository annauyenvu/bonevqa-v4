"""Build conservative component folds for development evaluation."""
import hashlib
import json
import random
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / '.evaluation_runtime/data/fracatlas_component_clean'
    reports = root / 'refactor_v2/reports'
    output = root / '.evaluation_runtime/data/fracatlas_component_cv_v3'
    output.mkdir(exist_ok=True)
    historical = json.loads((source / 'component_split_manifest.json').read_text())
    quarantine = json.loads((reports / 'strict_decoder_quarantine_impact_v3.json').read_text())
    excluded = set(quarantine['strict_decode_failed_ids']) | set(quarantine['contradictory_ids'])
    records = {}
    for split in ('train', 'val', 'test'):
        for row in json.loads((source / ('vqa_' + split + '.json')).read_text()):
            records.setdefault(row['image_id'], []).append(row)
    parent = {image: image for image in records}

    def find(image):
        while parent[image] != image:
            parent[image] = parent[parent[image]]
            image = parent[image]
        return image

    def union(left, right):
        if left in parent and right in parent:
            parent[find(left)] = find(right)

    for component in historical['components']:
        for image in component[1:]:
            union(component[0], image)
    edges = json.loads((reports / 'fracatlas_verified_candidate_edges_v3.json').read_text())
    for pair in edges['selected_edges']:
        union(Path(pair['left'].replace('\\', '/')).stem,
              Path(pair['right'].replace('\\', '/')).stem)
    groups = {}
    for image in records:
        groups.setdefault(find(image), []).append(image)
    rejected = []
    accepted = []
    for component in groups.values():
        labels = {records[image][0]['fractured'] for image in component}
        if len(labels) > 1 or excluded.intersection(component):
            rejected.extend(component)
        else:
            accepted.append(sorted(component))
    rng = random.Random(20261017)
    folds = [[] for _ in range(5)]
    for label in (0, 1):
        stratum = [group for group in accepted if records[group[0]][0]['fractured'] == label]
        rng.shuffle(stratum)
        stratum.sort(key=len, reverse=True)
        sizes = [0] * 5
        for group in stratum:
            index = min(range(5), key=lambda number: sizes[number])
            folds[index].extend(group)
            sizes[index] += len(group)
    artifacts = {}
    for outer in range(5):
        heldout = set(folds[outer])
        inner = [group for group in accepted if not heldout.intersection(group)]
        rng.shuffle(inner)
        assignments = {'train': [], 'model_val': [], 'calibration': [], 'test': sorted(heldout)}
        for label in (0, 1):
            stratum = [group for group in inner if records[group[0]][0]['fractured'] == label]
            target = max(1, round(sum(map(len, stratum)) * 0.1))
            for group in stratum:
                destination = next((name for name in ('model_val', 'calibration')
                                    if sum(records[image][0]['fractured'] == label
                                           for image in assignments[name]) < target), 'train')
                assignments[destination].extend(group)
        fold_dir = output / ('fold_' + str(outer))
        fold_dir.mkdir(exist_ok=True)
        artifacts[str(outer)] = {}
        for split, images in assignments.items():
            path = fold_dir / ('vqa_' + split + '.json')
            rows = [row for image in sorted(images) for row in records[image]]
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
            artifacts[str(outer)][split] = {'images': len(images), 'qa': len(rows),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        assert sum(len(images) for images in assignments.values()) == sum(map(len, accepted))
        assert len(set().union(*(set(images) for images in assignments.values()))) == sum(map(len, accepted))
    report = {'status': 'candidate_nested_manifests_generated', 'seed': 20261017,
              'excluded_component_images': sorted(rejected), 'components': accepted,
              'folds': artifacts,
              'limitations': ['Development-informed protocol; not an untouched external validation.',
              'Detector must be retrained within each outer training partition.',
              'Initialization provenance and cross-dataset audit require separate closure.']}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'retained_images': sum(map(len, accepted)), 'excluded_images': len(rejected),
                      'components': len(accepted), 'folds': artifacts}, indent=2))


if __name__ == '__main__':
    main()
