"""Inventory local manifests and audit exact image overlap without changing data."""
import hashlib
import json
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[2]
    data = project / 'data'
    prefix = 'BONE_VQA\\BaoYen_work\\data\\'
    report = {'status': 'running', 'datasets': {}, 'overlaps': []}
    image_sets = {}
    cache = {}
    for dataset, splits in {
        'roco': ['train', 'val', 'test'],
        'pmc_vqa': ['train', 'train_lon', 'test'],
        'vqa_rad': ['train', 'test'],
    }.items():
        for split in splits:
            manifest = data / (prefix + dataset + '\\' + split + '.json')
            if not manifest.exists():
                continue
            rows = json.loads(manifest.read_text())
            images = {row['image'] for row in rows}
            hashes = set()
            missing = []
            for image in sorted(images):
                path = data / (prefix + dataset + '\\' + image.replace('/', '\\'))
                if not path.exists():
                    missing.append(image)
                    continue
                if path not in cache:
                    cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes.add(cache[path])
            key = dataset + '/' + split
            image_sets[key] = hashes
            report['datasets'][key] = {
                'qa_or_caption_rows': len(rows), 'image_paths': len(images),
                'unique_image_hashes': len(hashes), 'missing_images': missing,
                'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
            print(key, report['datasets'][key]['image_paths'], flush=True)
    keys = sorted(image_sets)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            overlap = image_sets[left] & image_sets[right]
            if overlap:
                report['overlaps'].append({'left': left, 'right': right,
                                           'shared_sha256_images': len(overlap)})
    report['status'] = 'complete'
    report['limitations'] = ['Exact bytes only; perceptual duplicates and label quality require further audit.',
                              'Checkpoint training provenance remains unverified.']
    output = project / 'refactor_v2/reports/data_inventory_exact_v3.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
