"""Filter upstream training by transitive article and perceptual components."""
import hashlib
import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    data = root / 'data'
    candidates = root / '.evaluation_runtime/data/pretraining_candidates_v3'
    output = root / '.evaluation_runtime/data/pretraining_joint_groups_v3'
    output.mkdir(exist_ok=True)
    prefix = 'BONE_VQA\\BaoYen_work\\data\\'
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(first, second):
        parent[find(first)] = find(second)

    sources = {}
    for dataset in ('pmc_vqa', 'roco'):
        for split in (('train', 'test') if dataset == 'pmc_vqa' else ('train', 'val', 'test')):
            rows = json.loads((data / (prefix + dataset + '\\' + split + '.json')).read_text())
            sources[dataset + '/' + split] = rows
            for row in rows:
                path = str(data / (prefix + dataset + '\\' + row['image'].replace('/', '\\')))
                find(path)
                article = re.search(r'PMC\d+', row['image'])
                if article:
                    union(path, 'article:' + article.group())
    for name in ('cross_dataset', 'within_pmc', 'within_roco'):
        edges = json.loads((root / 'refactor_v2/reports' / (name + '_pixel_edges_v3.json')).read_text())
        for edge in edges['selected_edges']:
            union(edge['pmc'], edge['roco'])
    validation = json.loads((candidates / 'pmc_vqa_model_val_article_candidate.json').read_text())
    protected = set()
    for dataset, rows in [('pmc_vqa', validation), ('pmc_vqa', sources['pmc_vqa/test']),
                          ('roco', sources['roco/val']), ('roco', sources['roco/test'])]:
        for row in rows:
            protected.add(find(str(data / (prefix + dataset + '\\' + row['image'].replace('/', '\\')))))
    report = {'status': 'joint_groups_filtered_pending_final_verification', 'splits': {},
              'policy': 'Exclude complete article/perceptual components touching any protected evaluation cohort.'}
    for dataset in ('pmc_vqa', 'roco'):
        rows = sources[dataset + '/train']
        selected = [row for row in rows if find(str(data / (prefix + dataset + '\\' + row['image'].replace('/', '\\')))) not in protected]
        path = output / (dataset + '_train.json')
        path.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
        report['splits'][dataset] = {'source_rows': len(rows), 'retained_rows': len(selected),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
