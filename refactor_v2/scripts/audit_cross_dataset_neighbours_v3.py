"""Search ROCO/PMC perceptual neighbours using a Hamming metric tree."""
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--within-pmc', action='store_true')
    parser.add_argument('--within-roco', action='store_true')
    args = parser.parse_args()
    reports = Path(__file__).resolve().parents[2] / 'refactor_v2/reports'
    rows = [row for line in (reports / 'perceptual_fingerprints_v3.jsonl').open()
            if 'phash' in (row := json.loads(line))]
    tree = None
    for row in rows:
        if ('\\pmc_vqa\\' if args.within_pmc else '\\roco\\') not in row['path']:
            continue
        value = int(row['phash'], 16)
        if tree is None:
            tree = [value, [row['path']], {}]
            continue
        node = tree
        while True:
            distance = (value ^ node[0]).bit_count()
            if not distance:
                node[1].append(row['path'])
                break
            if distance not in node[2]:
                node[2][distance] = [value, [row['path']], {}]
                break
            node = node[2][distance]
    queries = [row for row in rows if ('\\roco\\' if args.within_roco else '\\pmc_vqa\\') in row['path']]
    count = 0
    stem = 'within_roco' if args.within_roco else 'within_pmc' if args.within_pmc else 'cross_dataset'
    with (reports / (stem + '_hamming_candidates_v3.jsonl')).open('w') as stream:
        for index, row in enumerate(queries, 1):
            value = int(row['phash'], 16)
            pending = [tree] if tree else []
            while pending:
                node = pending.pop()
                distance = (value ^ node[0]).bit_count()
                if distance <= 8:
                    for path in node[1]:
                        if (args.within_pmc or args.within_roco) and path >= row['path']:
                            continue
                        stream.write(json.dumps({'pmc': row['path'], 'roco': path,
                                                 'hamming': distance}) + '\n')
                        count += 1
                pending.extend(child for edge, child in node[2].items()
                               if distance - 8 <= edge <= distance + 8)
            if index % 500 == 0 or index == len(queries):
                stream.flush()
                progress = {'processed': index, 'total': len(queries), 'candidate_pairs': count,
                            'status': 'complete' if index == len(queries) else 'running'}
                (reports / (stem + '_neighbour_progress_v3.json')).write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)


if __name__ == '__main__':
    main()
