"""Saved-output exact-question strata; no model fitting or new inference."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'refactor_v2/reports'
OUT = R / 'step7_synthesis_v3_1'

def main():
    OUT.mkdir(exist_ok=True)
    target = OUT / 'question_strata.json'
    assert not target.exists(), 'Preserve existing analysis'
    paths = {}
    for arm in ('B', 'V_real', 'V_placebo'):
        paths['modelval_' + arm] = R / f'step4_localization_v3_1/run_{arm}/epoch_4_evaluation.json'
    for arm in ('P_real', 'P_random', 'P_unconditioned', 'L_real', 'L_original', 'L_placebo'):
        paths['modelval_' + arm] = R / f'step56_screening_v3_1/run_{arm}/epoch4_evaluation.json'
    results = {}
    sources = {}
    for name, path in paths.items():
        data = json.loads(path.read_text())
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        results[name] = summarize(data['records'])
    for seed in range(42, 47):
        path = R / f'step6_confirmation_v3_1/assessment_seed{seed}_B.json'
        data = json.loads(path.read_text())
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for variant in ('raw', 'calibrated'):
            results[f'assessment_{seed}_{variant}'] = summarize(data[variant + '_records'])
    target.write_text(json.dumps({'scope': 'Exact supplied question strata; no invented clinical grouping; modelval and assessment kept separate', 'sources': sources, 'results': results}, indent=2))
    print('Saved', target, 'conditions', len(results))

def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['question'], row['answer_type'])].append(row)
    output = []
    for (question, kind), records in sorted(groups.items()):
        correct = sum(str(r['pred']).strip().lower() == str(r['gt']).strip().lower() for r in records)
        output.append({'question': question, 'answer_type': kind, 'n': len(records), 'correct': correct, 'normalized_exact': correct / len(records)})
    assert sum(g['n'] for g in output) == len(rows)
    return output

if __name__ == '__main__':
    main()
