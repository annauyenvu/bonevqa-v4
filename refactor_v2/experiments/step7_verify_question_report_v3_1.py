"""Verify saved strata against original outputs and produce readable tables."""
import json, hashlib
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1]/'reports'
O=R/'step7_synthesis_v3_1'
def main():
    target=O/'question_verification.json';assert not target.exists()
    s=json.loads((O/'question_strata.json').read_text()); verified={}; tables=[]
    for source,digest in s['sources'].items():
        p=Path(source);assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        d=json.loads(p.read_text())
        if 'seed' in d:
            cases=[(f'assessment_{d["seed"]}_{v}',d[v+'_records'],d[v+'_metrics']) for v in ('raw','calibrated')]
        else:
            arm=p.parent.name.removeprefix('run_');cases=[('modelval_'+arm,d['records'],d['metrics'])]
        for name,rows,metrics in cases:
            total=Counter();correct=Counter()
            for row in rows:
                key=(row['question'],row['answer_type']);total[key]+=1
                correct[key]+=int(str(row['gt']).strip().lower()==str(row['pred']).strip().lower())
            assert len(total)==len(s['results'][name])
            for g in s['results'][name]:
                key=(g['question'],g['answer_type']);assert (total[key],correct[key])==(g['n'],g['correct'])
            assert abs(sum(correct.values())/len(rows)-metrics['overall_acc'])<1e-12
            verified[name]={'records':len(rows),'strata':len(total),'overall_exact_matches':True}
            tables += [f'## {name}','','| Exact question | Type | Correct / N | Exact % |','|---|---|---:|---:|']
            tables += [f'| {g["question"]} | {g["answer_type"]} | {g["correct"]}/{g["n"]} | {100*g["normalized_exact"]:.2f} |' for g in s['results'][name]]
            tables += ['']
    (O/'QUESTION_RESULTS.md').write_text('# Exact-question results\n\nThese are dataset question strata, not invented clinical categories. Model-validation and assessment are separate; repeated seeds are not additional independent images.\n\n'+'\n'.join(tables))
    target.write_text(json.dumps({'passed':True,'conditions':verified,'scope':'Recount original saved records; hashes and aggregate exact accuracy verified; no inference'},indent=2))
    print('Verified',len(verified),'conditions')
if __name__=='__main__':main()
