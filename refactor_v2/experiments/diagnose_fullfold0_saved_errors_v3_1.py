"""Post-hoc development error description from existing labels/predictions only.

No model, optimization, threshold search, relabeling or clinical subtype claims.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import string

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
BASE=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
OUT=ROOT/'refactor_v2/reports/fullfold0_saved_error_diagnostic_v3_1'
MODES=('token_mean','sequence_uniform')
CANONICAL='Is there a fracture in this X-ray?'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def normalize(text):
    text=re.sub(r'\b(a|an|the)\b',' ',(text or '').lower().strip())
    return ' '.join(''.join(c for c in text if c not in set(string.punctuation)).split())


def correct(row):
    pred,gt=normalize(row['pred']),normalize(row['gt'])
    if row['answer_type']=='CLOSED':
        pred='yes' if pred.startswith('yes') else 'no' if pred.startswith('no') else pred
        gt='yes' if gt.startswith('yes') else 'no' if gt.startswith('no') else gt
    return pred==gt


def main():
    assert not OUT.exists(), 'Preserve existing diagnostic'
    audit=read(BASE/'independent_artifact_audit.json')
    assert audit['passed'] and audit['report_sha256']==sha(BASE/'report.json')
    parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
    protocol=read(parent/'protocol.json')
    cfg=read(protocol['configs'][MODES[0]]['path'])
    paths=cfg['data']['loader_kwargs']['sealed_records_by_split']
    training,val=read(paths['train']),read(paths['model_val'])
    assert len(training)==11502 and len(val)==1432
    source_paths=[Path(__file__),BASE/'report.json',BASE/'independent_artifact_audit.json',
                  Path(paths['train']),Path(paths['model_val'])]
    frequencies=Counter(normalize(r['answer']) for r in training)
    consistency={}
    for name,rows in [('TRAIN',training),('modelval',val)]:
        canonical=[r for r in rows if r['question']==CANONICAL]
        by_image=defaultdict(list)
        for row in rows:
            by_image[row['image_id']].append(row)
        mismatches=[]
        for row in canonical:
            if normalize(row['answer'])!=('yes' if row['fractured']==1 else 'no'):
                mismatches.append({'image_id':row['image_id'],'kind':'canonical_answer_vs_existing_fractured_field'})
            if bool(row['fractured'])!=(int(row['fracture_count'])>0):
                mismatches.append({'image_id':row['image_id'],'kind':'fractured_vs_existing_count_positive'})
        inconsistent=[]
        for image,records in by_image.items():
            for field in ['fractured','fracture_count','region','hardware','views']:
                if len({json.dumps(r[field],sort_keys=True) for r in records})!=1:
                    inconsistent.append({'image_id':image,'field':field})
        consistency[name]={'canonical_rows':len(canonical),'cross_field_mismatches':mismatches,
                           'same_image_metadata_disagreements':inconsistent}
    canonical_truth={r['image_id']:r for r in val if r['question']==CANONICAL}
    results={};fn_sets={}
    identities=[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in val]
    for mode in MODES:
        path=BASE/(mode+'.json');source_paths.append(path)
        assert sha(path)==read(BASE/'report.json')['prediction_sha256'][mode]
        rows=read(path)['records']
        assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==identities
        by_image=defaultdict(list);per_question={};strata=defaultdict(lambda:defaultdict(Counter))
        for row in rows:
            by_image[row['image_id']].append(row)
            summary=per_question.setdefault(row['question'],{'n':0,'correct':0,'errors':0})
            summary['n']+=1;summary['correct']+=int(correct(row));summary['errors']+=int(not correct(row))
        cases=[];scores=defaultdict(list);fn=set()
        for row in rows:
            if row['question']!=CANONICAL:
                continue
            gt=normalize(row['gt'])=='yes';pred=row['p_yes']>=.5
            outcome='TP' if gt and pred else 'FN' if gt else 'FP' if pred else 'TN'
            scores[outcome].append(row['p_yes'])
            truth=canonical_truth[row['image_id']]
            for field in ['region','hardware','fracture_count','views']:
                # Exact existing field value; no inferred clinical subgroup.
                label=json.dumps(truth[field],sort_keys=True)
                strata[field][label][outcome]+=1
            if outcome=='FN':
                fn.add(row['image_id'])
                other=[r for r in by_image[row['image_id']] if r['question']!=CANONICAL]
                cases.append({'image_id':row['image_id'],'p_yes':row['p_yes'],
                    'existing_metadata':{k:truth[k] for k in ['region','hardware','fracture_count','views','width','height']},
                    'other_existing_QA':[{'question':r['question'],'gt':r['gt'],'pred':r['pred'],
                        'correct':correct(r),'normalized_GT_answer_count_in_fullTRAIN':frequencies[normalize(r['gt'])]}
                        for r in other]})
        fn_sets[mode]=fn
        score_summary={key:{'n':len(values),'minimum':min(values),'maximum':max(values),
                           'mean':sum(values)/len(values),'sorted_p_yes':sorted(values)} for key,values in scores.items()}
        results[mode]={'per_question_exact_accuracy_counts':per_question,
            'canonical_existing_metadata_strata':{field:{label:dict(counts) for label,counts in values.items()}
                                                 for field,values in strata.items()},
            'canonical_score_distributions_at_fixed_raw0_5':score_summary,
            'false_negative_cases':cases}
    common=fn_sets[MODES[0]]&fn_sets[MODES[1]]
    report={'scope':'Post-hoc descriptive development diagnostic; not causal/clinical confirmation',
        'source_sha256':{str(p):sha(p) for p in source_paths},
        'existing_label_consistency':consistency,'results':results,
        'FN_overlap':{'common':sorted(common),'native_only':sorted(fn_sets[MODES[0]]-common),
                      'uniform_only':sorted(fn_sets[MODES[1]]-common)},
        'limits':['Label-field consistency is not clinical ground-truth adjudication',
            'Small, overlapping existing metadata strata are descriptive only; no new clinical labels',
            'No threshold optimization, new predictions, TRAIN fitting, calibration/test access or relabeling',
            'Attribution to architecture, loss, pretraining or image subtype is not established',
            'Development-informed fold0, one training seed; do not use these strata to claim independent performance'],
        'model_instantiations':0,'optimizer_updates':0,'threshold_searches':0,
        'manifest_or_label_changes':0,'quality_training_released':False,'step3_complete':False}
    OUT.mkdir()
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report_sha256':sha(OUT/'report.json'),'FN_common':len(common),
        'FN_each':{m:len(v) for m,v in fn_sets.items()},'label_consistency':consistency,
        'models':0,'updates':0,'threshold_searches':0}))


if __name__=='__main__':
    main()
