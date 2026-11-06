"""Independent stdlib-only recount from saved outputs; no inference or fitting."""
import hashlib,itertools,json,math,re,string,sys
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1';CAN='Is there a fracture in this X-ray?'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(text):return ' '.join(''.join(c for c in re.sub(r'\b(a|an|the)\b',' ',text.lower().strip()) if c not in string.punctuation).split())
def recount(rows):
    canonical=[r for r in rows if r['question']==CAN];assert len(canonical)==len({r['image_id'] for r in canonical})
    assert all(r['gt'] in ('yes','no') and math.isfinite(r['p_yes']) for r in canonical)
    P=sum(r['gt']=='yes' for r in canonical);N=len(canonical)-P;assert P and N
    tp=sum(r['gt']=='yes' and r['p_yes']>=.5 for r in canonical);tn=sum(r['gt']=='no' and r['p_yes']<.5 for r in canonical)
    ap=0.;seen=positives=0
    for score,group in itertools.groupby(sorted(canonical,key=lambda r:-r['p_yes']),key=lambda r:r['p_yes']):
        g=list(group);p=sum(r['gt']=='yes' for r in g);seen+=len(g);positives+=p;ap+=(p/P)*(positives/seen)
    count=0;sum_rank=0.
    for score,group in itertools.groupby(sorted(canonical,key=lambda r:r['p_yes']),key=lambda r:r['p_yes']):
        g=list(group);rank=(count+1+count+len(g))/2;sum_rank+=rank*sum(r['gt']=='yes' for r in g);count+=len(g)
    correct=sum(norm(r['pred'])==norm(r['gt']) for r in rows)
    return {'TP':tp,'FN':P-tp,'TN':tn,'FP':N-tn,'fracture_sensitivity':tp/P,'fracture_specificity':tn/N,'fracture_balanced_accuracy':(tp/P+tn/N)/2,'fracture_auprc':ap,'fracture_auroc':(sum_rank-P*(P+1)/2)/(P*N),'overall_acc':correct/len(rows),'correct_QA':correct,'QA':len(rows),'images':len(canonical)}
fold=sys.argv[1];out=O/f'fold_{fold}'/'independent_saved_output_audit.json';assert not out.exists()
result={'fold':fold,'arms':{},'sources':{},'no_model_calls':True,'no_optimizer_updates':True,'method':'stdlib tie-group AP, Mann-Whitney AUROC, direct confusion and normalized QA recount'};identity=None
for arm in ('B','local_real_noaux'):
    p=O/f'fold_{fold}'/arm/'test_predictions.json';payload=read(p);rows=payload['records'];current=[(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]
    if identity is None:identity=current
    assert current==identity
    m=recount(rows)
    for k,v in m.items():
        if k in payload['metrics']:assert abs(v-payload['metrics'][k])<1e-12,(arm,k,v,payload['metrics'][k])
    result['arms'][arm]=m;result['sources'][str(p)]=sha(p)
result['passed']=True
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
