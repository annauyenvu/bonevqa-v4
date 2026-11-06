"""Final pooled metric verification and immutable evidence/report hash index."""
import hashlib
import json
from pathlib import Path
from src.utils import compute_metrics

ROOT=Path('/Users/uyen/Downloads/BONEVQA');REPORTS=ROOT/'refactor_v2/reports'
RUN=REPORTS/'grouped_reserved_evaluation_v3_1';ANALYSIS=REPORTS/'grouped_outer_bootstrap_v3_1'
OUT=REPORTS/'GROUPED_EXPERIMENT_CLOSURE_SEAL_V3_1.json'
CANONICAL='Is there a fracture in this X-ray?'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert not OUT.exists(),'Preserve existing closure seal'
    report=read(ANALYSIS/'report.json');assert report['passed_exact_replay']
    audit=read(ANALYSIS/'independent_audit.json');assert audit['passed'] and audit['results_sha256']==sha(ANALYSIS/'results.json')
    for release in (RUN/'execution_release.json',ANALYSIS/'plan.json'):
        for p,h in read(release)['dependencies'].items():assert sha(p)==h,p
    data=[read(RUN/f'fold_{f}.json') for f in ('0','1','2','3','4')]
    points=read(ANALYSIS/'results.json')['pooled_points'];checked={}
    for arm in ('raw','calibrated'):
        rows=[r for d in data for r in d[arm+'_records']]
        clinical=[r for r in rows if r['question']==CANONICAL]
        assert len(rows)==17998 and len(clinical)==len({r['image_id'] for r in clinical})==4013
        metrics={**compute_metrics(clinical,fracture_threshold=.5),**{k:v for k,v in compute_metrics(rows,fracture_threshold=.5).items() if k in ('overall_acc','closed_acc','open_acc')}}
        for key,value in points.items():
            if key.startswith(arm+'.'):
                name=key.split('.',1)[1];assert abs(metrics[name]-value)<1e-12,(key,value,metrics[name])
                checked[key]=value
    paths=[Path(__file__).resolve(),REPORTS/'GROUPED_OUTER_RESULTS_V3_1.md',REPORTS/'STEP3_EXIT_REVIEW_V3_4.md',RUN/'execution_release.json',RUN/'report.json',ANALYSIS/'plan.json',ANALYSIS/'results.json',ANALYSIS/'report.json',ANALYSIS/'independent_audit.json',REPORTS/'grouped_evaluation_bindings_v3_1/bindings.json',REPORTS/'grouped_evaluation_bindings_v3_1/independent_audit.json',REPORTS/'STEP3_TECHNICAL_REFERENCE_SEAL_V3.json']
    for f in ('0','1','2','3','4'):
        independent=RUN/f'fold_{f}_independent_audit.json';a=read(independent)
        assert a['passed'] and a['prediction_sha256']==sha(RUN/f'fold_{f}.json') and a['replay_sha256']==sha(RUN/f'fold_{f}_replay.json')
        paths += [RUN/f'fold_{f}.json',RUN/f'fold_{f}_replay.json',RUN/f'fold_{f}_timing.json',independent]
    result={'passed':True,'pooled_metrics_independently_crosschecked_with_shared_metric_implementation':checked,'artifacts':{str(p):sha(p) for p in paths},'experiment_complete':True,'quality_gate':'No promotion: limited sensitivity; calibrated primary BA worse','step3_complete':False,'production_released':False,'remaining_scope':'Original seven-step exit reconciliation and user direction for any materially new study'}
    with OUT.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'sealed_artifacts':len(paths),'pooled_metrics_checked':len(checked),'experiment_complete':True,'step3_complete':False,'production_released':False}))

if __name__=='__main__':main()
