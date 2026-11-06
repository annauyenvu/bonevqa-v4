"""Fail-closed v4 closure: retained failures, recovered exact replays, original wall."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bind():
    paths=[ROOT/'refactor_v2/experiments'/n for n in ['local_noaux_head_analysis_v3.py','local_noaux_head_analysis_v4.py','finalize_local_noaux_refinement_v1.py','finalize_local_noaux_refinement_v4.py']]
    paths += [ROOT/'refactor_v2/reports'/n for n in ['LOCAL_NOAUX_HEAD_CONTEXT_REVIEW_V3_20261029.md','LOCAL_NOAUX_HEAD_ROLE_REVIEW_V4_20261029.md']]
    paths += [O/p for p in ['postprocess_failure.json','postprocess_v3_failure.json','head_context_diagnostic_v3/receipt.json','postprocess_v4/started.json']]
    for arm in ['B','local_real_noaux']:
        paths += [O/f'fold_0/{arm}/head_refit'/n for n in ['HEAD_LAST.pt','fit.json','TRAIN_features.pt']]
    payload={'wall':time.time(),'dependencies':{str(p):sha(p) for p in paths},'original_deadline':read(O/'coordinator.json')['deadline'],'existing_fitted_heads_must_not_change':True,'closure_pending':True,'first_binding_failure':'System Python lacks hashlib.file_digest; empty v1 release preserved, no scientific work repeated'}
    with (O/'head_role_recovery_v4_release_v2.json').open('x') as f:json.dump(payload,f,indent=2)
def verify_recovery():
    for rel in ['head_batch_consistency_v2/release.json','head_role_recovery_v4_release_v2.json']:
        for p,h in read(O/rel)['dependencies'].items():assert sha(Path(p))==h,p
    assert not (O/'postprocess_v4_failure.json').exists()
    done=read(O/'postprocess_v4/completed.json');start=read(O/'postprocess_v4/started.json');deadline=read(O/'coordinator.json')['deadline']
    assert done['analysis_generated'] and done['within_original_true_wall']
    assert done['deadline']==start['deadline']==deadline
    assert start['wall']<=done['wall']<=deadline
    assert (O/'analysis/results.json').stat().st_mtime<=deadline
    assert read(O/'execution_completed.json')['all_replays_passed']
    assert not (O/'failure.json').exists()
    assert not (O/'wall_guard/deadline_exceeded.json').exists()

if __name__=='__main__':
    if sys.argv[1]=='bind':bind()
    else:
        if sys.argv[1]=='seal':verify_recovery()
        original=ROOT/'refactor_v2/experiments/finalize_local_noaux_refinement_v1.py';source=original.read_text()
        old="assert not any((O/p).exists() for p in ('failure.json','postprocess_failure.json','wall_guard/deadline_exceeded.json'))"
        assert source.count(old)==1;source=source.replace(old,"verify_recovery()")
        old="guard=read(O/'wall_guard/completed.json');assert guard.get('execution_and_analysis_within_true_wall') is True"
        assert source.count(old)==1;source=source.replace(old,"assert read(O/'postprocess_v4/completed.json')['within_original_true_wall'] is True")
        old="assert read(O/'postprocess/completed.json')['analysis_generated']"
        assert source.count(old)==1;source=source.replace(old,"assert read(O/'postprocess_v4/completed.json')['analysis_generated']")
        exec(compile(source,str(original),'exec'),globals())
