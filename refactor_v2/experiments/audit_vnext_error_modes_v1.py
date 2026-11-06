"""Descriptive fixed0.5 error/proposal audit from saved V outputs only."""
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';O=R/'vnext_local_tokens_v1';P=R/'step4_localization_v3_1';S=R/'step7_synthesis_v3_1'
ARMS=('B','real_aux','placebo_aux','grid_aux','real_noaux')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(partial=False):
    for p,h in read(S/'synthesis_closure.json')['artifacts'].items():assert sha(Path(p))==h,p
    old=read(S/'localization_saved.json')
    for p,h in old['sources'].items():assert sha(Path(p))==h,p
    localization={r['image_id']:r for r in old['rows']}
    reference=[r for r in read(P/'model_val.json') if r['question']=='Is there a fracture in this X-ray?']
    ids=[r['image_id'] for r in reference];assert len(ids)==len(set(ids))==321
    available=[a for a in ARMS if (O/f'run_{a}/epoch4_evaluation.json').exists()]
    assert 'B' in available
    if not partial:assert available==list(ARMS)
    rows=[];sources={};predictions={};replay_status={}
    for arm in available:
        path=O/f'run_{arm}/epoch4_evaluation.json';d=read(path);rs=[r for r in d['records'] if r['question']=='Is there a fracture in this X-ray?']
        assert [r['image_id'] for r in rs]==ids and [r['gt']=='yes' for r in rs]==[bool(r['fractured']) for r in reference]
        predictions[arm]={r['image_id']:r['p_yes']>=.5 for r in rs};sources[str(path)]=sha(path)
        replay=O/f'run_{arm}/fresh_replay.json';replay_status[arm]=replay.exists() and read(replay)['passed']
    for r in reference:
        info=localization[r['image_id']];assert info['positive']==bool(r['fractured'])
        rows.append({'image_id':r['image_id'],'positive':bool(r['fractured']),'hardware':r['hardware'],'fracture_count':r['fracture_count'],'proposal_count':info['proposal_count'],'real_best_IoU':info['best_IoU'],'predicted_positive':{a:predictions[a][r['image_id']] for a in available}})
    def summarize(group):
        positives=[r for r in group if r['positive']];negatives=[r for r in group if not r['positive']]
        results={}
        for arm in available:
            tp=sum(r['predicted_positive'][arm] for r in positives);tn=sum(not r['predicted_positive'][arm] for r in negatives)
            results[arm]={'TP':tp,'FN':len(positives)-tp,'TN':tn,'FP':len(negatives)-tn,'sensitivity':tp/len(positives) if positives else None,'specificity':tn/len(negatives) if negatives else None}
        return {'images':len(group),'positives':len(positives),'negatives':len(negatives),'arms':results}
    groups={'all':rows,'no_hardware':[r for r in rows if not r['hardware']],'hardware':[r for r in rows if r['hardware']],'single_fracture':[r for r in rows if r['positive'] and r['fracture_count']==1],'multiple_fractures':[r for r in rows if r['positive'] and r['fracture_count']>1],'proposals_empty':[r for r in rows if not r['proposal_count']],'proposals_present':[r for r in rows if r['proposal_count']],'positive_proposal_IoU_ge05':[r for r in rows if r['positive'] and r['real_best_IoU'] is not None and r['real_best_IoU']>=.5]}
    base_fn=[r for r in rows if r['positive'] and not r['predicted_positive']['B']];base_tp=[r for r in rows if r['positive'] and r['predicted_positive']['B']]
    rescue={a:{'baseline_FN_count':len(base_fn),'rescued_FN_ids':[r['image_id'] for r in base_fn if r['predicted_positive'][a]],'lost_baseline_TP_ids':[r['image_id'] for r in base_tp if not r['predicted_positive'][a]],'rescued_with_real_IoU_ge05_ids':[r['image_id'] for r in base_fn if r['predicted_positive'][a] and r['real_best_IoU'] is not None and r['real_best_IoU']>=.5]} for a in available if a!='B'}
    name='error_modes_final.json' if not partial else f'error_modes_partial_{len(available)}arms.json'
    evidence={'scope':'Descriptive inspected model-validation saved-output audit. Fixed raw0.5 scores may differ in calibration; rescue counts are not same-specificity causal comparisons. Small subgroup denominators are reported; no clinical extrapolation or tuning.','partial':partial,'available_arms':available,'fresh_replay_passed':replay_status,'groups':{n:summarize(g) for n,g in groups.items()},'baseline_FN_rescue':rescue,'records':rows,'dependencies':sources,'localization_source_sha256':sha(S/'localization_saved.json'),'source_sha256':sha(Path(__file__))}
    with (O/name).open('x') as f:json.dump(evidence,f,indent=2)
    print(json.dumps({'artifact':name,'rescue':rescue,'all':evidence['groups']['all']}))
if __name__=='__main__':main('--partial' in sys.argv)
