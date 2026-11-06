"""Close only saved-output synthesis; explicitly retain original-study gaps."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'reports';O=R/'step7_synthesis_v3_1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    target=O/'synthesis_closure.json';assert not target.exists()
    for p in (O/'question_strata.json',O/'localization_saved.json',O/'localization_component_intervals.json'):
        d=json.loads(p.read_text())
        for source,h in d['sources'].items():assert sha(source)==h,source
    q=json.loads((O/'question_verification.json').read_text());assert q['passed'] and len(q['conditions'])==19
    loc=json.loads((O/'localization_saved.json').read_text());assert len(loc['rows'])==321 and len({r['image_id'] for r in loc['rows']})==321
    bs=json.loads((O/'localization_component_intervals.json').read_text());assert sum(g['positive_images'] for g in bs['hardware_count_subgroups'])==57
    for a in ('B','V_real','V_placebo'):
        assert sum(g['TP'][a] for g in bs['hardware_count_subgroups'])==loc['positive']['predicted_positive'][a]
    historical={}
    for p in (R/'STEP3_TECHNICAL_REFERENCE_SEAL_V3.json',R/'GROUPED_EXPERIMENT_CLOSURE_SEAL_V3_1.json',R/'step4_localization_v3_1/step4_pilot_closure.json',R/'step6_confirmation_v3_1/steps56_closure.json'):
        d=json.loads(p.read_text());art=d['artifacts']
        for source,h in art.items():assert sha(source)==h,source
        historical[str(p)]={'sha256':sha(p),'verified':len(art)}
    artifacts={str(p):sha(p) for p in sorted(O.iterdir()) if p.is_file()}
    target.write_text(json.dumps({'saved_output_synthesis_complete':True,'full_original_seven_step_complete':False,'production_released':False,'scope':'Question strata, saved localization, descriptive component intervals, cohort-correct research synthesis; no new training/inference','artifacts':artifacts,'historical_seals_verified':historical,'remaining_original_requirements':['P consistency-off and trained bank-off quality controls','Current-cohort simple baselines','Independent or nested selection confirmation','Final validated model release'],'limitations':['Small exploratory subgroups','No causal hardware attribution','Bootstrap conditional on fitted models']},indent=2))
    print('Synthesis verified and sealed',len(artifacts),'artifacts')
if __name__=='__main__':main()
