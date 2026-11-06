"""Seal only after actual visual review and final interpretation; never trains."""
import json
from pathlib import Path
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.prepare_lapa_pilot_v1 import sha

OUT=ROOT/'reports/favp_bounded_quality_v1'


def main():
    assert not (OUT/'closure_seal.json').exists(),'Closure already exists; do not overwrite'
    assert (OUT/'execution_completed.json').exists() and (OUT/'analysis_generated.json').exists()
    completed=json.loads((OUT/'completed.json').read_text())
    assert completed['updates']==64 and completed['epochs']==2 and completed['frozen_state_unchanged']
    assert all(completed['changed_groups'].values())
    replay=json.loads((OUT/'replay.json').read_text())
    assert replay['records']==8 and replay['exact_tokens_answers_metrics']
    assert replay['checkpoint_sha256']==sha(OUT/'LAST.pt')
    release=json.loads((OUT/'execution_release.json').read_text())
    for p,h in release['dependencies'].items():assert sha(Path(p))==h,p
    audit=json.loads((OUT/'independent_audit.json').read_text())
    assert audit['all_three_saved_output_audits_passed']
    for p,h in audit['sources'].items():assert sha(Path(p))==h,p
    bootstrap=json.loads((OUT/'paired_component_bootstrap.json').read_text())
    assert bootstrap['seed']==20261103 and bootstrap['draws']==2000 and bootstrap['records']==8
    for p,h in bootstrap['sources'].items():assert sha(Path(p))==h,p
    figure=OUT/'learning_curves.png'
    review=json.loads((OUT/'figure_visual_review.json').read_text())
    assert review['visually_reviewed'] is True and review['figure_sha256']==sha(figure)
    interpretation=OUT/'FINAL_INTERPRETATION.md'
    assert interpretation.exists() and len(interpretation.read_text().strip())>200
    supplementary=ROOT/'reports/favp_constant_baseline_v1'
    comparison=json.loads((supplementary/'comparison.json').read_text())
    assert comparison['records']==8
    for p,h in comparison['sources'].items():assert sha(Path(p))==h,p
    files=[OUT/n for n in ['execution_release.json','coordinator.json','coordinator_progress.json','quality.log','replay.log',
        'PRE.json','epoch1.json','epoch2.json','LAST.pt','completed.json','replay.json','execution_completed.json',
        'analysis.log','independent_audit.json','paired_component_bootstrap.json','learning_curves.png','PILOT_RESULTS.md',
        'analysis_generated.json','figure_visual_review.json','FINAL_INTERPRETATION.md']]
    files+=[Path(__file__)]
    files+=[supplementary/'definition.json',supplementary/'comparison.json',
            ROOT/'experiments/define_favp_constant_baseline_v1.py',ROOT/'experiments/audit_favp_constant_baseline_v1.py']
    (OUT/'closure_seal.json').write_text(json.dumps({'complete_bounded_adapted_pilot':True,
        'not_original_FAVP_reproduction':True,'not_clinical_evidence':True,'artifacts':{str(p):sha(p) for p in files},
        'execution_dependencies_reverified':len(release['dependencies']),'updates':64,'validation_records':8,
        'automatic_promotion':False,'next_phase':'Lens technical implementation and bounded gates'},indent=2))


if __name__=='__main__':main()
