"""Final closure requires explicit figure review and interpretation first."""
import json
from pathlib import Path
from experiments.prepare_lapa_pilot_v1 import sha

OUT=Path(__file__).resolve().parents[1]/'reports/lapa_native_quality_v1'


def main():
    release=json.loads((OUT/'execution_release.json').read_text())
    for p,h in release['dependencies'].items():assert sha(Path(p))==h,p
    execution=json.loads((OUT/'execution_completed.json').read_text())
    assert execution['arms']==5 and execution['full_replays']==5 and execution['updates_per_arm']==292
    review=json.loads((OUT/'figure_visual_review.json').read_text())
    assert review['visually_reviewed'] is True
    assert review['figure_sha256']==sha(OUT/'learning_curves.png')
    bootstrap=json.loads((OUT/'paired_component_bootstrap.json').read_text())
    assert bootstrap['seed']==20261102 and bootstrap['requested_draws']==2000
    for p,h in bootstrap['sources'].items():assert sha(Path(p))==h
    paths=[OUT/n for n in ['execution_release.json','execution_completed.json','paired_component_bootstrap.json',
        'REFERENCE_PILOT_RESULTS.md','FINAL_INTERPRETATION.md','learning_curves.png','figure_visual_review.json']]
    for arm in release['arms']:
        assert json.loads((OUT/arm/'replay.json').read_text())=={'records':1432,'exact':True}
        audit=json.loads((OUT/arm/'independent_metric_audit.json').read_text())
        assert audit['saved_logits_recount_passed']
        for epoch in range(1,5):
            assert sha(OUT/arm/f'epoch{epoch}.json')==audit['epochs'][str(epoch)]['source_sha256']
        paths.extend(OUT/arm/n for n in ['completed.json','replay.json','independent_metric_audit.json','epoch1.json','epoch2.json','epoch3.json','epoch4.json','LAST.pt'])
    scripts=Path(__file__).parent
    paths.extend(scripts/n for n in ['audit_lapa_native_outputs_v1.py','bootstrap_lapa_native_v1.py','report_lapa_native_v1.py','run_lapa_native_workflow_v1.py','seal_lapa_native_closure_v1.py'])
    with (OUT/'closure_seal.json').open('x') as f:json.dump({'bounded_native_backbone_pilot_complete':True,
        'figure_visually_reviewed':True,'native_reproduction_complete':False,'production_ready':False,
        'hashes':{str(p):sha(p) for p in paths}},f,indent=2)
    print('Verified and sealed',len(paths),'artifacts')


if __name__=='__main__':main()

