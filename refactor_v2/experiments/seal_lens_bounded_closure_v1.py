"""Close only after independent evidence, actual PNG review and interpretation."""
import json
from pathlib import Path
from experiments.lens_native_common_v1 import ROOT,sha

OUT=ROOT/'reports/lens_bounded_quality_v1'


def main():
    assert not (OUT/'closure_seal.json').exists()
    release=json.loads((OUT/'execution_release.json').read_text())
    assert all(sha(Path(p))==h for p,h in release['dependencies'].items())
    completed=json.loads((OUT/'execution_completed.json').read_text())
    assert completed['all_two_arms_and_fresh_replays_complete'] and completed['matched_PRE_exact']
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_saved_token_target_metric_audits_passed']
    for p,h in audit['sources'].items():assert sha(Path(p))==h
    bootstrap=json.loads((OUT/'paired_component_bootstrap.json').read_text())
    assert bootstrap['seed']==20261104 and bootstrap['draws']==2000 and bootstrap['records']==8
    for p,h in bootstrap['sources'].items():assert sha(Path(p))==h
    review=json.loads((OUT/'figure_visual_review.json').read_text())
    assert review['visually_reviewed'] and review['figure_sha256']==sha(OUT/'learning_curves.png')
    interpretation=OUT/'FINAL_INTERPRETATION.md';assert interpretation.exists() and interpretation.stat().st_size>300
    paths=[OUT/n for n in ['execution_release.json','execution_completed.json','independent_audit.json',
        'paired_component_bootstrap.json','learning_curves.png','figure_visual_review.json','PILOT_RESULTS.md','FINAL_INTERPRETATION.md']]
    for arm in ['ce_only','lens_losses']:
        directory=OUT/arm
        worker=json.loads((directory/'completed.json').read_text())
        assert worker['updates']==12 and worker['frozen_invariance'] and worker['changed_tensor_count']==492
        replay=json.loads((directory/'replay.json').read_text());assert replay['exact_tokens_text_targets_metrics']
        assert replay['full_state_digest']==worker['final_full_state_digest']
        assert sha(directory/'LAST.pt')==worker['LAST_sha256']
        paths += [directory/n for n in ['started.json','PRE.json','stage1.json','qa_epoch1.json','LAST.json',
            'stage1_training.json','qa_epoch1_training.json','LAST_training.json','stage1.pt','qa_epoch1.pt','LAST.pt',
            'completed.json','fresh_replay.json','replay.json']]
    for file in ['audit_lens_bounded_outputs_v1.py','plot_lens_bounded_pilot_v1.py','seal_lens_bounded_closure_v1.py']:
        paths.append(ROOT/'experiments'/file)
    artifacts={str(p):sha(p) for p in paths}
    (OUT/'closure_seal.json').write_text(json.dumps({'closed_bounded_adapted_native_asset_pilot':True,
        'artifacts':artifacts,'execution_dependencies_unchanged':len(release['dependencies']),
        'scope':'one seed,28eligibleTRAIN/8reuseddevelopmentvalidation; contrastive-loss comparison conditional on same views; not full original reproduction, whole-view/combined-model ablation or clinical evidence'},indent=2))


if __name__=='__main__':main()
