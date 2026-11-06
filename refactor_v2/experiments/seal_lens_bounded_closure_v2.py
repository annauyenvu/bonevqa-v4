"""Closure v2: preserved evidence checks plus independent true-wall budget guard."""
import json
import os
from pathlib import Path
from experiments.lens_native_common_v1 import ROOT,sha

OUT=ROOT/'reports/lens_bounded_quality_v1'
GUARD=ROOT/'reports/lens_bounded_wall_guard_v1'


def main():
    assert not (OUT/'closure_seal.json').exists()
    assert not (GUARD/'deadline_exceeded.json').exists(),'True-wall budget exhausted; do not claim compliant closure'
    assert not (GUARD/'failure.json').exists()
    guard=json.loads((GUARD/'started.json').read_text())
    budget=json.loads((GUARD/'completed.json').read_text())
    assert budget['execution_completed_within_true_wall'] and budget['true_wall_seconds']<=14400
    assert sha(OUT/'coordinator.json')==guard['coordinator_receipt_sha256']
    assert sha(OUT/'execution_release.json')==guard['release_sha256']
    assert os.stat(OUT/'coordinator.json').st_birthtime==guard['coordinator_created_epoch']
    assert os.stat(OUT/'execution_completed.json').st_birthtime<=guard['deadline_epoch']
    assert sha(OUT/'execution_completed.json')==budget['source_sha256']
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
    interpretation=OUT/'FINAL_INTERPRETATION.md'
    assert interpretation.exists() and interpretation.stat().st_size>300
    assert 'sleep' in interpretation.read_text().lower() or 'ngủ' in interpretation.read_text().lower(), 'Disclose hardware suspension and compute vs wall timing'
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
    paths += [GUARD/'started.json',GUARD/'completed.json',ROOT/'reports/LENS_SLEEP_WALL_BUDGET_REVIEW_V1.md']
    for file in ['audit_lens_bounded_outputs_v1.py','plot_lens_bounded_pilot_v1.py','seal_lens_bounded_closure_v1.py',
                 'seal_lens_bounded_closure_v2.py','guard_lens_true_wall_v1.py']:
        paths.append(ROOT/'experiments'/file)
    (OUT/'closure_seal.json').write_text(json.dumps({'closed_bounded_adapted_native_asset_pilot':True,
        'closure_version':2,'artifacts':{str(p):sha(p) for p in paths},'execution_dependencies_unchanged':len(release['dependencies']),
        'true_wall_seconds_including_hardware_sleep':budget['true_wall_seconds'],'true_wall_budget_seconds':14400,
        'scope':'one seed,28eligibleTRAIN/8reuseddevelopmentvalidation; contrastive-loss comparison conditional on same views; hardware sleep disclosed and independent wall budget verified; not full original reproduction, combined-model ablation or clinical evidence'},indent=2))


if __name__=='__main__':main()
