"""Metadata-only draft, deliberately not a runnable experiment or launch seal."""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump


def main():
    output = ROOT/'refactor_v2/reports/NESTED_DESIGN_REQUIREMENTS_V3_1.json'
    assert not output.exists()
    base = ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3'
    sealpath = ROOT/'refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json'
    seal = json.loads(sealpath.read_text())
    folds = {}; outer = Counter()
    for fold in range(5):
        folder = base/f'fold_{fold}'
        detector = json.loads((folder/'detector_split_v3.json').read_text())
        components = detector['component_by_image']
        stats = {}; image_sets = {}; group_sets = {}
        for role in ('train','model_val','calibration','test'):
            p = folder/f'vqa_{role}.json'; h = sha(p)
            assert h == seal['artifacts'][str(p.relative_to(ROOT))]
            rows = json.loads(p.read_text()); ids = {r['image_id'] for r in rows}
            image_sets[role] = ids; group_sets[role] = {components[i] for i in ids}
            d_role = 'val' if role == 'model_val' else role
            assert detector['source_vqa'][d_role] == {'path':str(p),'sha256':h}
            assert set(detector['split_images'][d_role]) == ids
            stats[role] = {'manifest':str(p),'sha256':h,'QA':len(rows),
                'images':len(ids),'known_components':len(group_sets[role])}
        for a,b in combinations(image_sets,2):
            assert not image_sets[a]&image_sets[b]
            assert not group_sets[a]&group_sets[b]
        outer.update(image_sets['test'])
        vocab = folder/'answer_vocab_runtime_normalized_v3.json'
        assert sha(vocab) == seal['artifacts'][str(vocab.relative_to(ROOT))]
        stats['vocabulary'] = {'path':str(vocab),'sha256':sha(vocab),'answers':len(json.loads(vocab.read_text()))}
        stats['detector_metadata_sha256'] = sha(folder/'detector_split_v3.json')
        folds[str(fold)] = stats
    assert len(outer) == 4013 and set(outer.values()) == {1}
    evidence = ['STEP3_TECHNICAL_REFERENCE_SEAL_V3.json','ARCHITECTURE_LOSS_RUNTIME_PROTOCOL_V3_REVIEW.md',
        'FULL_MODELVAL_COVERAGE_OUTCOME_V3.md','CURRENT_GATE_LEDGER_V3_15.json',
        'full_modelval_coverage_v3_1/independent_artifact_audit.json',
        'upstream_byte_label_provenance_v3_1.json']
    result = {'status':'DRAFT_REQUIREMENTS_NOT_EXECUTABLE_OR_PREREGISTERED',
        'script_sha256':sha(Path(__file__).resolve()),'alignment_seal_sha256':sha(sealpath),
        'evidence_sha256':{name:sha(ROOT/'refactor_v2/reports'/name) for name in evidence},
        'scope':'Prospective evaluation of a development-informed pipeline; not erased historical inspection',
        'folds':folds,'outer_test_images_each_once':4013,
        'all_within_fold_image_and_known_component_roles_disjoint':True,
        'all_detector_role_metadata_parity':True,
        'proposed_minimal_comparison':['raw B native objective','B sequence-uniform objective'],
        'selection_bias_disclosure':'Uniform proposed because inspected full1432OPEN point result; not an independent discovery or new untouched cohort',
        'held_fixed':['offline CPU8 float32','original images and raw questions only',
            'T/V/L/P off; no PF','same per-fold PREtask shared tensors/new-head initialization',
            'same fold-specific TRAIN-only vocabulary','same batches/exposures/LR/horizon and LAST checkpoint',
            'all original auxiliary losses except the explicitly scoped OPEN reduction change'],
        'required_before_any_training_release':['Explicit primary endpoints, hypotheses and uncertainty rule',
            'Exact seed list, epoch/update horizon, failure rule and maximum measured CPU budget',
            'Per-fold private TRAIN adapter and public inference-bypass gates',
            'Per-fold clean common initialization and optimizer/state/source binding',
            'TRAIN-only actual timing/resource pilot; timing cannot be inferred from byte audit',
            'Frozen inner-selection rule without accessing outer results',
            'Selection and calibration chronology; no full-data calibration reused across outer folds',
            'Signed per-run execution release after all applicable technical gates'],
        'inner_model_validation':'Only within-fold model-validation selects allowed configuration; selection rule must be frozen before launch',
        'calibration':'Only within-fold calibration after candidate freeze; raw metrics retained; no fitting in this draft',
        'outer_evaluation':'Once per frozen fold/model; label as development-informed grouped evaluation, not untouched or independent clinical validation',
        'reporting':['Per-seed and per-fold QA/CLOSED/OPEN, AUROC/AUPRC, sensitivity/specificity, BA,Brier/ECE',
            'Actual optimizer updates, image/QA exposure, module gradients/routing and measured wall time',
            'Matched paired component uncertainty and all negative/inconclusive results',
            'Fold-aware out-of-fold calibration metrics; no pooling to refit a global threshold after viewing outer results'],
        'V_L_expansion':'Not part of this minimal draft; full TRAIN-only detector/proposals per fold and new efficacy rationale required',
        'positive_gain_required_to_report_completed_negative_screen':False,
        'prior_bounded_screens_complete':True,'original_CPU_technical_reference_complete':True,
        'future_quality_gain_or_full_step3_complete':False,
        'model_calls':0,'training_updates':0,'manifest_changes':0,
        'calibration_or_outer_model_inference':False,
        'long_training_authorized':False,'Final_Solution_selected':False,'production_released':False}
    dump(output,result)
    print(json.dumps({'status':result['status'],'fold_counts':{f:{r:v[r]['QA'] for r in ('train','model_val','calibration','test')} for f,v in folds.items()},
        'all4013_outer_once':True,'output_sha256':sha(output),'training_authorized':False},indent=2))


if __name__ == '__main__':
    main()
