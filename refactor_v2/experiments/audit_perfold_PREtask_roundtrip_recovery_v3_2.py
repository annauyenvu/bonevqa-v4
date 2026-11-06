"""Audit completed exports/replay while preserving original failed driver state."""
import gc
import json
import time
from pathlib import Path
import torch
from src.checkpointing import validate_manifest
from experiments.export_perfold_PREtask_roundtrip_v3_1 import OUT, UPSTREAM, inputs, validate
from experiments.run_T_bounded_current_v3_1 import hashes
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump


def main():
    started=time.perf_counter();target=ROOT/'refactor_v2/reports/perfold_PREtask_roundtrip_recovery_v3_2.json'
    assert not target.exists()
    protocol=validate();exports=json.loads((OUT/'exports.json').read_text())
    replay=json.loads((OUT/'fresh_process_replay.json').read_text());old_progress=json.loads((OUT/'progress.json').read_text())
    assert old_progress['status']=='failed_preserved_no_retry' and replay['passed']
    assert set(exports)==set(replay['folds'])=={str(i) for i in range(5)}
    shared=None;checkpoints={};counts={}
    for fold in range(5):
        key=str(fold);path=OUT/f'fold_{fold}_PREtask.pt';record=exports[key]
        assert sha(path)==record['checkpoint_sha256']==replay['folds'][key]['checkpoint_sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        cfg,modelcfg,vocab,public,meta=inputs(fold)
        assert payload['answer_vocab']==vocab and meta==record['TRAIN_binding']==payload['extra']['TRAIN_binding']
        assert payload['extra']['fold']==fold and payload['extra']['quality_optimizer_updates']==0
        assert not payload['extra']['scientific_execution_released']
        assert payload['extra']['protocol_sha256']==sha(OUT/'protocol.json')
        assert all(payload['config'][k]==v for k,v in modelcfg.items())
        validate_manifest(payload,payload['config'],vocab)
        assert len(payload['state'])==263 and finite_tree(payload['state'])
        h=hashes(payload['state']);assert h==record['all263_tensor_sha256']
        common={k:v for k,v in h.items() if not k.startswith(('closed_head.','fracture_head.'))}
        if shared is None:shared=common
        else:assert common==shared
        assert len(record['public_TRAIN_predictions'])==4
        assert replay['folds'][key]['all263_tensors_and_all4_public_predictions_exact']
        assert replay['folds'][key]['vocabulary_size']==len(vocab)
        checkpoints[key]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
        counts[key]={'vocabulary_size':len(vocab),'TRAIN_positive_weight':meta['TRAIN_positive_weight'],
            'TRAIN_canonical_counts':meta['TRAIN_canonical_counts'],'all263_tensor_hashes_verified':True,
            'strict_manifest_and_fresh_process_public_roundtrip':True}
        del payload;gc.collect()
    assert len(list(OUT.glob('*.pt')))==5 and not list(OUT.glob('*optimizer*'))
    old=json.loads((ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/report.json').read_text())
    assert exports['0']['all263_tensor_sha256']==old['all263_initial_tensor_hashes']
    report={'passed':True,'scope':'Recovery audit of zero-QA initial checkpoints, no quality training release',
        'script_sha256':sha(Path(__file__).resolve()),'protocol_sha256':sha(OUT/'protocol.json'),
        'exports_sha256':sha(OUT/'exports.json'),'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
        'original_failed_driver_progress_sha256':sha(OUT/'progress.json'),
        'replay_environment_failure_record_sha256':sha(OUT/'replay_launch_failure.json'),
        'original_failed_driver_progress_preserved':True,
        'recovery':'Only original --replay separately executed with shared-memory process permission; same offline CPU8 runtime, no model/data/source modification or duplicate export',
        'environment_cause_limit':'Observed OpenMP SHM failure and successful separately permitted replay; exact OS root cause not exhaustively isolated',
        'upstream_sha256':sha(UPSTREAM),'checkpoints':checkpoints,'folds':counts,
        'shared_non_task_head_tensor_count':len(shared),'shared_non_task_head_hashes_exact_across_folds':True,
        'fold0_all263_equal_previous125vocab_gate':True,
        'within_fold_B_uniform_parity':'One common initial file per fold available to both proposed arms; no trained arms claimed',
        'fresh_process_replay_constructor_seed':90211,
        'replayed_TRAIN_prediction_records':20,
        'auditor_neural_model_instantiations':0,'optimizer_updates':0,
        'modelval_calibration_outer_inference_used':False,
        'limits':['Only4 TRAIN inference fixtures per fold, not a quality evaluation',
            'Zero-QA checkpoint reload is not optimizer/RNG training-resume certification',
            'No post-training weights or per-fold learning efficacy certified',
            'Original driver failed before recording an end-to-end elapsed report; do not invent its total runtime'],
        'audit_elapsed_seconds':time.perf_counter()-started,
        'long_training_authorized':False,'step3_complete':False,'Final_Solution_selected':False,'production_released':False}
    dump(target,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
