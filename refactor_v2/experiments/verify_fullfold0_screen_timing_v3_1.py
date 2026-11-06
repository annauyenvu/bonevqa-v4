"""No-model independent checks of fixed pair design, schedules and timing artifacts."""
import json
import math
from pathlib import Path
from src.context_controls import ComponentBatchSampler,load_component_lookup,component_disjoint_donors
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT,sha,dump


def main():
    f=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1';t=ROOT/'refactor_v2/reports/fullfold0_accumulation_timing_v3_1'
    out=f/'independent_design_timing_verification.json';assert not out.exists()
    p=json.loads((f/'protocol.json').read_text());timing=json.loads((t/'report.json').read_text());tp=json.loads((t/'protocol.json').read_text())
    for name,h in p['dependency_sha256'].items():assert sha(Path(name))==h
    assert timing['passed'] and timing['screen_protocol_sha256']==tp['screen_protocol_sha256']==sha(f/'protocol.json')
    assert timing['protocol_sha256']==sha(t/'protocol.json') and tp['script_sha256']==sha(ROOT/'refactor_v2/experiments/time_fullfold0_accumulation_v3_1.py')
    schedules=json.loads((f/'component_schedule.json').read_text());assert sha(f/'component_schedule.json')==p['schedule_sha256']
    configs=[json.loads(Path(p['configs'][m]['path']).read_text()) for m in p['arm_order']]
    for mode,c in zip(p['arm_order'],configs):
        assert sha(Path(p['configs'][mode]['path']))==p['configs'][mode]['sha256']
        assert c['recipe_status']=='candidate_not_launch_sealed' and c['seed']==42 and c['train']['epochs']==2
        assert sha(Path(c['init_from']))==p['initial_checkpoint']['sha256']
        assert c['data']['loader_kwargs']['sealed_records_by_split']['test'] is None and c['data']['loader_kwargs']['sealed_records_by_split']['calibration'] is None
    a,b=configs;assert all(a[k]==b[k] for k in ('model','train','data','init_from','checkpoint_allow_mismatch_prefixes'))
    rows=json.loads(Path(a['data']['loader_kwargs']['sealed_records_by_split']['train']).read_text())
    bc=a['train']['component_batching'];lookup,_=load_component_lookup(bc['manifest'],bc['seal'],ROOT)
    sampler=ComponentBatchSampler([lookup[r['image_id']] for r in rows],4,42)
    for epoch,s in enumerate(schedules):
        sampler.set_epoch(epoch);actual=list(sampler);assert actual==s['ordered_indices'] and len(actual)==2876
        assert sorted(i for batch in actual for i in batch)==list(range(11502))
        assert math.ceil(len(actual)/4)==s['optimizer_updates']==719
        for batch in actual:component_disjoint_donors([lookup[rows[i]['image_id']] for i in batch])
    assert tp['TRAIN_microbatch_indices']==schedules[0]['ordered_indices'][:16]
    expected_open=sum(rows[i]['answer_type']=='OPEN' for batch in tp['TRAIN_microbatch_indices'] for i in batch)
    projections={}
    for mode,result in timing['results'].items():
        assert result==json.loads((t/(mode+'.json')).read_text())
        assert result['TRAIN_rows']==64 and result['TRAIN_OPEN_rows']==expected_open==32
        assert result['optimizer_updates']==4 and len(result['update_seconds'])==4
        assert all(math.isfinite(x) and x>0 for x in result['update_seconds'])
        assert sum(result['update_seconds'])==result['TRAIN_loop_seconds']
        assert result['TRAIN_loop_seconds']/4==result['mean_seconds_per_accumulated_update']
        assert all(math.isfinite(x) and x>0 for x in result['selected_gradient_max'].values())
        assert all(math.isfinite(x) and x>0 for x in result['selected_actual_updates'].values())
        assert result['all_trainable_states_finite'] and result['all_updated_weights_discarded'] and result['checkpoints_saved']==0
        projections[mode]=result['mean_seconds_per_accumulated_update']*1438
    assert timing['total_disposable_optimizer_updates']==8 and timing['quality_training_updates']==0
    assert not list(t.rglob('*.pt')) and not timing['modelval_calibration_test_used']
    audit={'passed':True,'script_sha256':sha(Path(__file__).resolve()),'screen_protocol_sha256':sha(f/'protocol.json'),
        'timing_report_sha256':sha(t/'report.json'),'timing_protocol_sha256':sha(t/'protocol.json'),
        'source_config_initial_schedule_hashes_and_no_drop_parity_verified':True,
        'timing_counters_scalar_arithmetic_and_gradient_update_records_verified':True,
        'TRAIN_only_linear_projection_seconds_each':projections,
        'TRAIN_only_linear_projection_seconds_pair':sum(projections.values()),
        'planning_estimate_after_launch':'3–5 awake hours for this fixed pair including validation/replay and overhead; uncertain extrapolation, not total-step3 ETA',
        'limits':['Only first64TRAINrows measured; thermals,CPU contention,sleep and token lengths may change rate',
                  'No independent neural benchmark replay; artifact/arithmetic audit only'],
        'neural_model_instantiations':0,'optimizer_updates':0,'quality_training_started':False,
        'long_training_authorized':False,'step3_complete':False}
    dump(out,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
