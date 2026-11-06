"""Fixed two-arm bounded fullfold0 screen proposal; no execution release."""
import copy
import json
import math
from pathlib import Path
from src.context_controls import ComponentBatchSampler, load_component_lookup, component_disjoint_donors
from experiments.export_perfold_PREtask_roundtrip_v3_1 import inputs
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump

OUT=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'


def main():
    assert not OUT.exists()
    recoverypath=ROOT/'refactor_v2/reports/perfold_PREtask_roundtrip_recovery_v3_3.json'
    recovery=json.loads(recoverypath.read_text());assert recovery['passed']
    initial=recovery['checkpoints']['0'];assert sha(Path(initial['path']))==initial['sha256']
    base,modelcfg,vocab,_,_=inputs(0)
    fold=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0'
    rows=json.loads((fold/'vqa_train.json').read_text());validation=json.loads((fold/'vqa_model_val.json').read_text())
    batching=base['train']['component_batching'];lookup,_=load_component_lookup(batching['manifest'],batching['seal'],ROOT)
    sampler=ComponentBatchSampler([lookup[r['image_id']] for r in rows],4,42)
    schedules=[]
    for epoch in range(2):
        sampler.set_epoch(epoch);batches=list(sampler)
        assert sorted(i for b in batches for i in b)==list(range(11502))
        for b in batches:component_disjoint_donors([lookup[rows[i]['image_id']] for i in b])
        schedules.append({'epoch':epoch,'microbatches':len(batches),'optimizer_updates':math.ceil(len(batches)/4),'ordered_indices':batches})
    assert [s['optimizer_updates'] for s in schedules]==[719,719]
    OUT.mkdir();dump(OUT/'component_schedule.json',schedules)
    cfgs={}
    for mode in ('token_mean','sequence_uniform'):
        cfg=copy.deepcopy(base);cfg['recipe_status']='candidate_not_launch_sealed'
        cfg.update(run_name=f'fullfold0_B_{mode}_seed42_v3_1',init_from=initial['path'],allow_random_init=False,
            checkpoint_allow_mismatch_prefixes=[],figure_dir=str(OUT/(mode+'_figures')))
        cfg['model']=modelcfg
        cfg['data']['sealed_answer_vocab']=str(fold/'answer_vocab_runtime_normalized_v3.json')
        cfg['data']['loader_kwargs'].update(seal_path=str(ROOT/'refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json'),
            root=str(ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'),with_masks=False,with_boxes=False,boxes_from=None,boxes_as_masks=False,
            sealed_records_by_split={'train':str(fold/'vqa_train.json'),'model_val':str(fold/'vqa_model_val.json'),'calibration':None,'test':None})
        cfg['train'].update(epochs=2,batch_size=4,grad_accum=4,lr=1e-4,weight_decay=.01,max_grad_norm=1.,log_every=10,eval_max_samples=None)
        cfg['sequence_loss_candidate']={'mode':mode,'weights':'all_one_TRAIN_only; no rarity weighting',
            'actual_model_gate_report':str(ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/report.json')}
        target=ROOT/cfg['checkpoint_dir']/cfg['run_name'];assert not target.exists()
        path=OUT/(mode+'.json');dump(path,cfg);cfgs[mode]={'path':str(path),'sha256':sha(path),'initial_sha256':initial['sha256']}
    dependencies=[Path(__file__).resolve(),recoverypath,ROOT/'refactor_v2/reports/STEP3_TECHNICAL_REFERENCE_SEAL_V3.json',
        ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/report.json',
        ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1/independent_artifact_verification.json',
        fold/'vqa_train.json',fold/'vqa_model_val.json',fold/'answer_vocab_runtime_normalized_v3.json',
        Path(batching['manifest']),Path(batching['seal']),
        ROOT/'refactor_v2/experiments/private_TRAIN_sequence_loss_adapter_v3_1.py',
        ROOT/'refactor_v2/experiments/sequence_answer_loss_candidate_v3_1.py']
    protocol={'status':'FIXED_DESIGN_PENDING_TIMING_AND_INTEGRATION_GATES_NOT_EXECUTION_RELEASE',
        'scope':'Two-arm fullfold0 development-informed coverage/optimization screen, not five-fold confirmation',
        'dependency_sha256':{str(p):sha(p) for p in dependencies},'configs':cfgs,
        'initial_checkpoint':initial,'schedule_sha256':sha(OUT/'component_schedule.json'),
        'seed_list':[42],'arm_order':['token_mean','sequence_uniform'],'epochs_each':2,
        'TRAIN_QA':len(rows),'TRAIN_OPEN_QA':sum(r['answer_type']=='OPEN' for r in rows),'TRAIN_images':len({r['image_id'] for r in rows}),
        'modelval_QA':len(validation),'modelval_OPEN_QA':sum(r['answer_type']=='OPEN' for r in validation),
        'microbatches_each_epoch':2876,'updates_each_epoch':719,'maximum_updates_each_arm':1438,
        'primary_checkpoint':'LAST at exact epoch2/update1438, no best-epoch/seed/template/threshold selection',
        'primary_contrast':'sequence_uniform minus token_mean full modelval OPEN exact accuracy',
        'secondary_metrics':['overall/CLOSED QA accuracy','fracture AUROC','fracture AUPRC','sensitivity','specificity','BA','Brier','ECE10'],
        'clinical_endpoint':'One exact canonical fracture-presence question per321 modelval images; raw threshold0.5, no calibration fit',
        'uncertainty':'500 paired known-component bootstrap resamples,seed20261021,report paired 95% intervals; single training seed, not seed uncertainty',
        'QA_endpoint_notes':'Existing normalized exact-answer rule unchanged; no semantic rescoring or validation-target vocabulary extension',
        'evaluation':'Fresh-process strict reload and all1432 public prediction/probability replay per arm required; report numerical batching differences rather than select favorable batching',
        'decision':'Report fixed contrast and all negative/inconclusive outcomes. Positive OPEN interval alone is not clinical benefit, independent confirmation, production release or automatic more-seed/epoch authorization',
        'no_model_selection':'Fixed LAST; trainer best files may be preserved as secondary artifacts, never substituted',
        'prior_inspection':'Full1432 and original142 already informed development; original142 primary screen NOT redefined',
        'comparability':'Full TRAIN/vocabulary/class weights differ from earlier1173-QA pilot; causal contrast only within this new matched pair',
        'failure_rule':'Preserve failures and stop, no automatic retry, added epochs or partial cohort substitution',
        'remaining_release_gates':['Matched TRAIN64-row accumulation4 timing with discarded weights','Exact frozen trainer integration/native eval bypass','Independent config/initial/schedule/source verification','Separate per-run release and idle-process guard'],
        'modelval_calibration_test_for_loss_weight_fitting':False,'calibration_or_test_model_inference_allowed':False,
        'quality_training_started':False,'long_training_authorized':False,'production_released':False}
    dump(OUT/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ('status','TRAIN_QA','TRAIN_OPEN_QA','modelval_QA','modelval_OPEN_QA','maximum_updates_each_arm')},indent=2))


if __name__=='__main__':main()
