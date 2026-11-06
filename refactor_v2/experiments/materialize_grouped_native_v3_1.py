"""Metadata/configuration preparation only. No training or model construction."""
import copy,json,hashlib,math
from pathlib import Path
from src.context_controls import ComponentBatchSampler,load_component_lookup
ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/grouped_native_preparation_v3_1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert not OUT.exists()
base=read(ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1/token_mean.json')
design=read(ROOT/'refactor_v2/reports/NESTED_DESIGN_REQUIREMENTS_V3_1.json')
exports=read(ROOT/'refactor_v2/reports/perfold_PREtask_roundtrip_v3_1/exports.json')
b=base['train']['component_batching'];lookup,_=load_component_lookup(b['manifest'],b['seal'],ROOT)
OUT.mkdir();results={}
for fold,info in design['folds'].items():
    for role in ('train','model_val','calibration','test'):
        assert sha(info[role]['manifest'])==info[role]['sha256']
    assert sha(info['vocabulary']['path'])==info['vocabulary']['sha256']
    cfg=copy.deepcopy(base)
    cfg['run_name']=f'grouped_native_fold{fold}_seed42_v3_1'
    cfg['figure_dir']=str(OUT/f'fold_{fold}_figures')
    cfg['init_from']=str(ROOT/f'refactor_v2/reports/perfold_PREtask_roundtrip_v3_1/fold_{fold}_PREtask.pt')
    assert sha(cfg['init_from'])==exports[fold]['checkpoint_sha256']
    cfg['data']['loader_kwargs']['sealed_records_by_split']={r:info[r]['manifest'] if r in ('train','model_val') else None for r in ('train','model_val','calibration','test')}
    cfg['data']['sealed_answer_vocab']=info['vocabulary']['path']
    records=read(info['train']['manifest'])
    canonical=[r for r in records if r['question']=='Is there a fracture in this X-ray?']
    positive=sum(r['answer']=='yes' for r in canonical);negative=sum(r['answer']=='no' for r in canonical)
    assert positive+negative==len(canonical) and positive>0
    cfg['model']['fracture_positive_weight']=negative/positive
    assert cfg['model']['fracture_positive_weight']==exports[fold]['TRAIN_binding']['TRAIN_positive_weight']
    cfg['recipe_status']='prepared_not_execution_released_LAST_only'
    sampler=ComponentBatchSampler([lookup[r['image_id']] for r in records],4,42)
    schedules=[]
    for epoch in range(2):
        sampler.set_epoch(epoch);batches=list(sampler)
        assert sorted(i for batch in batches for i in batch)==list(range(len(records)))
        assert all(len({lookup[records[i]['image_id']] for i in batch})==len(batch) for batch in batches)
        schedules.append({'epoch':epoch,'ordered_indices':batches,'batches':len(batches),'updates':math.ceil(len(batches)/4)})
    if fold=='0':
        historical=read(ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1/component_schedule.json')
        assert [r['ordered_indices'] for r in schedules]==[r['ordered_indices'] for r in historical]
        a=copy.deepcopy(cfg);z=copy.deepcopy(base)
        for key in ('run_name','figure_dir','recipe_status'):a.pop(key);z.pop(key)
        assert a==z,'Fold0 semantic configuration changed'
    dump(OUT/f'fold_{fold}.json',cfg);dump(OUT/f'fold_{fold}_schedule.json',schedules)
    results[fold]={'config_sha256':sha(OUT/f'fold_{fold}.json'),'schedule_sha256':sha(OUT/f'fold_{fold}_schedule.json'),'initial_sha256':sha(cfg['init_from']),'TRAIN_QA':len(records),'total_updates':sum(e['updates'] for e in schedules),'new_training_needed':fold!='0'}
dump(OUT/'report.json',{'source_sha256':sha(__file__),'folds':results,'fold0_semantic_config_and_order_exact':True,'fold0_checkpoint_reuse_audit_pending':True,'model_calls':0,'optimizer_updates':0,'training_released':False,'remaining':'Independent configuration/schedule audit, per-fold technical gates, executable calibration/uncertainty specification and scoped releases'})
print(json.dumps(results,indent=2))
