"""Independent metadata audit; does not instantiate a model or release training."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/grouped_native_preparation_v3_2'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not (OUT/'independent_audit.json').exists()
report=read(OUT/'report.json');base=read(ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1/token_mean.json')
design=read(ROOT/'refactor_v2/reports/NESTED_DESIGN_REQUIREMENTS_V3_1.json')
manifest=read(base['train']['component_batching']['manifest'])
lookup={image:i for i,images in enumerate(manifest['components']) for image in images}
results={}
for fold,binding in report['folds'].items():
    cp=OUT/f'fold_{fold}.json';sp=OUT/f'fold_{fold}_schedule.json'
    assert sha(cp)==binding['config_sha256'] and sha(sp)==binding['schedule_sha256']
    cfg=read(cp);schedules=read(sp);assert sha(cfg['init_from'])==binding['initial_sha256']
    roles=cfg['data']['loader_kwargs']['sealed_records_by_split']
    assert roles['calibration'] is None and roles['test'] is None
    assert cfg['train']==base['train'] and cfg['seed']==42
    assert cfg['sequence_loss_candidate']==base['sequence_loss_candidate']
    model=dict(cfg['model']);old=dict(base['model']);model.pop('fracture_positive_weight');old.pop('fracture_positive_weight');assert model==old
    group_roles={}
    for role in ('train','model_val','calibration','test'):
        info=design['folds'][fold][role];assert sha(info['manifest'])==info['sha256']
        rows=read(info['manifest']);group_roles[role]={lookup[r['image_id']] for r in rows}
        if role in ('train','model_val'):assert roles[role]==info['manifest']
    for i,a in enumerate(group_roles):
        for b in list(group_roles)[i+1:]:assert not group_roles[a]&group_roles[b]
    rows=read(roles['train']);counts=Counter(r['answer'] for r in rows if r['question']=='Is there a fracture in this X-ray?')
    assert cfg['model']['fracture_positive_weight']==counts['no']/counts['yes']
    assert len(schedules)==2
    repeated=0
    for epoch,s in enumerate(schedules):
        batches=s['ordered_indices'];assert s['epoch']==epoch
        assert sorted(i for batch in batches for i in batch)==list(range(len(rows)))
        for batch in batches:
            c=Counter(lookup[rows[i]['image_id']] for i in batch)
            assert max(c.values())<=len(batch)//2
            repeated+=int(len(c)<len(batch))
        assert s['updates']==math.ceil(len(batches)/4)
    assert sum(s['updates'] for s in schedules)==binding['total_updates']
    results[fold]={'passed':True,'total_updates':binding['total_updates'],'batches_with_repeated_components':repeated,'donor_disjoint_feasible':True}
parent=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
done=read(parent/'token_mean_completion.json')
folder=ROOT/base['checkpoint_dir']/base['run_name']
assert sha(folder/'last.pt')==done['last_sha256'] and sha(folder/'optimizer.pt')==done['optimizer_sha256']
history=read(folder/'history.json');assert len(history)==2 and history[-1]['epoch']==2
old_schedule=read(parent/'component_schedule.json');new_schedule=read(OUT/'fold_0_schedule.json')
assert [s['ordered_indices'] for s in old_schedule]==[s['ordered_indices'] for s in new_schedule]
new=read(OUT/'fold_0.json');original=dict(base)
for k in ('run_name','figure_dir','recipe_status'):new.pop(k);original.pop(k)
assert new==original
quality=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
audit=read(quality/'independent_artifact_audit.json');assert audit['passed'] and audit['report_sha256']==sha(quality/'report.json')
result={'passed':True,'auditor_sha256':sha(__file__),'preparation_report_sha256':sha(OUT/'report.json'),
    'generator_sources':{str(p):sha(p) for p in [ROOT/'refactor_v2/experiments/materialize_grouped_native_v3_1.py',ROOT/'refactor_v2/experiments/materialize_grouped_native_v3_2.py']},
    'folds':results,'fold0_reuse_compatible':True,'fold0_checkpoint_sha256':done['last_sha256'],
    'fold0_limit':'Historical training/development-informed selection; no untouched prospective claim',
    'new_fold_updates':sum(v['total_updates'] for k,v in results.items() if k!='0'),
    'remaining':'Per-fold model/loss/update/runtime gates, executable calibration and uncertainty specification, capped executor and releases',
    'training_released':False,'model_calls':0,'optimizer_updates':0}
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'fold0_reuse_compatible':True,'new_fold_updates':result['new_fold_updates'],'training_released':False}))
