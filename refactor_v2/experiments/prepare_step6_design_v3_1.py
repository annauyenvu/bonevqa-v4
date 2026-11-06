"""Materialize seed schedules and reserved-role bindings without loading a model."""
from pathlib import Path
import math
from src.context_controls import ComponentBatchSampler
from experiments.prepare_step4_localization_v3_1 import OUT as PARENT,ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step6_confirmation_v3_1'
def main():
    assert not OUT.exists();OUT.mkdir()
    binding=read(R/'grouped_evaluation_bindings_v3_1/bindings.json')['folds']['0']
    lookup=read(PARENT/'component_lookup.json');tr=read(PARENT/'train.json');val=read(PARENT/'model_val.json')
    roles={'train':tr,'model_val':val}
    deps={str(PARENT/n):sha(PARENT/n) for n in ('train.json','model_val.json','vocab.json','data_seal.json','component_lookup.json','B_PREtask.pt')}
    for role in ('calibration','test'):
        info=binding['roles'][role];assert sha(info['path'])==info['sha256'];roles[role]=read(info['path']);deps[info['path']]=info['sha256']
    groups={r:{lookup[x['image_id']] for x in rows} for r,rows in roles.items()}
    import itertools
    assert all(not groups[a]&groups[b] for a,b in itertools.combinations(groups,2))
    schedules={}
    for seed in (42,43,44,45,46):
        sampler=ComponentBatchSampler([lookup[r['image_id']] for r in tr],4,seed);batches=[]
        for epoch in range(4):
            sampler.set_epoch(epoch);bb=list(sampler);assert sorted(i for b in bb for i in b)==list(range(len(tr)));batches.append(bb)
        assert sum(math.ceil(len(b)/4) for b in batches)==292
        path=OUT/f'seed{seed}_schedule.json';dump(path,batches);schedules[str(seed)]={'path':str(path),'sha256':sha(path),'updates':292}
    dump(OUT/'role_binding.json',{'roles':{r:{'QA':len(rows),'images':len({x['image_id'] for x in rows}),'known_components':len(groups[r])} for r,rows in roles.items()},'reserved_paths':{r:binding['roles'][r]['path'] for r in ('calibration','test')},'dependencies':deps,'known_components_disjoint':True,'scope':'Previously inspected development-informed evaluation, not independent clinical confirmation'})
    dump(OUT/'preparation.json',{'seed_schedules':schedules,'training_released':False,'inference_released':False,'pending':['audited step5 selection','seed-specific initialization and shared-state checks','disposable routing/update and fresh reload gates','model/source/data execution release','five-seed training and fixed calibration/assessment'],'source_sha256':sha(__file__)})
    print('Five seed schedules and disjoint role bindings prepared; zero model updates')
if __name__=='__main__':main()
