"""Data/initialization/proposal checks and release sealing, no model training."""
import itertools,json,math
from pathlib import Path
import torch
from experiments.prepare_step4_localization_v3_1 import OUT,ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha,validate_roles
from experiments.audit_vl_matched_controls_v3_1 import union_area

def main():
    def explicit_matches(actual,requested):
        return all(k in actual and explicit_matches(actual[k],v) for k,v in requested.items()) if isinstance(requested,dict) else actual==requested
    torch.set_num_threads(8)
    manifest=read(ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json')
    lookup={i:str(k) for k,c in enumerate(manifest['components']) for i in c}
    validate_roles(OUT/'active_roles_manifest.json')
    rows={r:read(OUT/(r+'.json')) for r in ('train','model_val')}
    ids={r:{x['image_id'] for x in v} for r,v in rows.items()}
    assert not {lookup[i] for i in ids['train']}&{lookup[i] for i in ids['model_val']}
    if not (OUT/'component_lookup.json').exists(): dump(OUT/'component_lookup.json',lookup)
    seal=read(OUT/'data_seal.json')
    for p,h in seal['artifacts'].items(): assert sha(ROOT/p)==h
    provenance=read(OUT/'proposal_provenance.json'); real=read(OUT/'boxes.json'); placebo=read(OUT/'placebo.json')
    assert set(real)==set(placebo)==ids['train']|ids['model_val']
    assert sha(OUT/'boxes.json')==provenance['real_sha256'] and sha(OUT/'placebo.json')==provenance['placebo_sha256']
    assert sha(provenance['detector']['checkpoint'])==provenance['detector']['checkpoint_sha256']
    rawhash={}
    for role,rr in rows.items():
        for row in rr:
            iid=row['image_id']; info=provenance['images'][iid]
            path=ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'/row['image']
            rawhash[iid]=sha(path); assert rawhash[iid]==info['sha256'] and info['role']==role
            a,b=real[iid],placebo[iid]
            assert len(a)==len(b) and union_area(a)==union_area(b)
            assert [(x[2]-x[0],x[3]-x[1]) for x in a]==[(x[2]-x[0],x[3]-x[1]) for x in b]
            assert all(0<=x[0]<x[2]<=info['width'] and 0<=x[1]<x[3]<=info['height'] for x in a+b)
    assert not {rawhash[i] for i in ids['train']}&{rawhash[i] for i in ids['model_val']}
    schedule=read(OUT/'schedules.json'); n=len(rows['train'])
    assert len(schedule)==4 and all(sorted(i for b in ep for i in b)==list(range(n)) for ep in schedule)
    positive=sum(r['answer']=='yes' for r in rows['train'] if r['question']=='Is there a fracture in this X-ray?')
    states={}
    for arm in ('B','V_real','V_placebo'):
        cfg=read(OUT/(arm+'.json')); assert cfg['data']['loader_kwargs']['sealed_records_by_split']['test'] is None
        assert cfg['data']['loader_kwargs']['sealed_records_by_split']['calibration'] is None
        assert cfg['pilot_contract']['updates']==sum(math.ceil(len(ep)/4) for ep in schedule)==292
        assert cfg['model']['fracture_positive_weight']==(len(ids['train'])-positive)/positive
        payload=torch.load(OUT/(arm+'_PREtask.pt'),map_location='cpu',weights_only=False)
        assert explicit_matches(payload['config'],cfg['model'])
        states[arm]=payload['state']
    assert all(torch.equal(v,states[a][k]) for k,v in states['B'].items() for a in ('V_real','V_placebo'))
    assert states['V_real'].keys()==states['V_placebo'].keys()
    assert all(torch.equal(v,states['V_placebo'][k]) for k,v in states['V_real'].items())
    deps=list(OUT.glob('*.json'))+list(OUT.glob('*_PREtask.pt'))+list((ROOT/'refactor_v2/src').rglob('*.py'))
    deps+=[ROOT/'refactor_v2/experiments'/s for s in ('run_step4_localization_v3_1.py','prepare_step4_localization_v3_1.py','release_step4_localization_v3_1.py')]
    target=OUT/'smoke_release.json'
    dump(target,{'smoke_only':True,'training_released':False,'dependencies':{str(p):sha(p) for p in deps},'audit':'image bytes/components/geometry/schedules/common initialization exact','canonical_positive':positive})
    print('Smoke release sealed; no quality training release yet',flush=True)
if __name__=='__main__': main()
