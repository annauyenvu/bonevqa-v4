"""Fixed LAST410 public modelval evaluation in original1432 QA order; fresh replay."""
import sys, time, subprocess, hashlib
from pathlib import Path
from experiments.evaluate_fullfold0_fixed_LAST_v3_1 import read, sha, dump, bindings, ROOT, CANONICAL
OUT=ROOT/'refactor_v2/reports/frozen_head_public_evaluation_v3_1'
FIT=ROOT/'refactor_v2/reports/frozen_fracture_head_pilot_v3_1'
OLD=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'

def prepare():
    assert not OUT.exists()
    _,cfgs=bindings();cfg=cfgs['token_mean']
    assert read(OLD/'independent_artifact_audit.json')['passed']
    assert read(FIT/'report.json')['optimizer_updates']==410
    paths=[Path(__file__).resolve(),FIT/'HEAD_LAST410.pt',FIT/'report.json',FIT/'protocol.json',FIT/'execution_release.json',OLD/'token_mean.json',OLD/'independent_artifact_audit.json']
    paths += [Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['model_val'])]
    OUT.mkdir()
    dump(OUT/'plan.json',{'dependencies':{str(p):sha(p) for p in paths},'primary':'Candidate minus original canonical321 balanced accuracy at0.5',
        'scope':'Existing inspected modelval1432QA,321images,314knowncomponents; development only',
        'seed':42,'constructor_seeds':[90210,90211],'checkpoint':'Fixed LAST410 head only',
        'control':'Verified original native LAST public predictions in SAME1432order batch4',
        'fresh_replay_exact_required':True,'paired_bootstrap_required_after_inference':True,
        'calibration_test_used':False,'optimizer_updates':0,'production_released':False})

def measure(seed,track):
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed,compute_metrics
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment()
    for p,h in read(OUT/'plan.json')['dependencies'].items():assert sha(p)==h
    _,cfgs=bindings();cfg=cfgs['token_mean']
    parent=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
    payload=torch.load(FIT/'HEAD_LAST410.pt',map_location='cpu')
    assert sha(parent)==payload['parent_checkpoint_sha256'] and payload['updates']==410
    set_seed(seed);model=BoneVQAModel.load(str(parent),device='cpu').eval()
    hashes=lambda:{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
    original=hashes();model.fracture_head.load_state_dict(payload['head_state'],strict=True)
    before=hashes();assert {k for k in before if before[k]!=original[k]}=={'fracture_head.'+k for k in payload['head_state']}
    assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    assert len(ds)==1432
    rows=[];old=read(OLD/'token_mean.json')['records']
    for offset in range(0,len(ds),4):
        b=collate_fn([ds[i] for i in range(offset,min(offset+4,len(ds)))])
        with torch.no_grad():pred=model.predict_batch({'images':b['images'],'questions':b['questions']})
        for j,p in enumerate(pred):
            r={'image_id':b['image_ids'][j],'question':b['questions'][j],'gt':b['answers'][j],'answer_type':b['answer_types'][j],'pred':p['answer']}
            if 'p_yes' in p:r['p_yes']=p['p_yes']
            prior=old[len(rows)]
            assert all(r[k]==prior[k] for k in ('image_id','question','gt','answer_type'))
            if r['question']!=CANONICAL:assert r==prior,'Unexpected noncanonical endpoint change'
            rows.append(r)
        if track and offset%100==0:
            dump(OUT/'progress.json',{'status':'public_inference','completed_QA':len(rows),'total':1432})
            print(len(rows),flush=True)
    assert before==hashes() and sha(parent)==payload['parent_checkpoint_sha256']
    clinical=[r for r in rows if r['question']==CANONICAL];assert len(clinical)==321
    return {'records':rows,'metrics':compute_metrics(rows,fracture_threshold=.5),'canonical_metrics':compute_metrics(clinical,fracture_threshold=.5),'weights_unchanged':True}

def run():
    assert not (OUT/'progress.json').exists();start=time.perf_counter()
    dump(OUT/'progress.json',{'status':'starting'})
    try:
        dump(OUT/'candidate.json',measure(90210,True))
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        dump(OUT/'report.json',{'elapsed_seconds':time.perf_counter()-start,'candidate_sha256':sha(OUT/'candidate.json'),'plan_sha256':sha(OUT/'plan.json'),'paired_analysis_and_independent_audit_pending':True,'quality_gain_established':False})
        dump(OUT/'progress.json',{'status':'replayed_pending_paired_analysis_audit'})
    except Exception as e:
        dump(OUT/'failure.json',{'error':repr(e),'preserved_no_automatic_retry':True});raise

def replay():
    assert measure(90211,False)==read(OUT/'candidate.json')
    dump(OUT/'replay.json',{'passed':True,'all1432_records_scores_metrics_exact':True})

if __name__=='__main__':{'prepare':prepare,'run':run,'replay':replay}[sys.argv[1]]()
