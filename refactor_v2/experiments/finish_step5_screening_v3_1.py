"""Wait for screening, replay all LAST outputs, audit metrics and frozen selection."""
import os,sys,time,subprocess
from experiments.step56_workflow_v3_1 import OUT,PARENT,ROOT,ARMS,read,dump,sha,batch_for

def replay(arm):
    import torch
    from src.models.bonevqa import BoneVQAModel
    from src.data import build_dataset
    from src.train import build_split
    from src.utils import compute_metrics
    from experiments.gate_step4_localization_v3_1 import hashes
    torch.set_num_threads(8);cfg=read(OUT/(arm+'.json'));folder=OUT/('run_'+arm)
    done=read(folder/'completion.json');assert done['updates']==292 and len(done['history'])==4
    checkpoint=folder/'epoch4.pt';digest=sha(checkpoint);model=BoneVQAModel.load(str(checkpoint),device='cpu').eval();before=hashes(model.trainable_state_dict())
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    expected=read(PARENT/'model_val.json');lookup=read(PARENT/'component_lookup.json');boxes=read(cfg['screening']['proposal_path']);actual=[];started=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(expected),4):
            pred=model.predict_batch(batch_for(ds,range(i,min(i+4,len(expected))),arm,boxes,lookup,True));assert len(pred)==len(expected[i:i+4])
            for e,p in zip(expected[i:i+4],pred):
                r={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                if 'p_yes' in p:r['p_yes']=float(p['p_yes'])
                actual.append(r)
    saved=read(folder/'epoch4_evaluation.json');assert actual==saved['records'] and sha(checkpoint)==digest and hashes(model.trainable_state_dict())==before
    assert compute_metrics(actual,fracture_threshold=.5)==saved['metrics']==done['history'][-1]['metrics']
    dump(folder/'full_replay.json',{'passed':True,'records':len(actual),'checkpoint_sha256':digest,'output_metrics_exact':True,'weights_unchanged':True,'seconds':time.perf_counter()-started})

def select():
    import numpy as np
    from sklearn.metrics import average_precision_score,roc_auc_score
    release=read(OUT/'execution_release.json')
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    data={};scores={};histories={};deps={}
    expected=read(PARENT/'model_val.json')
    paths={a:PARENT/('run_'+a)/'epoch_4_evaluation.json' for a in ('B','V_real','V_placebo')}
    paths.update({a:OUT/('run_'+a)/'epoch4_evaluation.json' for a in ARMS})
    for arm,path in paths.items():
        if arm in ARMS:
            folder=path.parent;done=read(folder/'completion.json');assert done['updates']==292 and read(folder/'full_replay.json')['passed']
            assert [r['stage'] for r in done['history']]==([1,1,3,3] if arm.startswith('L') else [3]*4)
            histories[arm]=done['history']
        else:assert read(path.parent/'full_fresh_process_replay.json')['passed']
        d=read(path);rows=d['records'];assert [(r['image_id'],r['question'],r['gt']) for r in rows]==[(r['image_id'],r['question'],r['answer']) for r in expected]
        canonical=[r for r in rows if r['question']=='Is there a fracture in this X-ray?'];assert len(canonical)==321
        y=np.array([r['gt']=='yes' for r in canonical]);p=np.array([r['p_yes'] for r in canonical]);assert np.isfinite(p).all()
        scores[arm]=float(average_precision_score(y,p));assert abs(scores[arm]-d['metrics']['fracture_auprc'])<1e-12
        assert abs(roc_auc_score(y,p)-d['metrics']['fracture_auroc'])<1e-12
        data[arm]=(y,p);deps[str(path)]=sha(path)
    controls={'V_real':['V_placebo'],'P_real':['P_random','P_unconditioned'],'L_real':['L_original','L_placebo']}
    eligible=[a for a,cc in controls.items() if all(scores[a]-scores[b]>=.005 for b in ['B']+cc)]
    selected=sorted(eligible,key=lambda a:(-scores[a],a))[:1]
    dump(OUT/'screening_selection.json',{'passed_metric_budget_replay_checks':True,'AP':scores,'eligible':eligible,'selected':selected,'rule':read(OUT/'plan.json')['selection'],'dependencies':deps,'history':histories,'step6_arms':['B']+([selected[0]]+controls[selected[0]] if selected else []),'step6_seeds':[42,43,44,45,46],'next':'Materialize five-seed initialization/schedules, gate, seal training and calibration/assessment execution per STEP6_LOCKED_DESIGN_V3_1.md','production_released':False})
    print('SCREENING AUDITED; SELECTED '+str(selected),flush=True)

def main():
    assert not (OUT/'postprocessing_started.json').exists();dump(OUT/'postprocessing_started.json',{'pid':os.getpid(),'source_sha256':sha(__file__)})
    try:
        while not (OUT/'training_completed.json').exists():
            if (OUT/'operation_failure.json').exists():raise RuntimeError('Screening execution failed; inspect preserved logs')
            time.sleep(30)
        for arm in ARMS:
            with (OUT/(arm+'_full_replay.log')).open('x') as log:subprocess.run([sys.executable,'-u',__file__,'replay',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
        select()
    except Exception as error:
        dump(OUT/'postprocessing_failure.json',{'error':repr(error),'preserve_outputs':True});raise
if __name__=='__main__':
    if len(sys.argv)>1:replay(sys.argv[2])
    else:main()
