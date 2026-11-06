"""Sequential actual-runner gates, fresh replay, sealed pilot execution."""
import os,sys,subprocess,json,time
from pathlib import Path
from experiments.prepare_step4_localization_v3_1 import OUT,ROOT,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
ARMS=('B','V_real','V_placebo')
def replay(arm):
    import torch
    from src.models.bonevqa import BoneVQAModel
    from src.data import build_dataset,collate_fn
    from src.data.fracatlas_vqa import FracAtlasVQADataset
    from src.train import build_split
    torch.set_num_threads(8)
    folder=OUT/('runner_smoke_'+arm); cfg=read(OUT/(arm+'.json'))
    complete=read(folder/'completion.json'); assert complete['updates']==2 and complete['disposable']
    required=[k for k in complete['gradients'] if k.startswith('fracture_head.') or arm!='B']
    assert all(complete['gradients'][k]>0 and complete['selected_parameter_updates'][k]>0 for k in required)
    checkpoint=folder/'epoch_1.pt'; before=sha(checkpoint)
    model=BoneVQAModel.load(str(checkpoint),device='cpu').eval()
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    expected=read(OUT/'model_val.json')[:8]; boxes=read(cfg['pilot_contract']['proposal_path']); actual=[]
    with torch.no_grad():
        for i in range(0,8,4):
            b=collate_fn([ds[j] for j in range(i,i+4)])
            public={'images':b['images'],'questions':b['questions']}
            if arm!='B':
                public['boxes']=[boxes[k] or None for k in b['image_ids']]
                public['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[k],im.size) for k,im in zip(b['image_ids'],b['images'])]
            for e,p in zip(expected[i:i+4],model.predict_batch(public)):
                row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                if 'p_yes' in p: row['p_yes']=float(p['p_yes'])
                actual.append(row)
    assert actual==read(folder/'epoch_1_evaluation.json')['records'] and sha(checkpoint)==before
    dump(folder/'fresh_process_replay.json',{'passed':True,'all_8_records_probabilities_exact':True,'checkpoint_sha256':before,'zero_updates_in_replay':True})
def run():
    assert not (OUT/'operation_started.json').exists()
    dump(OUT/'operation_started.json',{'pid':os.getpid(),'started_utc_epoch':time.time(),'scope':'actual-runner smoke then three bounded quality arms'})
    script=ROOT/'refactor_v2/experiments/run_step4_localization_v3_1.py'
    try:
        for arm in ARMS:
            with (OUT/(arm+'_actual_runner_smoke.log')).open('x') as log:
                subprocess.run([sys.executable,'-u',str(script),arm,'--smoke'],stdout=log,stderr=subprocess.STDOUT,check=True)
            with (OUT/(arm+'_actual_runner_replay.log')).open('x') as log:
                subprocess.run([sys.executable,'-u',__file__,'replay',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
        release=read(OUT/'smoke_release.json')
        for p,h in release['dependencies'].items(): assert sha(p)==h
        dependencies=dict(release['dependencies'])
        for p in list(OUT.glob('runner_smoke_*/*.json'))+[Path(__file__).resolve()]: dependencies[str(p)]=sha(p)
        dump(OUT/'execution_release.json',{'training_released':True,'smoke_only':False,'dependencies':dependencies,'arms':ARMS,'updates_per_arm':292,'epochs':4,'fixed_last':True,'seed':42,'evaluation_scope':'inspected model-validation, not independent confirmation','lr_schedule':'constant1e-4','optimizer':'AdamW default betas .9/.999, weight_decay .01; four-microbatch accumulation including fixed tail scaling','calibration_test_allowed':False,'next':'historical L/P evidence reconciliation, no automatic seed expansion'})
        print('ACTUAL RUNNER GATES AND FRESH REPLAYS PASSED; QUALITY RELEASE SEALED',flush=True)
        for arm in ARMS:
            with (OUT/(arm+'_quality.log')).open('x') as log:
                subprocess.run([sys.executable,'-u',str(script),arm],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('COMPLETED '+arm,flush=True)
        dump(OUT/'operation_completed.json',{'quality_arms_completed':ARMS,'audits_and_LP_synthesis_pending':True})
    except Exception as error:
        dump(OUT/'operation_failure.json',{'error':repr(error),'preserve_outputs_no_automatic_retry':True}); raise
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='replay': replay(sys.argv[2])
    else: run()
