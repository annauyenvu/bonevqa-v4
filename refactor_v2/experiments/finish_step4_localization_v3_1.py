"""Wait without competing with training, then fresh replay/audit/curves/exit review."""
import os,sys,time,subprocess,json
from pathlib import Path
from experiments.prepare_step4_localization_v3_1 import OUT,ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
ARMS=('B','V_real','V_placebo')

def replay(arm):
    import torch
    from src.models.bonevqa import BoneVQAModel
    from src.data import build_dataset,collate_fn
    from src.data.fracatlas_vqa import FracAtlasVQADataset
    from src.train import build_split
    from experiments.gate_step4_localization_v3_1 import hashes
    torch.set_num_threads(8); start=time.perf_counter()
    release=read(OUT/'execution_release.json')
    for p,h in release['dependencies'].items(): assert sha(p)==h,p
    folder=OUT/('run_'+arm); assert read(folder/'completion.json')['updates']==292
    cfg=read(OUT/(arm+'.json')); checkpoint=folder/'epoch_4.pt'
    digest=sha(checkpoint); model=BoneVQAModel.load(str(checkpoint),device='cpu').eval()
    before=hashes(model.trainable_state_dict()); ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    expected=read(OUT/'model_val.json'); boxes=read(cfg['pilot_contract']['proposal_path']); actual=[]
    with torch.no_grad():
        for i in range(0,len(expected),4):
            b=collate_fn([ds[j] for j in range(i,min(i+4,len(expected)))])
            public={'images':b['images'],'questions':b['questions']}
            if arm!='B':
                public['boxes']=[boxes[k] or None for k in b['image_ids']]
                public['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[k],im.size) for k,im in zip(b['image_ids'],b['images'])]
            predictions=model.predict_batch(public); assert len(predictions)==len(b['questions'])
            for e,p in zip(expected[i:i+4],predictions):
                row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                if 'p_yes' in p: row['p_yes']=float(p['p_yes'])
                actual.append(row)
    assert actual==read(folder/'epoch_4_evaluation.json')['records']
    assert sha(checkpoint)==digest and before==hashes(model.trainable_state_dict())
    dump(folder/'full_fresh_process_replay.json',{'passed':True,'records':len(actual),'all_records_probabilities_exact':True,'weights_unchanged':True,'checkpoint_sha256':digest,'elapsed_seconds':time.perf_counter()-start})

def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit=read(OUT/'quality_artifact_metric_audit.json'); historical=read(OUT/'historical_LP_evidence.json')
    assert audit['passed']
    for arm in ARMS: assert read(OUT/('run_'+arm)/'full_fresh_process_replay.json')['passed']
    for p,h in audit['dependencies'].items(): assert sha(p)==h
    for family in ('historical_L','historical_P'):
        for arm,row in historical[family].items():
            if family=='historical_L':
                assert row['verified'] and len(row['history'])==4
                for p,h in row['sources'].items(): assert sha(p)==h
            else: assert sha(row['source'])==row['sha256'] and len(row['history'])==4
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for arm,history in audit['history'].items():
        axes[0].plot([r['epoch'] for r in history[1:]],[r['mean_train_loss'] for r in history[1:]],marker='o',label=arm)
        for ax,key in zip(axes[1:],('fracture_auprc','fracture_sensitivity')):
            ax.plot([r['epoch'] for r in history],[r['metrics'][key] for r in history],marker='o',label=arm)
    for ax,title in zip(axes,('TRAIN total loss','Model-validation fracture AP','Model-validation sensitivity @0.5')):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'B_V_learning_curves.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for family,axrow in zip(('historical_L','historical_P'),axes):
        for arm,row in historical[family].items():
            history=row['history']; epochs=list(range(1,len(history)+1))
            axrow[0].plot(epochs,[r['train_loss'] for r in history],marker='o',label=arm)
            scored=[(e,r.get('val_fracture_auprc')) for e,r in zip(epochs,history) if r.get('val_fracture_auprc') is not None]
            if scored: axrow[1].plot([x[0] for x in scored],[x[1] for x in scored],marker='o',label=arm)
        axrow[0].set_title(family+' TRAIN loss (phase-specific)'); axrow[1].set_title(family+' inspected small-validation AP')
        for ax in axrow: ax.set_xlabel('Recorded epoch'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'historical_L_P_learning_curves.png',dpi=180); plt.close(fig)
    rows=['# Step4 pilot exit evidence','', 'Minimal bounded-pilot evidence is complete for B/V/L/P with the cohort scopes below. This does not establish a useful module or certify screening/multi-seed confirmation.', '', '| New arm | LAST AP | AUROC | Sensitivity | Specificity |','|---|---:|---:|---:|---:|']
    for arm,history in audit['history'].items():
        m=history[-1]['metrics']; rows.append('| '+arm+' | '+' | '.join(f'{m[k]:.5f}' for k in ('fracture_auprc','fracture_auroc','fracture_sensitivity','fracture_specificity'))+' |')
    rows+=['',f"Primary V-real minus placebo AP: {audit['primary_AP_real_minus_placebo']:.6f}; paired component95% interval {audit['component_bootstrap2000_CI95']}; {audit['valid_draws']} valid draws.",'','New B/V:256TRAIN images1164QA, full321model-validation images1432QA, seed42, LAST4,292updates/arm. All three full fresh-process outputs replay exactly. Historical L/P:261TRAIN images1173QA,32-image142QA learning curves plus321-image canonical evaluation. Compare within matched historical controls; do not rank L/P against new B as equal-budget ablation. L contains two alignment and two VQA epochs; P conditioned/random/unconditioned controls have four epochs.','', 'Historical L/P screens did not establish semantic benefit. Negative outcomes satisfy a learning/feasibility pilot criterion; they do not justify automatic expansion. Learning-curve changes are descriptive; fixed LAST remains primary.','', 'Known components are not verified patients, all cohorts are development-informed, no new calibration or outer evaluation. Generic original seven-step or production completion is not certified.','', 'Next: prepare separately frozen step5 selection matrix and compute budget. Step6 requires five matched training seeds and frozen calibration/evaluation protocol; neither five folds nor replay constructor seeds satisfy this requirement. No automatic more-seed run from a positive point estimate.']
    with (OUT/'STEP4_EXIT_EVIDENCE.md').open('x') as f: f.write('\n'.join(rows)+'\n')
    files=[OUT/'quality_artifact_metric_audit.json',OUT/'historical_LP_evidence.json',OUT/'execution_release.json',OUT/'STEP4_EXIT_EVIDENCE.md',OUT/'B_V_learning_curves.png',OUT/'historical_L_P_learning_curves.png']
    files += [OUT/('run_'+a)/'full_fresh_process_replay.json' for a in ARMS]
    dump(OUT/'step4_pilot_closure.json',{'bounded_pilot_evidence_complete':True,'scope':'Minimal B/V/L/P pilot criterion with disclosed different cohorts; not common-cohort factorial','artifacts':{str(p):sha(p) for p in files},'step5_execution_released':False,'step6_execution_released':False,'production_released':False})

def supervise():
    assert not (OUT/'postprocessing_started.json').exists()
    dump(OUT/'postprocessing_started.json',{'pid':os.getpid(),'started_utc_epoch':time.time(),'script_sha256':sha(__file__)})
    try:
        while not (OUT/'operation_completed.json').exists():
            if (OUT/'operation_failure.json').exists(): raise RuntimeError('Training driver failed; inspect preserved logs')
            time.sleep(30)
        for arm in ARMS:
            with (OUT/(arm+'_full_replay.log')).open('x') as log:
                subprocess.run([sys.executable,'-u',__file__,'replay',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (OUT/'postrun_metric_audit.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'refactor_v2/experiments/audit_step4_results_v3_1.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        report()
        print('STEP4 MINIMAL PILOT EVIDENCE CLOSED; STEP5/6 DESIGN PREPARATION ONLY',flush=True)
    except Exception as error:
        dump(OUT/'postprocessing_failure.json',{'error':repr(error),'preserve_outputs':True}); raise
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='replay': replay(sys.argv[2])
    else: supervise()
