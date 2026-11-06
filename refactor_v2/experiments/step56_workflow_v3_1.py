"""Scoped common-cohort screening; prepare five-seed confirmation after selection."""
import copy,gc,json,math,os,sys,time,subprocess
from pathlib import Path
import torch
from src.models.bonevqa import BoneVQAModel
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.train import build_split,seed_training_epoch
from src.utils import set_seed,compute_metrics
from experiments.prepare_step4_localization_v3_1 import OUT as PARENT,ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step56_screening_v3_1'
ARMS=('P_real','P_random','P_unconditioned','L_real','L_original','L_placebo')

def batch_for(ds,indices,arm,boxes,lookup,public=False):
    b=collate_fn([ds[i] for i in indices]);b['component_ids']=[lookup[i] for i in b['image_ids']]
    if arm.startswith('L') and not public:
        b['boxes']=[boxes[i] or None for i in b['image_ids']]
        b['masks']=[FracAtlasVQADataset._boxes_to_masks(boxes[i],im.size) for i,im in zip(b['image_ids'],b['images'])]
    return {'images':b['images'],'questions':b['questions']} if public else b

def prepare():
    assert not OUT.exists();OUT.mkdir()
    baseline=torch.load(PARENT/'B_PREtask.pt',map_location='cpu',weights_only=False)
    configs={};hashes={}
    for arm in ARMS:
        cfg=read(PARENT/'B.json');cfg['model']=copy.deepcopy(baseline['config'])
        cfg['model'].update(prompt_variant=arm[0],use_lens=arm.startswith('L'),use_latent_prompt=arm.startswith('P'),module_init_seed=1771,latent_answer_bank_mode='random' if arm=='P_random' else 'train',latent_context_control='unconditioned' if arm=='P_unconditioned' else 'conditioned',latent_context_control_eval='unconditioned' if arm=='P_unconditioned' else 'conditioned',lens_training_only=True,lens_force_zero=arm=='L_original')
        cfg['run_name']='step5_'+arm; cfg['init_from']=str(OUT/(arm+'_PREtask.pt'))
        set_seed(42);model=BoneVQAModel(cfg['model'],baseline['answer_vocab'],device='cpu')
        model.load_state_dict(baseline['state'],strict=False)
        assert all(torch.equal(v,model.trainable_state_dict()[k]) for k,v in baseline['state'].items())
        model.save(cfg['init_from'],{'zero_QA_updates':True,'shared_B_parent_sha256':sha(PARENT/'B_PREtask.pt')})
        hashes[arm]={k:sha(cfg['init_from']) for k in ['initialization_sha256']}
        cfg['screening']={'seed':42,'epochs':4,'updates':292,'stages':[1,1,3,3] if arm.startswith('L') else [3]*4,'optimizer_reset_at_phase':arm.startswith('L'),'primary':'LAST4','LR':'constant1e-4','AdamW_betas':[.9,.999],'grad_accum':4,'proposal_path':str(PARENT/('placebo.json' if arm=='L_placebo' else 'boxes.json'))}
        dump(OUT/(arm+'.json'),cfg);configs[arm]=cfg
        del model;gc.collect()
    deps=list(OUT.glob('*.json'))+list(OUT.glob('*.pt'))+[Path(__file__).resolve()]
    deps+=list((ROOT/'refactor_v2/src').rglob('*.py'))
    deps+=[PARENT/n for n in ('train.json','model_val.json','data_seal.json','vocab.json','schedules.json','component_lookup.json','boxes.json','placebo.json','step4_pilot_closure.json')]
    dump(OUT/'plan.json',{'status':'FROZEN_BEFORE_NEW_LP_RUNS','new_arms':ARMS,'reuse_B_V':'Audited fixed-LAST seed42 step4 arms reused without rerun; already inspected, no independent screening claim','cohort':'same256TRAIN1164QA/full321modelval1432QA','primary':'canonical AP at fixed LAST4','one_phase_budget':292,'L':'146alignment+146VQA, same phases across L controls; B comparison is phase-unmatched','selection':'Candidate AP must exceed baseline and every own matched control by at least0.005; maximum one candidate, highest AP then alphabetical tie; pragmatic budget rule, not statistical significance','step6_seeds':[42,43,44,45,46],'calibration_evaluation':'development-informed repeated evaluation only; no independent clinical claim; freeze estimator/threshold and roles before any new reserved inference','production_released':False})
    deps.append(OUT/'plan.json')
    dump(OUT/'gate_release.json',{'dependencies':{str(p):sha(p) for p in deps},'quality_training_released':False})

def execute(arm,smoke):
    torch.set_num_threads(8); release=read(OUT/('gate_release.json' if smoke else 'execution_release.json'))
    if not smoke: assert release['quality_training_released']
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    folder=OUT/(('gate_' if smoke else 'run_')+arm);assert not folder.exists();folder.mkdir()
    cfg=read(OUT/(arm+'.json'));model=BoneVQAModel.load(cfg['init_from'],device='cpu');lookup=read(PARENT/'component_lookup.json')
    datasets={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')}
    schedule=read(PARENT/'schedules.json');boxes=read(cfg['screening']['proposal_path']);expected=read(PARENT/'model_val.json')
    if smoke:expected=expected[:8]
    def evaluate(epoch):
        model.eval();rows=[]
        with torch.no_grad():
            for i in range(0,len(expected),4):
                pred=model.predict_batch(batch_for(datasets['model_val'],range(i,min(i+4,len(expected))),arm,boxes,lookup,True))
                assert len(pred)==len(expected[i:i+4])
                for e,p in zip(expected[i:i+4],pred):
                    r={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                    if 'p_yes' in p:r['p_yes']=float(p['p_yes'])
                    rows.append(r)
        dump(folder/f'epoch{epoch}_evaluation.json',{'records':rows,'metrics':compute_metrics(rows,fracture_threshold=.5)})
        return compute_metrics(rows,fracture_threshold=.5)
    started=time.perf_counter();history=[];updates=0;laststage=None
    stages=cfg['screening']['stages']
    if smoke: stages=[1,3] if arm.startswith('L') else [3]
    parameters=[p for p in model.parameters() if p.requires_grad]
    selected={k:p for k,p in model.named_parameters() if (k.startswith('latent.') if arm.startswith('P') else k.startswith('lens_fusion.'))}
    gradients={k:0. for k in selected};before={k:p.detach().clone() for k,p in selected.items()}
    for epoch,stage in enumerate(stages):
        if stage!=laststage:optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=.01);laststage=stage
        seed_training_epoch(42,epoch);model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        batches=schedule[epoch]
        if smoke:
            records=read(PARENT/'train.json');batches=[b for b in batches if any(boxes[records[i]['image_id']] for i in b)][:8]
        for i,indices in enumerate(batches):
            output=model(batch_for(datasets['train'],indices,arm,boxes,lookup),stage=stage);assert torch.isfinite(output['loss'])
            losses.append(float(output['loss'].detach()));(output['loss']/4).backward()
            for k,p in selected.items():
                if p.grad is not None:assert torch.isfinite(p.grad).all();gradients[k]=max(gradients[k],float(p.grad.abs().max()))
            if (i+1)%4==0 or i+1==len(batches):
                assert updates<(4 if smoke and arm.startswith('L') else 2 if smoke else 292)
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(parameters,1));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        history.append({'epoch':epoch+1,'stage':stage,'updates':updates,'loss':sum(losses)/len(losses),'metrics':evaluate(epoch+1) if stage==3 else None})
        model.save(str(folder/f'epoch{epoch+1}.pt'),{'updates':updates,'disposable':smoke,'stage':stage})
        print(json.dumps(history[-1]),flush=True)
    changes={k:float((p-before[k]).detach().abs().max()) for k,p in selected.items()}
    assert any(v>0 for v in gradients.values()) and any(v>0 for v in changes.values())
    if not smoke:assert updates==292
    dump(folder/'completion.json',{'history':history,'updates':updates,'gradients':gradients,'updates_by_parameter':changes,'elapsed_seconds':time.perf_counter()-started,'full_replay_pending':True})

def operate():
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    prepare()
    for arm in ARMS:
        with (OUT/(arm+'_gate.log')).open('x') as log:subprocess.run([sys.executable,'-u',__file__,'smoke',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
        with (OUT/(arm+'_gate_replay.log')).open('x') as log:subprocess.run([sys.executable,'-u',__file__,'replay_gate',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Gates are disposable; never initialize quality runs from their weights.
    release=read(OUT/'gate_release.json');release['quality_training_released']=True
    dump(OUT/'execution_release.json',release)
    for arm in ARMS:
        with (OUT/(arm+'_quality.log')).open('x') as log:subprocess.run([sys.executable,'-u',__file__,'quality',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(OUT/'training_completed.json',{'new_arms':ARMS,'full_output_replay_selection_step6_pending':True})
def replay_gate(arm):
    torch.set_num_threads(8);cfg=read(OUT/(arm+'.json'));folder=OUT/('gate_'+arm);done=read(folder/'completion.json')
    epoch=len(done['history']);checkpoint=folder/f'epoch{epoch}.pt';digest=sha(checkpoint)
    model=BoneVQAModel.load(str(checkpoint),device='cpu').eval()
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs']);lookup=read(PARENT/'component_lookup.json');boxes=read(cfg['screening']['proposal_path'])
    expected=read(PARENT/'model_val.json')[:8];rows=[]
    with torch.no_grad():
        for i in (0,4):
            predictions=model.predict_batch(batch_for(ds,range(i,i+4),arm,boxes,lookup,True))
            for e,p in zip(expected[i:i+4],predictions):
                r={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                if 'p_yes' in p:r['p_yes']=float(p['p_yes'])
                rows.append(r)
    assert rows==read(folder/f'epoch{epoch}_evaluation.json')['records'] and sha(checkpoint)==digest
    dump(folder/'fresh_replay.json',{'passed':True,'8outputs_exact':True,'checkpoint_sha256':digest})

if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='replay_gate':replay_gate(sys.argv[2])
        elif len(sys.argv)>1:execute(sys.argv[2],sys.argv[1]=='smoke')
        else:operate()
    except Exception as error:
        if OUT.exists() and len(sys.argv)==1:dump(OUT/'operation_failure.json',{'error':repr(error),'preserve_outputs':True})
        raise
