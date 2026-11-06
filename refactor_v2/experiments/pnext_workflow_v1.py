"""Four new P diagnostic cells; two audited cells are reused by the analysis."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from src.data import build_dataset,collate_fn
from src.models.bonevqa import BoneVQAModel
from src.models.latent_prompt import LatentPromptGenerator
from src.train import build_split,seed_training_epoch
from src.utils import compute_metrics,set_seed

ROOT=Path(__file__).resolve().parents[2];R=ROOT/'refactor_v2/reports';P=R/'step4_localization_v3_1';V=R/'vnext_local_tokens_v1';L=R/'lnext_spatial_runs_v1';Q=R/'pnext_consistency_bank_v1_prepare2';O=R/'pnext_consistency_bank_runs_v1'
ARMS=('P_train_cs0','P_random_cs0','P_empty_cs1','P_empty_cs0')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2)
def check(d):
    for p,h in d.items():assert sha(p)==h,p
def endpoints(rows):return [{k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in r.items() if k!='masks'} for r in rows]
class DiagnosticModel(BoneVQAModel):
    def _encode_core(self,*args,**kwargs):
        core=super()._encode_core(*args,**kwargs);self._diagnostic_core=core;return core
def inputs(cfg):
    datasets={r:build_split(build_dataset,cfg['data']['dataset'],r,None,True,cfg['data']['loader_kwargs']) for r in ('train','model_val')};lookup=read(P/'component_lookup.json')
    def batch(role,indices,public=False):
        b=collate_fn([datasets[role][i] for i in indices])
        if public:return {'images':b['images'],'questions':b['questions']}
        b['component_ids']=[lookup[i] for i in b['image_ids']];return b
    return batch
def prepare():
    assert read(L/'analysis_closure.json')['L_analysis_complete'];check(read(L/'analysis_closure.json')['artifacts'])
    check(read(Q/'preparation_seal.json')['artifacts']);plan=read(Q/'plan.json')
    assert plan['new_arms']==list(ARMS)
    files=list(Q.glob('*.json'))+list(Q.glob('*.pt'))+list((ROOT/'refactor_v2/src').rglob('*.py'))
    files += [Path(__file__).resolve(),ROOT/'refactor_v2/experiments/report_pnext_v1.py',ROOT/'refactor_v2/experiments/report_lnext_v1.py',ROOT/'refactor_v2/experiments/lnext_spatial_v1.py',R/'PNEXT_CONSISTENCY_BANK_PROTOCOL_20261025.md',V/'gate_indices.json',L/'analysis_closure.json']
    files += [P/n for n in ('train.json','model_val.json','schedules.json','data_seal.json','vocab.json','component_lookup.json')]
    for cell in plan['cells'].values():
        if cell['reused']:files += [Path(cell['evaluation']),Path(cell['replay'])]
    write(O/'gate_release.json',{'dependencies':{str(p):sha(p) for p in files},'quality_released':False})
def bank_audit(model,mode):
    bank=model.latent.answer_bank
    if mode=='empty':assert bank.numel()==0;return {'mode':'empty','rows':0}
    assert bank.shape==(len(model.answer_vocab),model.cfg['d_model'])
    with torch.no_grad():base=model.text.encode_frozen(model.answer_vocab,model.device)['tokens'][:,0].float()
    if mode=='train':assert torch.equal(bank,base)
    else:
        assert torch.allclose(bank.norm(dim=1),base.norm(dim=1),atol=1e-5) and not torch.equal(bank,base)
    return {'mode':mode,'rows':len(bank),'sha256':hashlib.sha256(bank.detach().cpu().numpy().tobytes()).hexdigest(),'TRAIN_vocab_only':True}
def mechanics(model,batch,selected):
    model.eval();public=batch('train',selected,True)
    with torch.no_grad():
        normal=endpoints(model.predict_batch(public));dirty=dict(public,answers=['wrong']*4,answer_types=['OPEN']*4,regions=['wrong']*4)
        assert endpoints(model.predict_batch(dirty))==normal
        model.cfg['latent_force_zero']=True;forced=endpoints(model.predict_batch(public));model.cfg['use_latent_prompt']=False
        baseline=endpoints(model.predict_batch(public));model.cfg['use_latent_prompt']=True;model.cfg['latent_force_zero']=False
        assert forced==baseline
    return {'public_metadata_invariant':True,'forced_zero_equals_original_path':True,'normal_vs_forced_answer_or_probability_changed':normal!=forced}
def worker(arm,smoke):
    release=read(O/('gate_release.json' if smoke else 'training_release.json'));check(release['dependencies'])
    if not smoke:assert release['quality_released']
    cfg=read(Q/(arm+'.json'));folder=O/(('gate_' if smoke else 'run_')+arm);assert not folder.exists();folder.mkdir()
    set_seed(42);model=DiagnosticModel.load(cfg['init_from'],device='cpu');batch=inputs(cfg);selected=read(V/'gate_indices.json')
    schedule=[[selected]*8] if smoke else read(P/'schedules.json');expected=read(P/'model_val.json');expected=expected[:8] if smoke else expected
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    names={k:p for k,p in model.named_parameters() if k.startswith(('latent.','fusion.lp_','fusion.gamma_latent','fracture_head.')) or ('visual.' in k and 'lora' in k.lower() and p.requires_grad)}
    initial={k:p.detach().clone() for k,p in names.items()};grads={k:0. for k in names};history=[];diagnostic=[];updates=0;started=time.perf_counter();bank=None
    for epoch,batches in enumerate(schedule):
        seed_training_epoch(42,epoch);model.train();optimizer.zero_grad(set_to_none=True);losses=[];consistency=[];start=time.perf_counter()
        for i,indices in enumerate(batches):
            b=batch('train',indices);out=model(b,stage=3);assert torch.isfinite(out['loss'])
            if smoke and i in (0,4):
                reconstructed=(out['loss_answer'].detach()+model.cfg['eta']*out['loss_cs']+model.cfg['lam']*out['loss_con']+model.cfg.get('lens_alignment_weight',.1)*out['loss_lens_align']+model.cfg['type_loss_weight']*out['loss_type'])
                assert torch.allclose(out['loss'].detach(),reconstructed,atol=1e-6)
                core=model._diagnostic_core;target=model.text.encode_frozen(b['answers'],model.device)
                cs=LatentPromptGenerator.consistency_loss(core['x_lp'],target['tokens'],target['mask'])
                assert torch.allclose(cs.detach(),out['loss_cs'])
                parameter=model.latent.latent
                gcs=torch.autograd.grad(cs,parameter,retain_graph=True,allow_unused=True)[0]
                gqa=torch.autograd.grad(out['loss_answer'],parameter,retain_graph=True,allow_unused=True)[0]
                a=0. if gqa is None else float(gqa.norm());c=0. if gcs is None else float(gcs.norm())
                cosine=float(torch.nn.functional.cosine_similarity(gqa.flatten(),gcs.flatten(),dim=0)) if a>0 and c>0 else None
                diagnostic.append({'microbatch':i,'eta':model.cfg['eta'],'loss_decomposition_exact':True,'answer_latent_gradient_norm':a,'consistency_latent_gradient_norm':c,'gradient_cosine':cosine,'weighted_consistency_loss':float(model.cfg['eta']*cs.detach())})
                if bank is None:bank=bank_audit(model,cfg['pnext']['bank'])
            losses.append(float(out['loss'].detach()));consistency.append(float(out['loss_cs']))
            (out['loss']/4).backward()
            for k,p in names.items():
                if p.grad is not None:assert torch.isfinite(p.grad).all();grads[k]=max(grads[k],float(p.grad.abs().max()))
            if (i+1)%4==0 or i+1==len(batches):
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
        train_seconds=time.perf_counter()-start;model.eval();rows=[];start=time.perf_counter()
        with torch.no_grad():
            for i in range(0,len(expected),4):
                predictions=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
                for e,p in zip(expected[i:i+4],predictions):
                    row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                    if 'p_yes' in p:row['p_yes']=float(p['p_yes'])
                    rows.append(row)
        assert len(rows)==len(expected);metrics=compute_metrics(rows,fracture_threshold=.5)
        write(folder/f'epoch{epoch+1}_evaluation.json',{'records':rows,'metrics':metrics});model.save(str(folder/f'epoch{epoch+1}.pt'),{'updates':updates,'disposable':smoke,'model_class':'DiagnosticModel'})
        torch.save({'optimizer':optimizer.state_dict(),'updates':updates},folder/f'epoch{epoch+1}_optimizer.pt')
        history.append({'epoch':epoch+1,'updates':updates,'loss':sum(losses)/len(losses),'consistency_loss':sum(consistency)/len(consistency),'train_seconds':train_seconds,'eval_seconds':time.perf_counter()-start,'metrics':metrics});print(json.dumps(history[-1]),flush=True)
    assert updates==(2 if smoke else 292)
    changes={k:float((p-initial[k]).detach().abs().max()) for k,p in names.items()}
    assert grads['fusion.gamma_latent']>0 and changes['fusion.gamma_latent']>0
    if cfg['pnext']['bank']=='empty':assert all(v==0 for k,v in grads.items() if k.startswith(('latent.answer_sa.','latent.answer_proj.','latent.cross.')))
    tested=mechanics(model,batch,selected) if smoke else {}
    if smoke:assert tested['normal_vs_forced_answer_or_probability_changed']
    write(folder/'completion.json',{'history':history,'updates':updates,'elapsed_seconds':time.perf_counter()-started,'gradients':grads,'parameter_changes':changes,'TRAIN_gradient_diagnostics':diagnostic,'bank_audit':bank,'mechanics':tested,'fresh_replay_pending':True})
def replay(arm,smoke):
    cfg=read(Q/(arm+'.json'));folder=O/(('gate_' if smoke else 'run_')+arm);epoch=len(read(folder/'completion.json')['history']);path=folder/f'epoch{epoch}.pt';digest=sha(path)
    set_seed(90210);model=DiagnosticModel.load(str(path),device='cpu').eval();batch=inputs(cfg);expected=read(folder/f'epoch{epoch}_evaluation.json')['records'];rows=[]
    with torch.no_grad():
        for i in range(0,len(expected),4):
            predictions=model.predict_batch(batch('model_val',range(i,min(i+4,len(expected))),True))
            for e,p in zip(expected[i:i+4],predictions):
                row={k:e[k] for k in ('image_id','question','gt','answer_type')};row['pred']=p['answer']
                if 'p_yes' in p:row['p_yes']=float(p['p_yes'])
                rows.append(row)
    assert rows==expected and sha(path)==digest
    write(folder/'fresh_replay.json',{'passed':True,'records_exact':len(rows),'checkpoint_sha256':digest})
def child(mode,arm):
    with (O/f'{mode}_{arm}.log').open('x') as log:subprocess.run([sys.executable,'-u',__file__,mode,arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def operate():
    assert not O.exists();O.mkdir();write(O/'coordinator.json',{'pid':os.getpid(),'waiting_for':str(L/'analysis_closure.json'),'started':time.time(),'arms':ARMS})
    while not (L/'analysis_closure.json').exists():
        if list(L.glob('failure_*.json')) or (L/'analysis_failure.json').exists():raise RuntimeError('L failed; inspect before P release')
        time.sleep(30)
    prepare()
    for arm in ARMS:child('smoke',arm);child('replay_smoke',arm)
    release=read(O/'gate_release.json');check(release['dependencies']);release['quality_released']=True
    release['gate_evidence']={a:{'completion':sha(O/f'gate_{a}/completion.json'),'replay':sha(O/f'gate_{a}/fresh_replay.json')} for a in ARMS};write(O/'training_release.json',release)
    for arm in ARMS:child('quality',arm);child('replay_quality',arm)
    write(O/'execution_completed.json',{'new_arms':ARMS,'full_replays_passed':True,'reused_cells':2,'analysis_pending':True})
if __name__=='__main__':
    assert os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1';torch.set_num_threads(8)
    try:
        if len(sys.argv)==1:operate()
        elif sys.argv[1].startswith('replay'):replay(sys.argv[2],sys.argv[1]=='replay_smoke')
        else:worker(sys.argv[2],sys.argv[1]=='smoke')
    except Exception as e:
        if O.exists():
            path=O/('failure_'+str(os.getpid())+'.json')
            if not path.exists():write(path,{'error':repr(e),'argv':sys.argv,'preserve_all_outputs':True})
        raise
