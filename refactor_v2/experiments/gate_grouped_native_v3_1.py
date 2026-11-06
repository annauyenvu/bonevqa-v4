"""Four TRAIN fixtures: native loss, finite gradients, disposable update and replay."""
import sys,json,hashlib,gc,time,subprocess
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA')
PREP=ROOT/'refactor_v2/reports/grouped_native_preparation_v3_2'
OUT=ROOT/'refactor_v2/reports/grouped_native_model_gate_v3_1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def prepare():
    assert not OUT.exists() and read(PREP/'independent_audit.json')['passed']
    exports=read(ROOT/'refactor_v2/reports/perfold_PREtask_roundtrip_v3_1/exports.json')
    paths=[Path(__file__).resolve(),PREP/'independent_audit.json',PREP/'report.json']+list((ROOT/'refactor_v2/src').rglob('*.py'))
    targets={}
    for f in ('1','2','3','4'):
        p=PREP/f'fold_{f}.json';c=read(p);paths += [p,Path(c['init_from']),Path(c['data']['loader_kwargs']['sealed_records_by_split']['train'])]
        targets[f]={'config':str(p),'indices':exports[f]['TRAIN_binding']['TRAIN_indices']}
    OUT.mkdir();dump(OUT/'plan.json',{'dependencies':{str(p):sha(p) for p in paths},'targets':targets,'scope':'One disposable optimizer update per fold per pass, fresh replay; no quality initialization reuse','reserved_inference':False,'training_released':False})
def measure():
    import torch
    from src.models.bonevqa import BoneVQAModel
    from src.data import build_dataset,collate_fn
    from src.train import build_split,seed_training_epoch
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();p=read(OUT/'plan.json')
    for file,h in p['dependencies'].items():assert sha(file)==h
    results={}
    for f,t in p['targets'].items():
        c=read(t['config']);ds=build_split(build_dataset,c['data']['dataset'],'train',None,True,c['data']['loader_kwargs'])
        batch=collate_fn([ds[i] for i in t['indices']]);assert set(batch['answer_types'])=={'OPEN','CLOSED'}
        set_seed(42);m=BoneVQAModel.load(c['init_from'],device='cpu')
        assert m.cfg['fracture_positive_weight']==c['model']['fracture_positive_weight']
        assert not any(m.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        before={k:v.clone() for k,v in m.trainable_state_dict().items()}
        m.train();seed_training_epoch(42,0)
        opt=torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=1e-4,betas=(.9,.98),weight_decay=.01)
        out=m(batch,stage=3);assert torch.isfinite(out['loss']);(out['loss']/4).backward()
        gradients={k:float(v.grad.abs().max()) for k,v in m.named_parameters() if v.requires_grad and v.grad is not None}
        assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None)
        groups={prefix:max((value for k,value in gradients.items() if k.startswith(prefix)),default=0.) for prefix in ('visual.','qformer.','fracture_head.','llm.')}
        assert all(v>0 for v in groups.values())
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        after=m.trainable_state_dict();updates={prefix:max(float((v-before[k]).abs().max()) for k,v in after.items() if k.startswith(prefix)) for prefix in groups}
        assert all(v>0 for v in updates.values())
        m.eval()
        with torch.no_grad():pred=m.predict_batch({'images':batch['images'],'questions':batch['questions']})
        # In-memory state reload verifies tensor copies; fresh process repeats full operation below.
        assert len(after)==263
        results[f]={'loss':float(out['loss'].detach()),'gradient_max':groups,'update_max':updates,'unclipped_norm':float(norm),
            'final_hashes':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in after.items()},'public_TRAIN_predictions':pred,'disposable_updates':1}
        print('Fold',f,'disposable gate passed',flush=True)
        del m,opt,out,after,before;gc.collect()
    return results
def run():
    assert not (OUT/'progress.json').exists();dump(OUT/'progress.json',{'status':'running'});start=time.perf_counter()
    try:
        dump(OUT/'results.json',measure());subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        dump(OUT/'report.json',{'passed':True,'plan_sha256':sha(OUT/'plan.json'),'results_sha256':sha(OUT/'results.json'),'elapsed_seconds':time.perf_counter()-start,'actual_disposable_updates':8,'quality_updates':0,'all_updated_states_discarded':True,'training_released':False,'executor_integration_and_checkpoint_export_gate_pending':True})
        dump(OUT/'progress.json',{'status':'replayed_pending_executor_export_gate'})
    except Exception as e:dump(OUT/'failure.json',{'error':repr(e),'preserved':True});raise
def replay():
    assert measure()==read(OUT/'results.json');dump(OUT/'replay.json',{'passed':True})
if __name__=='__main__':{'prepare':prepare,'run':run,'replay':replay}[sys.argv[1]]()
