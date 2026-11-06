"""Mixed-QA batch-exact assessment recovery; preserve all fitted heads."""
import copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from experiments import local_noaux_head_analysis_v2 as h
from experiments.vnext_model_v1 import VNextModel

O=h.O
_extract=h.extract
_predict=h.predict
CACHE=[]
def extract(model,fold,arm,role):
    if role=='train':return _extract(model,fold,arm,role)
    CACHE.clear();ds,collate=h.batcher(fold,arm);records=ds[role].records
    features=[];rows=[];capture=[];model.eval();start=time.time()
    handle=model.fracture_head.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().cpu().clone()))
    try:
        with torch.no_grad():
            for off in range(0,len(records),4):
                h.guards(start);ids=list(range(off,min(off+4,len(records))));b=collate([ds[role][i] for i in ids]);capture.clear()
                out=model.predict_batch(h.public(b,arm));x=capture[-1]
                pp=model.fracture_head(x).sigmoid().flatten().tolist()
                keep=[j for j,i in enumerate(ids) if records[i]['question']==h.CAN]
                if not keep:continue
                assert all(pp[j]==float(out[j]['p_yes']) for j in keep)
                CACHE.append((x,keep));features.append(x[keep])
                for j in keep:
                    e=records[ids[j]];v=out[j]
                    rows.append({'image_id':e['image_id'],'question':h.CAN,'gt':e['answer'],'answer_type':'CLOSED','pred':v['answer'],'p_yes':float(v['p_yes'])})
    finally:handle.remove()
    assert len(rows)==len({r['image_id'] for r in rows})
    return torch.cat(features),rows

def cache_prob(head):
    with torch.no_grad():return [float(v) for x,keep in CACHE for v in head(x).sigmoid().flatten()[keep].tolist()]

def predict(model,fold,arm,role,canonical=False,**kwargs):
    rows=_predict(model,fold,arm,role,canonical=False,**kwargs)
    return [r for r in rows if r['question']==h.CAN] if canonical else rows

# Keep scientific head fitting byte-identical; change only evaluation batch context.
src=(h.ROOT/'refactor_v2/experiments/local_noaux_head_analysis_v1.py').read_text()
start=src.index('def head(fold,arm):');end=src.index('\ndef head_replay(',start)
headsrc=src[start:end]
old='with torch.no_grad():prob=train_head(x).sigmoid().flatten().tolist()'
assert headsrc.count(old)==1
headsrc=headsrc.replace(old,'prob=cache_prob(train_head)')
env=dict(vars(h));env.update(extract=extract,cache_prob=cache_prob,predict=predict)
exec(compile(headsrc,__file__,'exec'),env)
h.head=env['head'];h.extract=extract;h.predict=predict

def recover_head(fold,arm):
    """Complete evaluation only of the already fitted local fold-0 head."""
    folder=O/f'fold_{fold}'/arm;dest=folder/'head_refit'
    assert not (dest/'completed.json').exists()
    fit=h.read(dest/'fit.json');assert fit['all_model_parameters_unchanged_during_fit']
    assert h.sha(dest/'HEAD_LAST.pt')==fit['checkpoint_sha256']
    model=VNextModel.load(str(folder/'LAST.pt'),device='cpu').eval();before=h.fingerprint(model)
    payload=torch.load(dest/'HEAD_LAST.pt',map_location='cpu',weights_only=False)
    assert payload['parent_sha256']==h.sha(folder/'LAST.pt')
    assert payload['updates']==fit['updates']
    th=copy.deepcopy(model.fracture_head);th.load_state_dict(payload['state'],strict=True);th.eval();counts={}
    for role in ('model_val','calibration','test'):
        x,original=extract(model,fold,arm,role)
        expected=[r for r in h.read(folder/(role+'_predictions.json'))['records'] if r['question']==h.CAN]
        assert original==expected
        pp=cache_prob(th);rows=[dict(r,p_yes=p,pred='yes' if p>=.5 else 'no') for r,p in zip(original,pp)]
        fp=dest/(role+'_features.pt');assert not fp.exists();torch.save({'features':x,'original_rows':original,'batch_contract':'mixed_QA_batch4'},fp)
        if role=='calibration':full=rows
        else:
            byid={r['image_id']:r for r in rows};full=[byid[r['image_id']] if r['question']==h.CAN else r for r in h.read(folder/(role+'_predictions.json'))['records']]
        h.dump(dest/(role+'_predictions.json'),{'records':full},True);counts[role]=len(full)
    assert h.fingerprint(model)==before
    h.dump(dest/'completed.json',{'roles':counts,'fit_sha256':h.sha(dest/'fit.json'),'parent_sha256':h.sha(folder/'LAST.pt'),'head_sha256':h.sha(dest/'HEAD_LAST.pt'),'no_noncanonical_changes':True,'reused_fitted_head_no_new_updates':True},True)

def supervise():
    dest=O/'postprocess_v3';dest.mkdir(exist_ok=False);deadline=h.read(O/'coordinator.json')['deadline']
    h.dump(dest/'started.json',{'pid':os.getpid(),'wall':time.time(),'deadline':deadline,'preserved_failure_sha256':h.sha(O/'postprocess_failure.json')},True)
    for fold in map(str,range(5)):
        for arm in h.ARMS:
            folder=O/f'fold_{fold}'/arm;hd=folder/'head_refit'
            for mode,receipt in [('head',hd/'completed.json'),('head_replay',hd/'replay.json'),('operating',hd/'operating.json')]:
                if receipt.exists():continue
                actual='recover_head' if mode=='head' and (hd/'fit.json').exists() else mode
                log=dest/f'{actual}_{fold}_{arm}.log'
                with log.open('x') as f:
                    p=subprocess.Popen([sys.executable,'-u',__file__,actual,fold,arm],cwd=h.ROOT,stdout=f,stderr=subprocess.STDOUT)
                    h.dump(dest/'progress.json',{'mode':actual,'fold':fold,'arm':arm,'pid':p.pid,'wall':time.time()})
                    try:code=p.wait(timeout=max(1,min(7200,deadline-time.time())))
                    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
                    assert code==0,str(log)
    assert time.time()<deadline
    h.analyze();h.dump(dest/'completed.json',{'analysis_generated':True,'wall':time.time(),'deadline':deadline,'within_original_true_wall':True,'closure_pending':True},True)

if __name__=='__main__':
    torch.set_num_threads(8)
    try:
        h.verify()
        if sys.argv[1]=='supervise':supervise()
        elif sys.argv[1]=='recover_head':recover_head(*sys.argv[2:])
        else:getattr(h,sys.argv[1])(*sys.argv[2:])
    except Exception as e:
        h.dump(O/'postprocess_v3_failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'wall':time.time(),'preserve_outputs':True});raise
