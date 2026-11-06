"""TRAIN-only head refit and fixed calibration operating points; no model search."""
import copy, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from experiments.local_noaux_refinement_v1 import ROOT,R,O,CAN,ARMS,read,sha,dump,verify,guards,batcher,public,predict,frozen_hash

def fingerprint(model):
    from experiments.local_noaux_refinement_v1 import tensor_hash
    return {k:tensor_hash(v) for k,v in model.state_dict().items()}

def extract(model,fold,arm,role):
    ds,collate=batcher(fold,arm);indices=[i for i,r in enumerate(ds[role].records) if r['question']==CAN]
    features=[];rows=[];model.eval();capture=[]
    def hook(module,args):capture.append(args[0].detach().cpu().clone())
    handle=model.fracture_head.register_forward_pre_hook(hook)
    start=time.time()
    try:
        with torch.no_grad():
            for offset in range(0,len(indices),4):
                guards(start);ids=indices[offset:offset+4];b=collate([ds[role][i] for i in ids]);capture.clear();out=model.predict_batch(public(b,arm))
                # Local model computes original and local heads; LAST call is the actual public decision.
                assert capture and capture[-1].shape==(len(ids),768)
                x=capture[-1];assert torch.isfinite(x).all()
                p=torch.sigmoid(model.fracture_head(x)).flatten().tolist()
                assert p==[float(v['p_yes']) for v in out]
                features.append(x)
                for i,v in zip(ids,out):
                    e=ds[role].records[i];rows.append({'image_id':e['image_id'],'question':CAN,'gt':e['answer'],'answer_type':'CLOSED','pred':v['answer'],'p_yes':float(v['p_yes'])})
    finally:handle.remove()
    assert len(rows)==len({r['image_id'] for r in rows})
    return torch.cat(features),rows

def head(fold,arm):
    from experiments.vnext_model_v1 import VNextModel
    from src.models.heads import FracturePresenceHead
    verify();folder=O/f'fold_{fold}'/arm;dest=folder/'head_refit';assert not dest.exists();dest.mkdir()
    cp=folder/'LAST.pt';model=VNextModel.load(str(cp),device='cpu').eval();original=fingerprint(model)
    train_x,train_rows=extract(model,fold,arm,'train');y=torch.tensor([r['gt']=='yes' for r in train_rows],dtype=torch.float32)
    assert y.sum()>0 and y.sum()<len(y)
    train_head=copy.deepcopy(model.fracture_head);assert len(dict(train_head.named_parameters()))==4
    weight=(len(y)-y.sum())/y.sum();torch.manual_seed(42);generator=torch.Generator().manual_seed(42)
    opt=torch.optim.AdamW(train_head.parameters(),lr=1e-4,betas=(.9,.98),weight_decay=.01)
    history=[];updates=0;start=time.time()
    for epoch in range(10):
        train_head.train();order=torch.randperm(len(y),generator=generator);total=0.
        for off in range(0,len(y),64):
            guards(start);idx=order[off:off+64];opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(train_head(train_x[idx]).flatten(),y[idx],pos_weight=weight)
            assert torch.isfinite(loss);loss.backward();assert torch.isfinite(torch.nn.utils.clip_grad_norm_(train_head.parameters(),1));opt.step();updates+=1;total+=float(loss.detach())*len(idx)
        history.append({'epoch':epoch+1,'updates':updates,'TRAIN_BCE':total/len(y)})
    assert updates==10*math.ceil(len(y)/64)
    train_head.eval();torch.save({'state':train_head.state_dict(),'parent_sha256':sha(cp),'TRAIN_image_ids':[r['image_id'] for r in train_rows],'updates':updates,'seed':42},dest/'HEAD_LAST.pt')
    torch.save({'features':train_x,'rows':train_rows,'parent_sha256':sha(cp)},dest/'TRAIN_features.pt')
    # No backbone/model parameter changed while fitting the detached head.
    assert fingerprint(model)==original
    dump(dest/'fit.json',{'TRAIN_images':len(y),'positive_weight':float(weight),'updates':updates,'history':history,'all_model_parameters_unchanged_during_fit':True,'checkpoint_sha256':sha(dest/'HEAD_LAST.pt'),'wall_seconds':time.time()-start},True)
    role_records={}
    for role in ('model_val','calibration','test'):
        x,original_rows=extract(model,fold,arm,role)
        expected=[r for r in read(folder/(role+'_predictions.json'))['records'] if r['question']==CAN]
        assert original_rows==expected
        with torch.no_grad():prob=train_head(x).sigmoid().flatten().tolist()
        rows=[dict(r,p_yes=float(p),pred='yes' if p>=.5 else 'no') for r,p in zip(original_rows,prob)]
        torch.save({'features':x,'original_rows':original_rows},dest/(role+'_features.pt'))
        if role=='calibration':full=rows
        else:
            byid={r['image_id']:r for r in rows}
            full=[byid[r['image_id']] if r['question']==CAN else r for r in read(folder/(role+'_predictions.json'))['records']]
        dump(dest/(role+'_predictions.json'),{'records':full},True);role_records[role]=len(full)
    assert fingerprint(model)==original
    dump(dest/'completed.json',{'roles':role_records,'fit_sha256':sha(dest/'fit.json'),'parent_sha256':sha(cp),'head_sha256':sha(dest/'HEAD_LAST.pt'),'no_noncanonical_changes':True},True)

def head_replay(fold,arm):
    from experiments.vnext_model_v1 import VNextModel
    verify();folder=O/f'fold_{fold}'/arm;dest=folder/'head_refit'
    model=VNextModel.load(str(folder/'LAST.pt'),device='cpu').eval()
    p=torch.load(dest/'HEAD_LAST.pt',map_location='cpu',weights_only=False);assert p['parent_sha256']==sha(folder/'LAST.pt')
    model.fracture_head.load_state_dict(p['state'],strict=True)
    # Public canonical predictions recompute the real image path, not only a cached-head matmul.
    counts={}
    for role in ('model_val','calibration','test'):
        actual=predict(model,fold,arm,role,canonical=True)
        expected=[r for r in read(dest/(role+'_predictions.json'))['records'] if r['question']==CAN]
        assert actual==expected,(fold,arm,role)
        counts[role]=len(actual)
    dump(dest/'replay.json',{'exact_canonical_public_records':counts,'HEAD_sha256':sha(dest/'HEAD_LAST.pt')},True)

def threshold(rows):
    neg=np.array([r['p_yes'] for r in rows if r['gt']=='no'],dtype=np.float64)
    assert len(neg)>0 and any(r['gt']=='yes' for r in rows)
    candidates=np.r_[0.,np.nextafter(np.unique(neg),np.inf)]
    eligible=[float(t) for t in candidates if float(np.mean(neg<t))>=.90]
    assert eligible
    t=min(eligible)
    return {'threshold':t,'calibration_negative_n':len(neg),'calibration_specificity':float(np.mean(neg<t)),'target_specificity':.90,'rule':'minimal_threshold_negative_only_with_ties_nextafter'}

def metrics(rows,t=.5):
    from sklearn.metrics import average_precision_score,roc_auc_score
    from src.utils import normalize_answer
    can=[r for r in rows if r['question']==CAN];y=np.array([r['gt']=='yes' for r in can]);p=np.array([r['p_yes'] for r in can]);d=p>=t
    tp=int(np.sum(y&d));fn=int(np.sum(y&~d));tn=int(np.sum(~y&~d));fp=int(np.sum(~y&d))
    sens=tp/(tp+fn);spec=tn/(tn+fp)
    correct=[];closed=[];opened=[]
    for r in rows:
        pred=('yes' if r['p_yes']>=t else 'no') if r['question']==CAN else r['pred']
        v=normalize_answer(pred)==normalize_answer(r['gt']);correct.append(v)
        (closed if r['answer_type']=='CLOSED' else opened).append(v)
    return {'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'sensitivity':sens,'specificity':spec,'BA':(sens+spec)/2,'TP':tp,'FN':fn,'TN':tn,'FP':fp,'Brier':float(np.mean((p-y)**2)),'QA':float(np.mean(correct)),'CLOSED':float(np.mean(closed)) if closed else None,'OPEN':float(np.mean(opened)) if opened else None,'n':len(rows),'canonical_n':len(can),'threshold':t}

def operating(fold,arm):
    verify();folder=O/f'fold_{fold}'/arm
    for variant,dest in (('original',folder),('head_refit',folder/'head_refit')):
        assert (dest/'replay.json').exists()
        cal=read(dest/'calibration_predictions.json')['records'];point=threshold(cal)
        source={r:sha(dest/(r+'_predictions.json')) for r in ('calibration','model_val','test')}
        result={'variant':variant,'fit_from':'calibration_negatives_only','operating_point':point,'sources':source,'roles':{}}
        for role in ('model_val','test'):
            rows=read(dest/(role+'_predictions.json'))['records']
            result['roles'][role]={'raw':metrics(rows),'spec90':metrics(rows,point['threshold'])}
        dump(dest/'operating.json',result,True)

def analyze():
    from src.context_controls import load_component_lookup
    from src.utils import normalize_answer
    verify();assert (O/'execution_completed.json').exists();dest=O/'analysis';assert not dest.exists();dest.mkdir()
    cfg=read(O/'fold_0/B.json');lookup,_=load_component_lookup(cfg['train']['component_batching']['manifest'],cfg['train']['component_batching']['seal'],ROOT)
    report={'primary':'original_head_pooled_outer_AP_local_minus_B','folds':{},'pooled':{},'bootstrap':{},'audit':{}};arrays={};qa={}
    deps={};identity=None
    for variant in ('original','head_refit'):
        for arm in ARMS:
            allrows=[];pp=[];yy=[];dd=[];ff=[];cc=[]
            for fold in map(str,range(5)):
                base=O/f'fold_{fold}'/arm;path=base if variant=='original' else base/'head_refit'
                assert (path/'replay.json').exists()
                op=read(path/'operating.json');rows=read(path/'test_predictions.json')['records'];cal=read(path/'calibration_predictions.json')['records']
                assert threshold(cal)==op['operating_point']
                calc=metrics(rows);assert calc==op['roles']['test']['raw']
                canonical=[r for r in rows if r['question']==CAN]
                report['folds'].setdefault(fold,{})[variant+'_'+arm]=op['roles']['test']
                allrows+=rows;pp += [r['p_yes'] for r in canonical];yy += [r['gt']=='yes' for r in canonical];dd += [r['p_yes']>=op['operating_point']['threshold'] for r in canonical];ff += [fold]*len(canonical);cc += [lookup[r['image_id']] for r in canonical]
                deps[str(path/'test_predictions.json')]=sha(path/'test_predictions.json');deps[str(path/'operating.json')]=sha(path/'operating.json')
            order=[(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in allrows]
            if identity is None:identity=order
            assert identity==order
            assert len({r['image_id'] for r in allrows})==4013 and len(allrows)==17998
            key=variant+'_'+arm;arrays[key]={'p':np.array(pp),'y':np.array(yy),'d':np.array(dd),'fold':np.array(ff),'component':np.array(cc)}
            raw=metrics(allrows);y=np.array(yy);d=np.array(dd);sens=float(d[y].mean());spec=float((~d[~y]).mean())
            # QA at operating point changes only canonical decisions.
            noncorrect=sum(normalize_answer(r['pred'])==normalize_answer(r['gt']) for r in allrows if r['question']!=CAN)
            report['pooled'][key]={'raw':raw,'spec90':{'sensitivity':sens,'specificity':spec,'BA':(sens+spec)/2,'QA':(noncorrect+int(np.sum(d==y)))/len(allrows)}}
    from sklearn.metrics import average_precision_score
    rng=np.random.default_rng(20261105);base=arrays['original_B'];groups={f:{c:np.where((base['fold']==f)&(base['component']==c))[0] for c in np.unique(base['component'][base['fold']==f])} for f in map(str,range(5))}
    draws={v:{'AP':[],'BA_spec90':[],'sensitivity_spec90':[],'specificity_spec90':[]} for v in ('original','head_refit')}
    for _ in range(2000):
        indices=np.concatenate([np.concatenate([bycomp[c] for c in rng.choice(list(bycomp),size=len(bycomp),replace=True)]) for bycomp in groups.values()])
        for v in draws:
            a=arrays[v+'_local_real_noaux'];b=arrays[v+'_B'];y=a['y'][indices]
            assert np.array_equal(y,b['y'][indices]) and y.any() and (~y).any()
            ds=float(a['d'][indices][y].mean()-b['d'][indices][y].mean());dn=float((~a['d'][indices][~y]).mean()-(~b['d'][indices][~y]).mean())
            draws[v]['AP'].append(float(average_precision_score(y,a['p'][indices])-average_precision_score(y,b['p'][indices])))
            draws[v]['BA_spec90'].append((ds+dn)/2);draws[v]['sensitivity_spec90'].append(ds);draws[v]['specificity_spec90'].append(dn)
    for v,values in draws.items():report['bootstrap'][v]={k:{'ci95':np.quantile(vals,[.025,.975]).tolist(),'draws':vals} for k,vals in values.items()}
    a=report['pooled']['original_local_real_noaux'];b=report['pooled']['original_B'];interval=report['bootstrap']['original']
    report['promotion_rule_passed']=bool(interval['AP']['ci95'][0]>0 and interval['BA_spec90']['ci95'][0]>0 and a['spec90']['sensitivity']>b['spec90']['sensitivity'] and a['spec90']['specificity']>=b['spec90']['specificity']-.02 and a['raw']['QA']>=b['raw']['QA']-.005)
    historical=[];newbase=[];report['historical_baseline_comparison']={}
    for fold in map(str,range(5)):
        path=R/f'grouped_reserved_evaluation_v3_1/fold_{fold}.json';old=read(path)['raw_records'];new=read(O/f'fold_{fold}/B/test_predictions.json')['records']
        keys=lambda rows:[(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]
        assert keys(old)==keys(new)
        report['historical_baseline_comparison'][fold]={'same_id_question_target_order':True,'records_bitwise_same':old==new,'historical_metrics':metrics(old),'new_metrics':metrics(new)}
        historical+=old;newbase+=new;deps[str(path)]=sha(path)
    report['historical_baseline_comparison']['pooled']={'historical':metrics(historical),'new':metrics(newbase),'reason_for_any_difference':'Must be investigated in final review; matched new B remains the controlled comparator, not silently substituted historical results.'}
    place=O/'fold_0/local_placebo_noaux/test_predictions.json'
    report['placebo_fold0']={'real':metrics(read(O/'fold_0/local_real_noaux/test_predictions.json')['records']),'placebo':metrics(read(place)['records']),'purpose':'same-noaux spatial control, development diagnostic'}
    report['audit']={'full_test_order_and_targets_match':True,'fivefold_coverage_images':4013,'QA':17998,'calibration_only_thresholds_recomputed':True,'all_public_replays_present':True,'conditional_one_training_seed':True,'known_components_not_patients':True,'dependencies':deps}
    dump(dest/'results.json',report,True);dump(dest/'analysis_completed.json',{'results_sha256':sha(dest/'results.json'),'visual_review_and_final_interpretation_pending':True},True)
    lines=['# Local noaux refinement results','','Fixed five-fold development-informed study; not independent clinical confirmation.','','| Variant | AP | raw BA | spec90 BA | spec90 sensitivity | spec90 specificity | QA |','|---|---:|---:|---:|---:|---:|---:|']
    for name,x in report['pooled'].items():lines.append(f"| {name} | {x['raw']['AP']:.6f} | {x['raw']['BA']:.6f} | {x['spec90']['BA']:.6f} | {x['spec90']['sensitivity']:.6f} | {x['spec90']['specificity']:.6f} | {x['raw']['QA']:.6f} |")
    lines += ['',f"Predeclared promotion rule passed: {report['promotion_rule_passed']}",'','Raw-output audits, source bindings and complete bootstrap draws: results.json. One seed; components not verified patients. Figure visual review, historical-baseline comparison and final closure still required.']
    (dest/'RESULTS.md').write_text('\n'.join(lines))

def supervise():
    # Separate supervisor started while quality runs; no duplicate quality coordinator.
    import subprocess
    dest=O/'postprocess';assert not dest.exists();dest.mkdir();dump(dest/'started.json',{'pid':os.getpid(),'wall':time.time(),'waiting_for':'execution_completed.json'},True)
    start=time.time();deadline=read(O/'coordinator.json')['deadline']
    while not (O/'execution_completed.json').exists():
        if (O/'failure.json').exists():dump(dest/'stopped.json',{'quality_failure':read(O/'failure.json')},True);return
        assert time.time()<deadline
        time.sleep(20)
    for fold in map(str,range(5)):
        for arm in ARMS:
            for mode in ('head','head_replay','operating'):
                log=dest/(mode+'_'+fold+'_'+arm+'.log')
                with log.open('x') as f:
                    p=subprocess.Popen([sys.executable,'-u',__file__,mode,fold,arm],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                    dump(dest/'progress.json',{'mode':mode,'fold':fold,'arm':arm,'pid':p.pid,'wall':time.time()})
                    try:code=p.wait(timeout=max(1,min(7200,deadline-time.time())))
                    except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=30);raise
                    assert code==0,str(log)
    analyze();dump(dest/'completed.json',{'analysis_generated':True,'main_visual_review_final_closure_pending':True},True)

if __name__=='__main__':
    torch.set_num_threads(8)
    try:
        release=O/'head_analysis_release.json'
        if release.exists():
            for p,h in read(release)['dependencies'].items():assert sha(p)==h,p
        globals()[sys.argv[1]](*sys.argv[2:])
    except Exception as e:
        dump(O/'postprocess_failure.json',{'error':repr(e),'argv':sys.argv,'wall':time.time(),'preserve_outputs':True});raise
