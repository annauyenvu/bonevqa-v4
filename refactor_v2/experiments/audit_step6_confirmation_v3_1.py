"""Independent saved calibration/count/metric audit and paired component bootstrap."""
import json,sys,time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from experiments.prepare_step4_localization_v3_1 import ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step6_confirmation_v3_1';Q='Is there a fracture in this X-ray?'
PARAMS=dict(C=1.,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=42,fit_intercept=True,class_weight=None)
def metric(y,p):
    positive=y.astype(bool);d=p>=.5;tp=int((positive&d).sum());fn=int((positive&~d).sum());tn=int((~positive&~d).sum());fp=int((~positive&d).sum())
    sensitivity=tp/(tp+fn);specificity=tn/(tn+fp)
    ece=0.
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else p<=1)
        if mask.any():ece+=float(mask.mean())*abs(float(p[mask].mean())-float(y[mask].mean()))
    return {'AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),'sensitivity':sensitivity,'specificity':specificity,'BA':(sensitivity+specificity)/2,'Brier':float(((p-y)**2).mean()),'ECE10':ece,'TP':tp,'FN':fn,'TN':tn,'FP':fp}
def main():
    assert not (OUT/'independent_analysis.json').exists();started=time.perf_counter()
    complete=read(OUT/'execution_completed.json');assert complete['all_assessment_fresh_replays_passed']
    release=read(OUT/'assessment_release.json')
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    binding=read(OUT/'role_binding.json');cal_expected=[r for r in read(binding['reserved_paths']['calibration']) if r['question']==Q];test_expected=read(binding['reserved_paths']['test'])
    lookup=read(R/'step4_localization_v3_1/component_lookup.json');data={};points={};sources={};identities=None;diagnostics={}
    for seed in (42,43,44,45,46):
        for arm in complete['arms']:
            key=f'{seed}_{arm}';path=OUT/f'assessment_seed{key}.json';d=read(path);replay=read(OUT/f'assessment_seed{key}_replay.json')
            assert replay['passed'] and replay['source_sha256']==sha(path);sources[str(path)]=sha(path)
            assert d['checkpoint_sha256']==release['checkpoints'][str(seed)][arm]['sha256']
            for rows,expected in ((d['calibration_records'],cal_expected),(d['raw_records'],test_expected)):
                assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]==[(r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
            cal=d['calibration_records'];cy=np.array([r['gt']=='yes' for r in cal],int);cp=np.clip([r['p_yes'] for r in cal],1e-6,1-1e-6)
            estimator=LogisticRegression(**PARAMS).fit(np.log(cp/(1-cp)).reshape(-1,1),cy)
            fitted=d['calibration']
            if fitted['passed']:
                assert fitted['parameters']==PARAMS and fitted['slope']>0
                assert abs(float(estimator.coef_[0,0])-fitted['slope'])<1e-12 and abs(float(estimator.intercept_[0])-fitted['intercept'])<1e-12
                for raw,calibrated in zip(d['raw_records'],d['calibrated_records']):
                    if raw['question']!=Q:assert raw==calibrated
                    else:
                        p=float(np.clip(raw['p_yes'],1e-6,1-1e-6));z=fitted['slope']*np.log(p/(1-p))+fitted['intercept'];expected=float(np.exp(-np.logaddexp(0,-z)))
                        assert abs(calibrated['p_yes']-expected)<1e-12 and calibrated['pred']==('yes' if expected>=.5 else 'no')
            else:assert d['calibrated_records'] is None
            for mode in ('raw','calibrated'):
                rows=d[mode+'_records']
                if rows is None:continue
                canonical=[r for r in rows if r['question']==Q];ids=[r['image_id'] for r in canonical]
                assert len(ids)==len(set(ids))==803
                if identities is None:identities=ids
                assert identities==ids
                y=np.array([r['gt']=='yes' for r in canonical],int);p=np.array([r['p_yes'] for r in canonical]);assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                values=metric(y,p);saved=d[mode+'_metrics']
                names={'AP':'fracture_auprc','AUROC':'fracture_auroc','sensitivity':'fracture_sensitivity','specificity':'fracture_specificity','BA':'fracture_balanced_accuracy','Brier':'fracture_brier','ECE10':'fracture_ece10'}
                for name,k in names.items():assert abs(values[name]-saved[k])<1e-12
                data[(seed,arm,mode)]=(y,p);points[f'{key}_{mode}']=values
                diagnostics[f'{key}_{mode}']={'positive_probability_quantiles':np.quantile(p[y==1],[.1,.25,.5,.75,.9]).tolist(),'negative_probability_quantiles':np.quantile(p[y==0],[.1,.25,.5,.75,.9]).tolist(),'predicted_positive_fraction':float((p>=.5).mean()),'true_prevalence':float(y.mean()),'positive_below025':int(((y==1)&(p<.25)).sum()),'positive_025_to05':int(((y==1)&(p>=.25)&(p<.5)).sum()),'calibration_fit':fitted,'interpretation':'Descriptive score-position diagnostic at frozen0.5, not a causal explanation or threshold selection'}
    groups=sorted({lookup[i] for i in identities});index=[np.array([j for j,i in enumerate(identities) if lookup[i]==g]) for g in groups];rng=np.random.default_rng(20261023)
    draws={};valid=0
    for _ in range(2000):
        selected=np.concatenate([index[j] for j in rng.integers(0,len(groups),len(groups))]);sums={}
        for (seed,arm,mode),(y,p) in data.items():
            if len(set(y[selected]))<2:continue
            m=metric(y[selected],p[selected]);sums.setdefault((arm,mode),[]).append(m)
        for (arm,mode),mm in sums.items():
            if len(mm)!=5:continue
            for name in ('AP','AUROC','sensitivity','specificity','BA','Brier','ECE10'):
                draws.setdefault(f'{arm}_{mode}_{name}',[]).append(float(np.mean([r[name] for r in mm])))
        # Same resampled components across seeds, arms and raw/calibrated variants.
        for arm in complete['arms']:
            if (arm,'raw') in sums and (arm,'calibrated') in sums and len(sums[(arm,'raw')])==len(sums[(arm,'calibrated')])==5:
                for name in ('AP','BA','sensitivity','specificity','Brier','ECE10'):
                    draws.setdefault(f'{arm}_cal_minus_raw_{name}',[]).append(float(np.mean([m[name] for m in sums[(arm,'calibrated')]])-np.mean([m[name] for m in sums[(arm,'raw')]])))
            if arm!='B':
                for mode in ('raw','calibrated'):
                    if len(sums.get((arm,mode),[]))==len(sums.get(('B',mode),[]))==5:
                        for name in ('AP','BA','sensitivity','specificity'):
                            draws.setdefault(f'{arm}_minus_B_{mode}_{name}',[]).append(float(np.mean([m[name] for m in sums[(arm,mode)]])-np.mean([m[name] for m in sums[('B',mode)]])))
        valid+=1
    aggregate={}
    for arm in complete['arms']:
        for mode in ('raw','calibrated'):
            mm=[points[f'{seed}_{arm}_{mode}'] for seed in (42,43,44,45,46) if f'{seed}_{arm}_{mode}' in points]
            aggregate[f'{arm}_{mode}']={name:{'mean':float(np.mean([r[name] for r in mm])),'sample_SD':float(np.std([r[name] for r in mm],ddof=1)) if len(mm)>1 else None,'n_seeds':len(mm)} for name in ('AP','AUROC','sensitivity','specificity','BA','Brier','ECE10')} if mm else {}
    intervals={k:{'CI95':np.quantile(v,[.025,.975]).tolist(),'valid_draws':len(v)} for k,v in draws.items()}
    dump(OUT/'independent_analysis.json',{'passed':True,'per_seed':points,'score_position_diagnostics':diagnostics,'five_seed_aggregate':aggregate,'paired_component_intervals':intervals,'sources':sources,'elapsed_seconds':time.perf_counter()-started,'component_bootstrap_seed':20261023,'bootstrap_draws':2000,'scope':'Conditional on five fitted seeds; known components not verified patients; development-informed existing assessment, no independent clinical confirmation','same_images_not_stacked_as_independent':True,'production_released':False,'report_pending':True})
    print('Independent calibration/metric checks and fixed-five-seed component bootstrap complete')
if __name__=='__main__':main()
