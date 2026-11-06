"""Independent TRAIN-transform reconstruction and descriptive paired contrasts."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.utils import compute_metrics,normalize_answer
from src.context_controls import load_component_lookup
from experiments.run_geometry_diagnostic_current_v3_1 import sources
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def main():
    root=Path.cwd();folder=root/'refactor_v2/reports/geometry_diagnostic_current_v3_1'
    output=folder/'transform_and_interpretation_audit.json';assert not output.exists()
    verified=json.loads((folder/'independent_verification.json').read_text())
    assert verified['passed'] and sha(folder/'report.json')==verified['report_sha256']
    assert sha(folder/'TRAIN_fitted_linear_controls.joblib')==verified['model_sha256']
    report=json.loads((folder/'report.json').read_text())
    train,val,full,canonical,arrays,_=sources()
    bundle=joblib.load(folder/'TRAIN_fitted_linear_controls.joblib')
    for key,source in (('binary_scaler','canonical_train'),('QA_scaler','QA_train')):
        recomputed=StandardScaler().fit(arrays[source])
        for attribute in ('mean_','scale_','var_'):
            assert np.array_equal(getattr(bundle[key],attribute),getattr(recomputed,attribute))
    tfidf=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1).fit([r['question'] for r in train])
    assert tfidf.vocabulary_==bundle['vectorizer'].vocabulary_ and np.array_equal(tfidf.idf_,bundle['vectorizer'].idf_)
    assert set(bundle['question_only'].classes_)=={normalize_answer(r['answer']) for r in train}
    endpoints={'geometry_only':report['clinical_geometry_only']['records']}
    vroot=root/'refactor_v2/reports/V_full321_development_v3_1'
    endpoints['V_real']=json.loads((vroot/'V_real.json').read_text())['records']
    bfolder=root/'refactor_v2/reports/fracatlas_scale_pilot_v3'
    b_report=json.loads((bfolder/'independent_evaluation.json').read_text())
    assert sha(bfolder/'independent_evaluation.json')==json.loads((bfolder/'independent_evaluation_progress.json').read_text())['report_sha256']
    endpoints['B_ROCO_seed42']=b_report['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']['records']
    identities=[(r['image_id'],r['gt']) for r in endpoints['geometry_only']]
    for rows in endpoints.values():assert [(r['image_id'],r['gt']) for r in rows]==identities
    cfg=json.loads((bfolder/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    binding=cfg['train']['component_batching'];lookup,_=load_component_lookup(binding['manifest'],binding['seal'],root)
    groups=sorted({lookup[i] for i,_ in identities});assert len(groups)==314
    indices={g:[j for j,(iid,_) in enumerate(identities) if lookup[iid]==g] for g in groups}
    keys=('fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity')
    draws={control:{k:[] for k in keys} for control in endpoints if control!='geometry_only'}
    rng=np.random.default_rng(20261020)
    for _ in range(500):
        take=[i for g in rng.choice(groups,len(groups),replace=True) for i in indices[g]]
        metrics={a:compute_metrics([r[i] for i in take],fracture_threshold=.5) for a,r in endpoints.items()}
        if 'fracture_auroc' not in metrics['geometry_only']:continue
        for control in draws:
            for k in keys:draws[control][k].append(metrics['geometry_only'][k]-metrics[control][k])
    confusions={}
    for name,rows in endpoints.items():
        counts={k:0 for k in ('TP','FN','TN','FP')}
        for r in rows:
            actual=normalize_answer(r['gt'])=='yes';pred=r['p_yes']>=.5
            counts[('TP' if pred else 'FN') if actual else ('FP' if pred else 'TN')]+=1
        confusions[name]=counts
    result={'passed_transform_audit':True,'scope':'exploratory same-development linear-control interpretation, NOT clinical superiority',
            'source_script_sha256':sha(__file__),'report_sha256':sha(folder/'report.json'),
            'TRAIN_scaler_mean_variance_scale_exact':True,'TRAIN_vectorizer_vocabulary_IDF_exact':True,
            'classifier_refits':0,'TRAIN_positive_prevalence':133/261,'modelval_positive_prevalence':57/321,
            'confusions_raw_half':confusions,
            'paired314component_bootstrap95_geometry_minus_reference':{a:{k:{'interval':np.quantile(v,[.025,.975]).tolist(),'valid_resamples':len(v)} for k,v in d.items()} for a,d in draws.items()},
            'conclusions':['Geometry is predictive in this development cohort, but does not establish lesion-pixel grounding or explain all neural performance',
                           'Geometry raw accuracy81.93percent comes with17false negatives versus5inVreal; do not select by overall accuracy alone',
                           'TRAIN class mix differs from modelval; class-balanced linear scores at0.5 are not a validated calibrated production decision rule',
                           'Detector upstream consumed pixels and2568TRAINannotations; downstream no-pixel does NOT mean the entire pipeline never saw images',
                           'Question-only and question-plus-geometry are small unmatched-capacity diagnostics, not a causal proof that neural VQA ignores pixels'],
            'calibration_or_test_used':False,'new_multimodal_training_updates':0,'long_training_authorized':False,'Final_Solution_selected':False}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
