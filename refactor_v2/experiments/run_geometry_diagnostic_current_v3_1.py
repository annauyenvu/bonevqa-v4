"""Fixed TRAIN-fitted downstream geometry/template diagnostic, not grounding.

Never call the historical geometry CLI (it evaluates reserved roles). Upstream
automatic detector used pixels and full TRAIN GT; these linear controls read
ONLY proposal geometry/dimensions and optionally raw question text.
"""
import argparse
import json
import time
import warnings
from datetime import datetime,timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix,hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scripts.run_geometry_only_control_v3 import geometry
from src.data.sealed_manifest import load_sealed_records
from src.context_controls import load_component_lookup
from src.endpoints import is_canonical_fracture_presence_question
from src.utils import compute_metrics,normalize_answer
from experiments.vl_bounded_proposal_contract_v3_1 import sha,validate_prepared_proposals
from experiments.V_full321_proposal_contract_v3_1 import validate as validate_full_proposals
from experiments.run_vl_bounded_pilot_v3_1 import check_environment

ROOT=Path.cwd();FOLDER=ROOT/'refactor_v2/reports/geometry_diagnostic_current_v3_1'
FEATURES=['presence','count','sum_normalized_area','max_normalized_area','mean_normalized_area',
          'mean_aspect','mean_center_x','mean_center_y','mean_normalized_width','mean_normalized_height']
PARAMS={'C':1.0,'class_weight':'balanced','solver':'lbfgs','max_iter':2000,'tol':1e-4,'random_state':42}


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def sources():
    pilot=ROOT/'refactor_v2/reports/fracatlas_scale_pilot_v3'
    train_path=pilot/'train.json';val_path=pilot/'model_val.json';seal=pilot/'data_seal.json'
    train=load_sealed_records(train_path,seal,ROOT);val=load_sealed_records(val_path,seal,ROOT)
    train_boxes_path=ROOT/'refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json'
    validate_prepared_proposals(train_boxes_path)
    val_boxes_path=ROOT/'refactor_v2/reports/V_full321_proposals_v3_1/boxes.json'
    validate_full_proposals(val_boxes_path)
    full_path=ROOT/'refactor_v2/reports/V_full321_development_v3_1/V_real.json'
    checked=ROOT/'refactor_v2/reports/V_full321_development_v3_1/independent_artifact_verification.json'
    verifier=json.loads(checked.read_text());assert verifier['passed']
    assert verifier['checks']['V_real']['records_sha256']==sha(full_path)
    full=json.loads(full_path.read_text())['records']
    train_boxes=json.loads(train_boxes_path.read_text());val_boxes=json.loads(val_boxes_path.read_text())
    train_meta=json.loads(train_boxes_path.with_suffix('.provenance.json').read_text())['images']
    val_meta=json.loads(val_boxes_path.with_suffix('.provenance.json').read_text())['images']
    canonical=[r for r in train if is_canonical_fracture_presence_question(r['question'])]
    assert len(train)==1173 and len(val)==142 and len(full)==321 and len(canonical)==261
    assert len({r['image_id'] for r in canonical})==261
    assert sum(normalize_answer(r['answer'])=='yes' for r in canonical)==133
    assert sum(normalize_answer(r['answer'])=='no' for r in canonical)==128
    train_ids={r['image_id'] for r in train};val_ids={r['image_id'] for r in full}
    assert not train_ids&val_ids and val_ids==set(val_boxes)
    assert {r['image_id'] for r in val}<=val_ids
    base=json.loads((pilot/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    bc=base['train']['component_batching'];lookup,_=load_component_lookup(bc['manifest'],bc['seal'],ROOT)
    assert not {lookup[i] for i in train_ids}&{lookup[i] for i in val_ids}
    # Strip all path/label/region fields: downstream features have a strict
    # whitelist and cannot read fracture-containing filenames or patient IDs.
    def design(rows,boxes,meta):
        matrix=np.asarray([geometry(boxes[r['image_id']],{'width':meta[r['image_id']]['width'],
                           'height':meta[r['image_id']]['height']}) for r in rows],dtype=np.float64)
        assert matrix.shape==(len(rows),10) and np.isfinite(matrix).all()
        return matrix
    arrays={'canonical_train':design(canonical,train_boxes,train_meta),
            'QA_train':design(train,train_boxes,train_meta),'QA_modelval':design(val,val_boxes,val_meta),
            'full321_modelval':design(full,val_boxes,val_meta)}
    paths=[train_path,val_path,seal,train_boxes_path,train_boxes_path.with_suffix('.provenance.json'),
           val_boxes_path,val_boxes_path.with_suffix('.provenance.json'),full_path,checked,
           ROOT/'refactor_v2/scripts/run_geometry_only_control_v3.py',Path(__file__).resolve()]
    return train,val,full,canonical,arrays,{str(p):sha(p) for p in paths}


def fit(model,x,y):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y)
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    return model


def predict(bundle,train,val,full,arrays):
    binary=bundle['binary'];scaler=bundle['binary_scaler']
    column=list(binary.classes_).index(1)
    full_prob=binary.predict_proba(scaler.transform(arrays['full321_modelval']))[:,column]
    full_records=[{**{k:r[k] for k in ('image_id','question','gt','answer_type')},
                   'pred':'yes' if p>=.5 else 'no','p_yes':float(p)} for r,p in zip(full,full_prob)]
    text=bundle['vectorizer'].transform([r['question'] for r in val])
    combined=hstack([text,csr_matrix(bundle['QA_scaler'].transform(arrays['QA_modelval']))]).tocsr()
    qa={}
    for name,x in (('question_only',text),('question_plus_geometry',combined)):
        model=bundle[name];predictions=model.predict(x);rows=[]
        probabilities=model.predict_proba(x);classes=list(model.classes_)
        yes=probabilities[:,classes.index('yes')];no=probabilities[:,classes.index('no')]
        canonical_prob=yes/np.maximum(yes+no,1e-15)
        for source,prediction,prob in zip(val,predictions,canonical_prob):
            row={'image_id':source['image_id'],'question':source['question'],'answer_type':source['answer_type'],
                 'gt':source['answer'],'pred':str(prediction)}
            if is_canonical_fracture_presence_question(row['question']):
                row.update(pred='yes' if prob>=.5 else 'no',p_yes=float(prob))
            rows.append(row)
        qa[name]={'records':rows,'metrics':compute_metrics(rows,fracture_threshold=.5),
                  'canonical_endpoint':'canonical question uses this SAME QA classifier normalized yes/no probabilities at0.5; question_only NEVER uses geometry'}
    return {'clinical_geometry_only':{'records':full_records,'metrics':compute_metrics(full_records,fracture_threshold=.5)},
            'QA142':qa}


def run():
    check_environment();assert not FOLDER.exists()
    assert not (ROOT/'refactor_v2/reports/vl_bounded_quality_v3_1/ACTIVE.lock').exists()
    train,val,full,canonical,arrays,digests=sources()
    FOLDER.mkdir()
    protocol={'scope':'current fold downstream geometry/template shortcut diagnostic ONLY',
              'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':digests,
              'TRAIN_QA':1173,'TRAIN_images':261,'TRAIN_positive_images':133,'TRAIN_negative_images':128,
              'modelvalidation_QA':142,'fullmodelvalidation_images':321,
              'binary_features':FEATURES,'fixed_parameters':PARAMS,
              'primary':'geometry-only canonical fracture classifier fit once on261TRAINimages',
              'secondary':['question-only QA classifier','question-plus-geometry QA classifier'],
              'TFIDF':{'lowercase':True,'ngram_range':[1,2],'min_df':1},
              'scaler_and_vectorizer_fit_on':'TRAIN only; binary scaler261images, QA scaler1173QA',
              'threshold':.5,'hyperparameter_search':False,'validation_used_for_fit_or_selection':False,
              'calibration_or_test_allowed':False,'fit_count':3,
              'upstream_exposure':'detector trained on2568TRAINimages; this is not detector-GT-exposure or capacity matched to B',
              'downstream_forbidden_inputs':['pixels','image_id/path as features','answer_type','region','annotation boxes/masks','GT at prediction time']}
    dump(FOLDER/'protocol.json',protocol)
    started=time.perf_counter();state={'status':'fitting_fixed_diagnostic','protocol_sha256':sha(FOLDER/'protocol.json')}
    dump(FOLDER/'progress.json',state)
    try:
        bundle={}
        binary_scaler=StandardScaler().fit(arrays['canonical_train'])
        y=np.asarray([int(normalize_answer(r['answer'])=='yes') for r in canonical])
        bundle['binary_scaler']=binary_scaler
        bundle['binary']=fit(LogisticRegression(**PARAMS),binary_scaler.transform(arrays['canonical_train']),y)
        vectorizer=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1)
        text=vectorizer.fit_transform([r['question'] for r in train]);bundle['vectorizer']=vectorizer
        qa_scaler=StandardScaler().fit(arrays['QA_train']);bundle['QA_scaler']=qa_scaler
        combined=hstack([text,csr_matrix(qa_scaler.transform(arrays['QA_train']))]).tocsr()
        answers=np.asarray([normalize_answer(r['answer']) for r in train])
        assert len(set(answers))==68
        bundle['question_only']=fit(LogisticRegression(**PARAMS),text,answers)
        bundle['question_plus_geometry']=fit(LogisticRegression(**PARAMS),combined,answers)
        fit_seconds=time.perf_counter()-started
        joblib.dump(bundle,FOLDER/'TRAIN_fitted_linear_controls.joblib')
        result=predict(bundle,train,val,full,arrays)
        records={**result,'protocol_sha256':sha(FOLDER/'protocol.json'),
                 'model_sha256':sha(FOLDER/'TRAIN_fitted_linear_controls.joblib'),
                 'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,
                 'runtime_versions':{'sklearn':sklearn.__version__,'numpy':np.__version__},
                 'fit_iterations':{k:v.n_iter_.tolist() for k,v in bundle.items() if isinstance(v,LogisticRegression)},
                 'limits':['Reused development diagnostic, not independent confirmation or visual grounding',
                           'Question-only and question-plus-geometry heads have different parameter counts; not an isolated capacity-matched causal experiment',
                           'Canonical clinical QA uses each QA classifier normalized yes/no probabilities; primary321 geometry-only classifier is separate',
                           'No neural checkpoint change, no threshold fitting, no Final Solution or production release'],
                 'multimodal_optimizer_updates':0,'calibration_or_test_used':False,'long_training_authorized':False}
        dump(FOLDER/'report.json',records)
        state.update(status='complete_pending_independent_reload',report_sha256=sha(FOLDER/'report.json'));dump(FOLDER/'progress.json',state)
        print(json.dumps({'fit_seconds':fit_seconds,'clinical_geometry_only':result['clinical_geometry_only']['metrics'],
                          'QA142':{k:v['metrics'] for k,v in result['QA142'].items()}},indent=2))
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(FOLDER/'progress.json',state);raise


def verify():
    check_environment();output=FOLDER/'independent_verification.json';assert not output.exists()
    protocol=json.loads((FOLDER/'protocol.json').read_text());report=json.loads((FOLDER/'report.json').read_text())
    progress=json.loads((FOLDER/'progress.json').read_text())
    assert progress['status']=='complete_pending_independent_reload' and sha(FOLDER/'report.json')==progress['report_sha256']
    assert sha(FOLDER/'protocol.json')==report['protocol_sha256']
    assert protocol['fixed_parameters']==PARAMS and protocol['binary_features']==FEATURES
    train,val,full,canonical,arrays,digests=sources()
    assert digests==protocol['source_sha256']
    assert sha(FOLDER/'TRAIN_fitted_linear_controls.joblib')==report['model_sha256']
    bundle=joblib.load(FOLDER/'TRAIN_fitted_linear_controls.joblib')
    assert int(bundle['binary_scaler'].n_samples_seen_)==261
    assert int(bundle['QA_scaler'].n_samples_seen_)==1173
    assert bundle['vectorizer'].vocabulary_==TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=1).fit([r['question'] for r in train]).vocabulary_
    # Recomputing a TRAIN-only vocabulary audit does not fit a classifier or
    # let validation text enter the stored vectorizer.
    replay=predict(bundle,train,val,full,arrays)
    assert replay['clinical_geometry_only']==report['clinical_geometry_only'] and replay['QA142']==report['QA142']
    for name in ('binary','question_only','question_plus_geometry'):
        assert all(bundle[name].get_params()[k]==v for k,v in PARAMS.items())
    checks={'passed':True,'scope':'fresh-process frozen-linear-model reload; no classifier refit',
            'report_sha256':sha(FOLDER/'report.json'),'model_sha256':report['model_sha256'],
            'clinical_scores_answers_metrics_exact':321,'QA_answers_scores_metrics_exact_per_head':142,
            'binary_scaler_TRAIN_samples':261,'QA_scaler_TRAIN_samples':1173,'TRAIN_only_vocabulary_reproduced':True,
            'calibration_or_test_used':False,'production_or_efficacy_release':False}
    dump(output,checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','verify']);args=parser.parse_args()
    run() if args.mode=='run' else verify()
