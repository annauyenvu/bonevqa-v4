"""Evaluation-only automatic640 proposals for existing321modelval images.

Does not reuse a TRAIN-only/preparation scope, change production guards,
export reserved roles, or authorize training. Preserve every old artifact.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from src.data import build_dataset
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from src.detector_protocol import load_detector_manifest,verify_detector_checkpoint_binding
from src.detector_runtime import FractureDetectorRuntime
from src.preprocessing import canonicalize_raster,RASTER_PREPROCESSING_VERSION
from src.protocol import manifest,records_of
from src.provenance import validate_detector_proposals
from src.train import build_split
from experiments.detector640_checkpoint_v3 import load_bound_640
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.audit_vl_matched_controls_v3_1 import rounded,union_area,group_translate
from experiments.vl_bounded_proposal_contract_v3_1 import sha

ROOT=Path.cwd()
FOLDER=ROOT/'refactor_v2/reports/V_full321_proposals_v3_1'
SCOPE='existing_full321_modelvalidation_evaluation_ONLY'


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def context():
    lfolder=ROOT/'refactor_v2/reports/L_full321_development_v3_1'
    binding=json.loads((lfolder/'evaluation_binding.json').read_text())
    verified=json.loads((lfolder/'independent_artifact_verification.json').read_text())
    assert verified['passed'] and sha(lfolder/'report.json')==verified['report_sha256']
    cfgpath=ROOT/'refactor_v2/configs/revised_sealed_candidates_v3/b.json'
    assert sha(cfgpath)==binding['full_data_config_sha256']
    cfg=json.loads(cfgpath.read_text())
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    assert manifest(ds)==binding['full_modelval_manifest']
    rows=[r for r in records_of(ds) if r['question']=='Is there a fracture in this X-ray?']
    assert [r['image_id'] for r in rows]==binding['image_ids'] and len(rows)==321
    prior=ROOT/'refactor_v2/reports/vl_pilot_proposals640_v3_1/active_roles_manifest.json'
    source=json.loads(prior.read_text())
    parent,ids,digest=load_detector_manifest(source['parent_detector_manifest']['path'])
    assert digest==source['parent_detector_manifest']['sha256']
    assert set(binding['image_ids'])==set(ids['val'])
    assert not set(ids['val'])&(set(ids['train'])|set(ids['calibration'])|set(ids['test']))
    assert not {parent['component_by_image'][i] for i in ids['val']}&{parent['component_by_image'][i] for i in ids['train']}
    return rows,source,parent,ids,digest


class EvaluationDetector640(FractureDetectorRuntime):
    def __init__(self):
        _,source,parent,_,digest=context()
        self.checkpoint=Path(source['detector_checkpoint']['path'])
        assert sha(self.checkpoint)==source['detector_checkpoint']['sha256']
        self.model,payload=load_bound_640(self.checkpoint)
        verify_detector_checkpoint_binding(payload,parent,digest)
        assert payload['epoch']==2 and payload['optimizer_steps']==1408
        assert payload['preprocessing_version']==RASTER_PREPROCESSING_VERSION
        self.device=torch.device('cpu');self.image_size=640;self.score_threshold=.05;self.max_boxes=3
        self.model.eval()
        self.normalization_sha=hashlib.sha256(json.dumps(payload['normalization_spec'],sort_keys=True).encode()).hexdigest()
        self.transform=payload['detector_transform']


def validate(path):
    path=Path(path);artifact=validate_detector_proposals(path,require_complete=False)
    assert artifact['scope']==SCOPE and artifact['complete'] is False
    assert artifact['training_released'] is False and artifact['evaluation_role']=='model_val'
    assert artifact['complete_for_evaluation_role'] is True and artifact['coverage']==321
    assert artifact['score_threshold']==.05 and artifact['max_boxes']==3
    assert artifact['detector_transform']=={'min_size':[640],'max_size':640,'size_divisible':32}
    assert sha(artifact['runtime_export_source'])==artifact['runtime_export_source_sha256']
    assert sha(artifact['geometry_control_source'])==artifact['geometry_control_source_sha256']
    assert sha(artifact['full_QA_binding'])==artifact['full_QA_binding_sha256']
    assert sha(artifact['geometry_control'])==artifact['geometry_control_sha256']
    rows,source,_,_,_=context()
    assert artifact['checkpoint_sha256']==source['detector_checkpoint']['sha256']
    checkpoint=torch.load(artifact['checkpoint'],map_location='cpu',weights_only=False)
    assert artifact['normalization_spec_sha256']==hashlib.sha256(json.dumps(checkpoint['normalization_spec'],sort_keys=True).encode()).hexdigest()
    boxes=json.loads(path.read_text());side=json.loads(path.with_suffix('.provenance.json').read_text())
    control=json.loads(Path(artifact['geometry_control']).read_text())
    expected={r['image_id']:Path(FRACATLAS_ROOT)/r['image'] for r in rows}
    assert set(boxes)==set(control)==set(side['images'])==set(expected)
    for iid,items in boxes.items():
        image=side['images'][iid];w,h=image['width'],image['height']
        assert image['role']=='model_val' and image['input_image_path']==str(expected[iid])
        assert sha(expected[iid])==image['input_image_sha256']
        assert len(items)<=3 and all(0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h for b in items)
        real=rounded(items,w,h)
        assert control[iid]==group_translate(real,w,h) and union_area(real)==union_area(control[iid])
    return artifact


@torch.no_grad()
def prepare():
    check_environment();assert not FOLDER.exists()
    assert not (ROOT/'refactor_v2/reports/vl_bounded_quality_v3_1/ACTIVE.lock').exists()
    rows,source,_,ids,digest=context();FOLDER.mkdir()
    state={'status':'exporting','scope':SCOPE,'completed_images':0,'training_released':False}
    dump(FOLDER/'progress.json',state);started=time.perf_counter()
    try:
        runtime=EvaluationDetector640();boxes={};metadata={};control={}
        for row in rows:
            iid=row['image_id'];path=Path(FRACATLAS_ROOT)/row['image']
            with Image.open(path) as image:
                image.load()
                # Only original pixels enter detector; no GT fields/question.
                boxes[iid]=runtime.predict(image)
                w,h=canonicalize_raster(image).size
            metadata[iid]={'role':'model_val','width':w,'height':h,'input_image_path':str(path),'input_image_sha256':sha(path)}
            real=rounded(boxes[iid],w,h);control[iid]=group_translate(real,w,h)
            assert union_area(real)==union_area(control[iid])
            if len(boxes)%50==0:
                state['completed_images']=len(boxes);dump(FOLDER/'progress.json',state)
        dump(FOLDER/'boxes.json',boxes);dump(FOLDER/'group_translation_location_placebo.json',control)
        artifact={'schema_version':4,'scope':SCOPE,'complete':False,'complete_for_evaluation_role':True,
                  'evaluation_role':'model_val','training_released':False,'coverage':321,
                  'expected_coverage':len(set().union(*(set(x) for x in ids.values()))),
                  'coordinate_space':'exif_canonical_pixels','preprocessing_version':RASTER_PREPROCESSING_VERSION,
                  'proposal_sha256':sha(FOLDER/'boxes.json'),'score_threshold':.05,'max_boxes':3,
                  'checkpoint':source['detector_checkpoint']['path'],'checkpoint_sha256':source['detector_checkpoint']['sha256'],
                  'split_manifest':source['parent_detector_manifest']['path'],'split_manifest_sha256':digest,
                  'detector_transform':runtime.transform,'normalization_spec_sha256':runtime.normalization_sha,
                  'runtime_export_source':str(Path(__file__).resolve()),
                  'geometry_control_source':str(ROOT/'refactor_v2/experiments/audit_vl_matched_controls_v3_1.py'),
                  'full_QA_binding':str(ROOT/'refactor_v2/reports/L_full321_development_v3_1/evaluation_binding.json'),
                  'geometry_control':str(FOLDER/'group_translation_location_placebo.json')}
        for key in ('runtime_export_source','geometry_control_source','full_QA_binding','geometry_control'):
            artifact[key+'_sha256']=sha(artifact[key])
        dump((FOLDER/'boxes.json').with_suffix('.provenance.json'),{'artifact':artifact,'images':metadata})
        validate(FOLDER/'boxes.json')
        try:validate_detector_proposals(FOLDER/'boxes.json',require_complete=True)
        except RuntimeError as error:
            assert 'chưa phủ toàn bộ sealed splits' in str(error);rejection=str(error)
        else:raise AssertionError('Partial evaluation artifact passed production guard')
        old=json.loads((ROOT/'refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json').read_text())
        overlap={i:items for i,items in old.items() if i in boxes}
        assert len(overlap)==32 and all(boxes[i]==items for i,items in overlap.items())
        summary={'passed_preparation':True,'scope':SCOPE,'coverage':321,'nonempty':sum(bool(x) for x in boxes.values()),
                 'empty':sum(not x for x in boxes.values()),'old32_overlap_boxes_exact':True,
                 'all321_control_union_pixels_preserved':True,'proposal_sha256':sha(FOLDER/'boxes.json'),
                 'sidecar_sha256':sha((FOLDER/'boxes.json').with_suffix('.provenance.json')),
                 'fullfold_guard_rejection':rejection,'elapsed_seconds':time.perf_counter()-started,
                 'calibration_or_test_exported':False,'training_released':False}
        dump(FOLDER/'summary.json',summary)
        state.update(status='prepared_pending_independent_contract_verification',completed_images=321);dump(FOLDER/'progress.json',state)
        print(json.dumps(summary,indent=2))
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(FOLDER/'progress.json',state);raise


@torch.no_grad()
def verify():
    check_environment();output=FOLDER/'independent_contract_verification.json';assert not output.exists()
    source=FOLDER/'boxes.json';validate(source)
    boxes=json.loads(source.read_text());side=json.loads(source.with_suffix('.provenance.json').read_text())
    summary=json.loads((FOLDER/'summary.json').read_text())
    assert summary['passed_preparation'] and sha(source)==summary['proposal_sha256']
    rejected={}
    for name in ('missing_image','claimed_test_role','wrong_transform','wrong_normalization','wrong_image_hash','false_fullfold_complete','invalid_coordinates'):
        payload=copy.deepcopy(boxes);wrong=copy.deepcopy(side);iid=next(i for i,b in payload.items() if b)
        if name=='missing_image':payload.pop(iid);wrong['images'].pop(iid);wrong['artifact']['coverage']=320
        elif name=='claimed_test_role':wrong['images'][iid]['role']='test'
        elif name=='wrong_transform':wrong['artifact']['detector_transform']['min_size']=[320]
        elif name=='wrong_normalization':wrong['artifact']['normalization_spec_sha256']='0'*64
        elif name=='wrong_image_hash':wrong['images'][iid]['input_image_sha256']='0'*64
        elif name=='false_fullfold_complete':wrong['artifact']['complete']=True
        else:payload[iid][0][0]=-1
        fixture=FOLDER/('negative_'+name+'.json');dump(fixture,payload)
        wrong['artifact']['proposal_sha256']=sha(fixture);dump(fixture.with_suffix('.provenance.json'),wrong)
        try:validate(fixture)
        except (AssertionError,RuntimeError):rejected[name]=True
        else:raise AssertionError('Invalid fixture accepted: '+name)
    replay=[i for i,b in boxes.items() if b][:2]+[i for i,b in boxes.items() if not b][:2]
    assert len(replay)==4
    runtime=EvaluationDetector640()
    for iid in replay:
        with Image.open(side['images'][iid]['input_image_path']) as image:
            image.load();assert runtime.predict(image)==boxes[iid]
    dump(output,{'passed':True,'scope':SCOPE,'proposal_sha256':sha(source),'sidecar_sha256':sha(source.with_suffix('.provenance.json')),
                 'negative_cases_rejected':rejected,'fresh_process_image_only_replay_exact':replay,
                 'calibration_or_test_used':False,'training_released':False})
    print('Full321 evaluation-only proposal contract independently verified; no training release')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','verify']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else verify()
