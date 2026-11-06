"""Fixed LAST V/control on321inspected modelval images, automatic proposals only."""
import gc
import hashlib
import json
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from src.context_controls import load_component_lookup
from src.data import build_dataset,collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.preprocessing import canonicalize_raster
from src.train import build_split
from src.utils import compute_metrics,set_seed
from experiments.V_full321_proposal_contract_v3_1 import FOLDER,context,validate
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def hashes(model):
    return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}


@torch.no_grad()
def main():
    check_environment();root=Path.cwd();parent=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    output=root/'refactor_v2/reports/V_full321_development_v3_1'
    assert not output.exists() and not (parent/'ACTIVE.lock').exists()
    release=json.loads((parent/'execution_release.json').read_text())
    for path,digest in release['dependency_sha256'].items():assert sha(path)==digest
    artifact=validate(FOLDER/'boxes.json')
    verified=json.loads((FOLDER/'independent_contract_verification.json').read_text())
    assert verified['passed'] and verified['proposal_sha256']==sha(FOLDER/'boxes.json')
    assert verified['sidecar_sha256']==sha((FOLDER/'boxes.json').with_suffix('.provenance.json'))
    assert all(verified['negative_cases_rejected'].values())
    boxes=json.loads((FOLDER/'boxes.json').read_text())
    metadata=json.loads((FOLDER/'boxes.json').with_suffix('.provenance.json').read_text())['images']
    for image in metadata.values():
        with Image.open(image['input_image_path']) as pixels:
            assert canonicalize_raster(pixels).size==(image['width'],image['height'])
    location=json.loads(Path(artifact['geometry_control']).read_text())
    cfg=json.loads((root/'refactor_v2/configs/revised_sealed_candidates_v3/b.json').read_text())
    ds=build_split(build_dataset,cfg['data']['dataset'],'model_val',None,True,cfg['data']['loader_kwargs'])
    indices=[i for i,r in enumerate(records_of(ds)) if r['question']=='Is there a fracture in this X-ray?']
    rows,_,_,_,_=context()
    assert [records_of(ds)[i] for i in indices]==rows
    old_folder=root/'refactor_v2/reports/fracatlas_scale_pilot_v3'
    old=json.loads((old_folder/'independent_evaluation.json').read_text())
    assert sha(old_folder/'independent_evaluation.json')==json.loads((old_folder/'independent_evaluation_progress.json').read_text())['report_sha256']
    b=old['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']
    assert [(r['image_id'],r['answer']) for r in rows]==[(r['image_id'],r['gt']) for r in b['records']]
    pilot=json.loads((old_folder/'fracatlas_B_roco_scale_pilot_v3.json').read_text())
    batch_binding=pilot['train']['component_batching']
    lookup,_=load_component_lookup(batch_binding['manifest'],batch_binding['seal'],root)
    arms=['V_real','V_location_placebo'];reports={}
    for arm in arms:
        reports[arm]=json.loads((parent/arm/'independent_LAST296_verification.json').read_text())
        assert reports[arm]['passed'] and reports[arm]['global_step']==296
        assert sha(parent/arm/'last.pt')==reports[arm]['checkpoint_sha256']
    output.mkdir();started=time.perf_counter()
    binding={'scope':'evaluation_ONLY_existing321modelvalidation; NOT independent test or training',
             'source_script_sha256':sha(__file__),'proposal_sha256':sha(FOLDER/'boxes.json'),
             'proposal_sidecar_sha256':sha((FOLDER/'boxes.json').with_suffix('.provenance.json')),
             'independent_proposal_verification_sha256':sha(FOLDER/'independent_contract_verification.json'),
             'checkpoint_sha256':{a:reports[a]['checkpoint_sha256'] for a in arms},
             'image_ids':[r['image_id'] for r in rows],
             'B_reference_report_sha256':sha(old_folder/'independent_evaluation.json'),
             'public_input_keys':['images','questions','automatic_boxes_or_declared_group_location_placebo'],
             'threshold':.5,'training_updates':0,'calibration_or_test_allowed':False}
    binding['all321_canonical_dimensions_independently_checked']=True
    dump(output/'evaluation_binding.json',binding)
    state={'status':'running','scope':binding['scope'],'started_utc':datetime.now(timezone.utc).isoformat()}
    dump(output/'progress.json',state);results={}
    try:
        for arm in arms:
            set_seed(90210);model=BoneVQAModel.load(str(parent/arm/'last.pt'),device='cpu').eval()
            assert model.cfg['prompt_variant']=='V' and not model.cfg['use_region_in_prompt']
            assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
            before=hashes(model);predictions=[];active=boxes if arm=='V_real' else location
            for offset in range(0,321,4):
                batch=collate_fn([ds[i] for i in indices[offset:offset+4]])
                public={'images':batch['images'],'questions':batch['questions'],
                        'boxes':[active[i] or None for i in batch['image_ids']]}
                pred=model.predict_batch(public)
                for j,item in enumerate(pred):
                    predictions.append({'image_id':batch['image_ids'][j],'question':batch['questions'][j],
                                        'gt':batch['answers'][j],'answer_type':batch['answer_types'][j],
                                        'pred':item['answer'],'p_yes':item['p_yes']})
                state.update(arm=arm,completed_images=len(predictions));dump(output/'progress.json',state)
            assert hashes(model)==before and sha(parent/arm/'last.pt')==reports[arm]['checkpoint_sha256']
            assert [(r['image_id'],r['gt']) for r in predictions]==[(r['image_id'],r['gt']) for r in b['records']]
            small=json.loads((parent/arm/'independent_LAST296_predictions.json').read_text())
            small={r['image_id']:r for r in small if r['question']=='Is there a fracture in this X-ray?'}
            error=max(abs(r['p_yes']-small[r['image_id']]['p_yes']) for r in predictions if r['image_id'] in small)
            assert error<=1e-4
            results[arm]={'metrics':compute_metrics(predictions,fracture_threshold=.5),
                          'records':predictions,'weights_unchanged':True,'overlap32_max_probability_error':error}
            dump(output/(arm+'.json'),results[arm]);del model;gc.collect()
        endpoints={a:r['records'] for a,r in results.items()};endpoints['B_ROCO_seed42_descriptive_only']=b['records']
        groups=sorted({lookup[r['image_id']] for r in b['records']})
        ids={g:[i for i,r in enumerate(b['records']) if lookup[r['image_id']]==g] for g in groups}
        keys=['fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity']
        diffs={a:{k:[] for k in keys} for a in endpoints if a!='V_real'}
        rng=np.random.default_rng(20261020)
        for _ in range(500):
            take=[i for g in rng.choice(groups,len(groups),replace=True) for i in ids[g]]
            metrics={a:compute_metrics([r[i] for i in take],fracture_threshold=.5) for a,r in endpoints.items()}
            if 'fracture_auroc' not in metrics['V_real']:continue
            for control in diffs:
                for k in keys:diffs[control][k].append(metrics['V_real'][k]-metrics[control][k])
        report={'scope':binding['scope'],'evaluation_binding_sha256':sha(output/'evaluation_binding.json'),
                'results':{a:{k:v for k,v in r.items() if k!='records'} for a,r in results.items()},
                'B_ROCO_seed42_descriptive_only_metrics':b['metrics'],
                'paired_component_bootstrap95_real_minus_controls':{a:{k:{'interval':np.quantile(v,[.025,.975]).tolist(),'valid_resamples':len(v)} for k,v in d.items()} for a,d in diffs.items()},
                'images':321,'positive_images':57,'component_count':len(groups),'raw_threshold':.5,
                'elapsed_seconds':time.perf_counter()-started,'calibration_or_test_used':False,'training_updates':0,
                'limitations':['Reused inspected development, single seed, not independent confirmation',
                               'Location control preserves group shape priors; not a perfect semantic null',
                               'B not capacity/extra-detector-GT/compute matched; descriptive reference only',
                               'Binary fracture endpoint: no OPEN questions evaluated',
                               'No threshold fitting, new training, Final Solution selection or production release']}
        dump(output/'report.json',report)
        state.update(status='complete_development_only',report_sha256=sha(output/'report.json'),finished_utc=datetime.now(timezone.utc).isoformat());dump(output/'progress.json',state)
        print(json.dumps(report,indent=2))
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(output/'progress.json',state);raise


if __name__=='__main__':main()
