"""Three verified L LAST checkpoints on the existing full model-val cohort.

Canonical fracture question only, original-image/question inference, no new
training, no calibration or test access, and no independent-confirmation claim.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

import numpy as np
import torch
from src.context_controls import load_component_lookup
from src.data import build_dataset,collate_fn
from src.detector_protocol import load_detector_manifest
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest,records_of
from src.train import build_split
from src.utils import compute_metrics,set_seed
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def hashes(model):
    return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}


@torch.no_grad()
def main():
    check_environment()
    root=Path.cwd();parent=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    output=root/'refactor_v2/reports/L_full321_development_v3_1'
    assert not output.exists(), 'Preserve existing evaluation, no overwrite or automatic retry'
    assert not (parent/'ACTIVE.lock').exists()
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any('Python' in r and 'run_vl_bounded_pilot_v3_1.py --run' in r for r in commands.splitlines())
    release=json.loads((parent/'execution_release.json').read_text())
    for path,digest in release['dependency_sha256'].items():assert sha(path)==digest
    continuation=json.loads((parent/'remaining_VL_continuation_progress.json').read_text())
    assert continuation['status']=='five_bounded_arms_verified_development_comparisons_complete_NOTfull_step3'
    arms=['L_real','L_original_teacher','L_location_placebo']
    verified={a:json.loads((parent/a/'independent_LAST296_verification.json').read_text()) for a in arms}
    for arm,report in verified.items():
        assert report['passed'] and report['global_step']==296 and not report['calibration_or_test_used']
        assert sha(parent/arm/'last.pt')==report['checkpoint_sha256']
    base_path=root/'refactor_v2/configs/revised_sealed_candidates_v3/b.json'
    base=json.loads(base_path.read_text())
    # This config is used ONLY to load the previously sealed full model_val
    # records, never its historical initialization/model/training recipe.
    ds=build_split(build_dataset,base['data']['dataset'],'model_val',None,True,base['data']['loader_kwargs'])
    indices=[i for i,r in enumerate(records_of(ds)) if r['question']=='Is there a fracture in this X-ray?']
    selected=[records_of(ds)[i] for i in indices]
    assert len(ds)==1432 and len(indices)==321 and len({r['image_id'] for r in selected})==321
    pilot=json.loads((root/'refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json').read_text())
    binding=pilot['train']['component_batching']
    lookup,_=load_component_lookup(binding['manifest'],binding['seal'],root)
    active=json.loads((root/'refactor_v2/reports/vl_pilot_proposals640_v3_1/active_roles_manifest.json').read_text())
    _,detector_ids,_=load_detector_manifest(active['parent_detector_manifest']['path'])
    assert {r['image_id'] for r in selected}==set(detector_ids['val'])
    assert not ({r['image_id'] for r in selected}&set(detector_ids['train']))
    assert not ({lookup[r['image_id']] for r in selected}&{lookup[i] for i in detector_ids['train']})
    assert sum(str(r['answer']).strip().lower()=='yes' for r in selected)==57
    assert sum(str(r['answer']).strip().lower()=='no' for r in selected)==264
    baseline_folder=root/'refactor_v2/reports/fracatlas_scale_pilot_v3'
    baseline=json.loads((baseline_folder/'independent_evaluation.json').read_text())
    assert sha(baseline_folder/'independent_evaluation.json')==json.loads((baseline_folder/'independent_evaluation_progress.json').read_text())['report_sha256']
    b=baseline['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']
    assert [(r['image_id'],r['answer']) for r in selected]==[(r['image_id'],r['gt']) for r in b['records']]
    output.mkdir()
    execution={'scope':'read-only existing321modelval canonical fracture endpoint; NOT training or independent test',
               'source_script_sha256':sha(__file__),'full_data_config_sha256':sha(base_path),
               'full_modelval_manifest':manifest(ds),'image_ids':[r['image_id'] for r in selected],
               'checkpoint_sha256':{a:verified[a]['checkpoint_sha256'] for a in arms},
               'prior_B_report_sha256':sha(baseline_folder/'independent_evaluation.json'),
               'input_keys_only':['images','questions'],'classification_threshold':.5,
               'training_updates':0,'calibration_or_test_allowed':False}
    dump(output/'evaluation_binding.json',execution)
    state={'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'scope':execution['scope']}
    dump(output/'progress.json',state);started=time.perf_counter();results={}
    try:
        for arm in arms:
            set_seed(90210)
            model=BoneVQAModel.load(str(parent/arm/'last.pt'),device='cpu').eval()
            assert model.cfg['lens_training_only'] and not model.cfg['use_region_in_prompt']
            assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
            before=hashes(model);rows=[]
            for offset in range(0,321,4):
                batch=collate_fn([ds[i] for i in indices[offset:offset+4]])
                predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
                for j,pred in enumerate(predictions):
                    rows.append({'image_id':batch['image_ids'][j],'question':batch['questions'][j],
                                 'gt':batch['answers'][j],'answer_type':batch['answer_types'][j],
                                 'pred':pred['answer'],'p_yes':pred['p_yes']})
                state.update(arm=arm,completed_images=len(rows));dump(output/'progress.json',state)
            assert hashes(model)==before and sha(parent/arm/'last.pt')==verified[arm]['checkpoint_sha256']
            assert [(r['image_id'],r['gt']) for r in rows]==[(r['image_id'],r['gt']) for r in b['records']]
            # Existing 32-image replay provides an overlapping endpoint check
            # across different batch composition; do not tune on disagreement.
            small=json.loads((parent/arm/'independent_LAST296_predictions.json').read_text())
            small={r['image_id']:r for r in small if r['question']=='Is there a fracture in this X-ray?'}
            overlap=max(abs(r['p_yes']-small[r['image_id']]['p_yes']) for r in rows if r['image_id'] in small)
            assert overlap<=1e-4
            metrics=compute_metrics(rows,fracture_threshold=.5)
            assert metrics['n']==metrics['n_fracture_presence']==321
            results[arm]={'metrics':metrics,'records':rows,'weights_unchanged':True,'overlap32_max_probability_error':overlap}
            dump(output/(arm+'.json'),results[arm]);del model;gc.collect()
        endpoints={a:r['records'] for a,r in results.items()}
        endpoints['B_ROCO_seed42_descriptive_only']=b['records']
        groups=sorted({lookup[r['image_id']] for r in b['records']})
        ids={g:[i for i,r in enumerate(b['records']) if lookup[r['image_id']]==g] for g in groups}
        keys=['fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity']
        differences={a:{k:[] for k in keys} for a in endpoints if a!='L_real'}
        rng=np.random.default_rng(20261020)
        for _ in range(500):
            take=[i for g in rng.choice(groups,len(groups),replace=True) for i in ids[g]]
            metrics={a:compute_metrics([rows[i] for i in take],fracture_threshold=.5) for a,rows in endpoints.items()}
            if 'fracture_auroc' not in metrics['L_real']:continue
            for control in differences:
                for k in keys:differences[control][k].append(metrics['L_real'][k]-metrics[control][k])
        report={'scope':execution['scope'],'evaluation_binding_sha256':sha(output/'evaluation_binding.json'),
                'results':{a:{k:v for k,v in r.items() if k!='records'} for a,r in results.items()},
                'B_ROCO_seed42_descriptive_only_metrics':b['metrics'],
                'paired_component_bootstrap95_real_minus_controls':{a:{k:{'interval':np.quantile(v,[.025,.975]).tolist(),'valid_resamples':len(v)} for k,v in d.items()} for a,d in differences.items()},
                'component_count':len(groups),'images':321,'positive_images':57,'raw_threshold':.5,
                'elapsed_seconds':time.perf_counter()-started,'calibration_or_test_used':False,'training_updates':0,
                'limitations':['Already inspected model-validation, not independent confirmation',
                               'Single seed; L controls phase/capacity/view matched, B not phase/capacity/detector-GT-exposure matched',
                               'V full321 not evaluated; requires separately bound automatic modelval proposals',
                               'No threshold search/calibration fitting or production/Final Solution release']}
        dump(output/'report.json',report)
        state.update(status='complete_development_only',report_sha256=sha(output/'report.json'),finished_utc=datetime.now(timezone.utc).isoformat());dump(output/'progress.json',state)
        print(json.dumps(report,indent=2))
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(output/'progress.json',state);raise


if __name__=='__main__':main()
