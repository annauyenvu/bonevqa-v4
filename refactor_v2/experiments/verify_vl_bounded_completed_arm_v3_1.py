"""Independent LAST296 reload on the same 142 development QA, not a test.

Must run after training exits; no checkpoint updates, reserved-role inference,
threshold fitting, epoch selection, or claim of module efficacy.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from src.models.bonevqa import BoneVQAModel
from src.data import collate_fn
from src.utils import compute_metrics, set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.vl_bounded_execution_contract_v3_1 import ARMS,validate_plan
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.run_vl_bounded_pilot_v3_1 import check_environment,setup_data


def hashes(state):
    return {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in state.items()}


@torch.no_grad()
def main(arm):
    check_environment()
    root=Path.cwd()
    folder=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    run=folder/arm
    destination=run/'independent_LAST296_verification.json'
    assert not destination.exists(), 'Preserve existing verification'
    assert not (folder/'ACTIVE.lock').exists(), 'Do not compete with active training'
    processes=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any('run_vl_bounded_pilot_v3_1.py --run' in row and 'Python' in row for row in processes.splitlines())
    progress=json.loads((run/'progress.json').read_text())
    assert progress['status']=='complete_pending_independent_evaluation' and progress['global_step']==296
    release=json.loads((folder/'execution_release.json').read_text())
    execution=json.loads((folder/'independent_execution_verification.json').read_text())
    assert execution['passed'] and execution['execution_release_sha256']==sha(folder/'execution_release.json')
    for path,digest in release['dependency_sha256'].items():assert sha(path)==digest
    plan=json.loads((root/'refactor_v2/reports/vl_bounded_execution_contract_v3_1'/(arm+'.json')).read_text())
    validate_plan(plan)
    assert sha(run/'last.pt')==progress['last_sha256']
    assert sha(run/'optimizer.pt')==progress['optimizer_sha256']
    started=time.perf_counter()
    payload=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    optimizer=torch.load(run/'optimizer.pt',map_location='cpu',weights_only=False)
    history=json.loads((run/'history.json').read_text())
    assert payload['config']==plan['model'] and len(payload['answer_vocab'])==68
    assert payload['extra']['history']==history and len(history)==4
    assert payload['extra']['global_step']==optimizer['global_step']==296
    assert payload['extra']['base_seed']==42 and payload['extra']['execution_release_sha256']==sha(folder/'execution_release.json')
    assert [r['global_epoch'] for r in history]==[0,1,2,3]
    assert [r['global_step'] for r in history]==[74,148,222,296]
    assert [r['stage'] for r in history]==([3]*4 if arm.startswith('V_') else [1,1,3,3])
    assert optimizer['phase_updates']==(296 if arm.startswith('V_') else 148)
    assert optimizer['scheduler']['last_epoch']==optimizer['phase_updates']
    assert finite_tree(payload['state']) and finite_tree(optimizer) and finite_tree(history)
    for phase in plan['phases']:
        qc=json.loads((run/('stage'+str(phase['stage'])+'_module_qc.json')).read_text())
        assert qc['finite'] and all(v>0 for v in qc['gradient_max'].values())
        assert all(v>0 for v in qc['actual_update_max'].values())
    expected=json.loads((run/'model_val_epoch3.json').read_text())
    assert len(expected)==142
    initial=release['fresh_initializations'][arm]
    assert sha(initial['path'])==initial['sha256']
    expected_weights=hashes(payload['state'])
    del payload,optimizer;gc.collect()
    datasets,_,_=setup_data(plan)
    ds=datasets['model_val']
    location=json.loads(Path(plan['controls']['geometry_path']).read_text())
    set_seed(90210)
    model=BoneVQAModel.load(str(run/'last.pt'),device='cpu').eval()
    assert hashes(model.trainable_state_dict())==expected_weights
    actual=[]
    for start in range(0,len(ds),4):
        batch=collate_fn([ds[i] for i in range(start,min(start+4,len(ds)))])
        # Public prediction receives no GT answer/type/region, IDs or masks.
        public={'images':batch['images'],'questions':batch['questions']}
        if arm.startswith('V_'):
            public['boxes']=([location[i] or None for i in batch['image_ids']]
                             if arm.endswith('location_placebo') else batch['boxes'])
        predictions=model.predict_batch(public)
        for prediction,answer,kind,question,image in zip(predictions,batch['answers'],batch['answer_types'],batch['questions'],batch['image_ids']):
            row={'pred':prediction['answer'],'gt':answer,'answer_type':kind,'question':question,'image_id':image}
            if 'p_yes' in prediction:row['p_yes']=prediction['p_yes']
            actual.append(row)
    assert len(actual)==142
    score_error=0.
    for observed,saved in zip(actual,expected):
        assert {k:v for k,v in observed.items() if k!='p_yes'}=={k:v for k,v in saved.items() if k!='p_yes'}
        assert ('p_yes' in observed)==('p_yes' in saved)
        if 'p_yes' in observed:score_error=max(score_error,abs(observed['p_yes']-saved['p_yes']))
    assert score_error<=1e-6, 'Frozen replay score mismatch'
    metrics=compute_metrics(actual)
    metric_error=max(abs(v-history[-1][k]) for k,v in metrics.items() if isinstance(v,(float,int)))
    assert metric_error<=1e-6 and hashes(model.trainable_state_dict())==expected_weights
    report={'passed':True,'arm':arm,'scope':'LAST296 same142modelval QA replay; not independent holdout',
            'checkpoint_sha256':progress['last_sha256'],'optimizer_sha256':progress['optimizer_sha256'],
            'global_step':296,'fresh_process_constructor_seed':90210,
            'all_trainable_weights_exact_and_unchanged':True,'answers_exact':142,
            'max_p_yes_error':score_error,'max_metric_error':metric_error,'metrics':metrics,
            'public_input_keys':['images','questions','boxes_from_automatic_detector_or_declared_placebo'] if arm.startswith('V_') else ['images','questions'],
            'module_gradient_and_update_QC_passed':True,
            'train_wall_seconds':progress['elapsed_seconds'],'verification_seconds':time.perf_counter()-started,
            'clinical_threshold':.5,'calibration_or_test_used':False,
            'limitations':['Reused32image development subset includes only5positiveimages',
                           'Full321Vfracture evaluation not done; automatic proposals currently cover32modelvalimages',
                           'Matched controls/multiple seeds/independent confirmation required before module contribution claim'],
            'efficacy_or_production_release':False}
    (run/'independent_LAST296_predictions.json').write_text(json.dumps(actual,indent=2)+'\n')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    main(parser.parse_args().arm)
