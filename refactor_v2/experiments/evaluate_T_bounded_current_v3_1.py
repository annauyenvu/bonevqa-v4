"""Predeclared T LAST296 development evaluation and fresh-process replay.

No fitting, reserved-role inference, checkpoint selection, or training release.
Prepare while training is running; run only after all three sealed arms finish.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_split
from src.utils import compute_metrics, set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree
from experiments.run_T_bounded_current_v3_1 import FOLDER, BASE, STYLES, data_and_schedule, hashes
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha

ROOT = Path.cwd()
OUT = ROOT/'refactor_v2/reports/T_independent_quality_current_v3_1'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def cohorts():
    base, datasets, _ = data_and_schedule()
    full_config = ROOT/'refactor_v2/configs/revised_sealed_candidates_v3/b.json'
    full = json.loads(full_config.read_text())
    ds = build_split(build_dataset, full['data']['dataset'], 'model_val', None, True, full['data']['loader_kwargs'])
    indices = [i for i, r in enumerate(records_of(ds)) if r['question']=='Is there a fracture in this X-ray?']
    selected = [records_of(ds)[i] for i in indices]
    assert len(ds)==1432 and len(indices)==321 and len({r['image_id'] for r in selected})==321
    assert sum(r['answer'].strip().lower()=='yes' for r in selected)==57
    from src.context_controls import load_component_lookup
    binding = base['train']['component_batching']
    lookup, _ = load_component_lookup(binding['manifest'], binding['seal'], ROOT)
    train_groups = {lookup[r['image_id']] for r in records_of(datasets['train'])}
    assert not train_groups & {lookup[r['image_id']] for r in selected}
    return datasets['model_val'], ds, indices, lookup, full_config


def prepare():
    check_environment()
    assert not OUT.exists(), 'Preserve existing evaluation preparation'
    qa, full, indices, _, full_config = cohorts()
    dependencies = {str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for p in [Path(__file__), FOLDER/'quality_evaluation_protocol.json', FOLDER/'preflight_report.json',
              FOLDER/'independent_preflight_verification.json', full_config,
              ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py',
              ROOT/'refactor_v2/experiments/evaluate_pmc_objective_pilot_v3_2.py',
              ROOT/'refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py',
              ROOT/'refactor_v2/experiments/vl_bounded_proposal_contract_v3_1.py']:
        dependencies[str(p.resolve())] = sha(p)
    baseline = BASE/'independent_evaluation.json'
    assert sha(baseline)==json.loads((BASE/'independent_evaluation_progress.json').read_text())['report_sha256']
    dependencies[str(baseline)] = sha(baseline)
    OUT.mkdir()
    dump(OUT/'evaluation_plan.json', {
        'created_utc':now(), 'dependencies_sha256':dependencies,
        'QA_manifest':manifest(qa), 'full_modelval_manifest':manifest(full),
        'canonical_image_ids':[records_of(full)[i]['image_id'] for i in indices],
        'styles':list(STYLES), 'checkpoint':'LAST296_only',
        'secondary':'clinical_checkpoint_inference_clinical_paraphrase_on142QA_only',
        'bootstrap_resamples':500, 'bootstrap_seed':20261020,
        'public_input_keys':['images','questions'], 'raw_threshold':.5,
        'calibration_or_test_allowed':False, 'training_updates':0,
        'independent_confirmation':False, 'long_training_authorized':False})
    print('Prepared immutable evaluation plan; no model job or training launched')


def validate_ready():
    check_environment()
    plan = json.loads((OUT/'evaluation_plan.json').read_text())
    for path, digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest, path
    assert not (FOLDER/'ACTIVE.lock').exists(), 'Wait for serial training, do not compete'
    progress = json.loads((FOLDER/'progress.json').read_text())
    assert progress['status']=='three_T_arms_complete_pending_independent_quality_evaluation'
    assert progress['completed_styles']==list(STYLES)
    commands = subprocess.check_output(['ps','-axo','command'], text=True)
    assert not any(' -m src.train ' in row for row in commands.splitlines())
    qa, full, indices, lookup, _ = cohorts()
    assert manifest(qa)==plan['QA_manifest'] and manifest(full)==plan['full_modelval_manifest']
    assert [records_of(full)[i]['image_id'] for i in indices]==plan['canonical_image_ids']
    preflight = json.loads((FOLDER/'preflight_report.json').read_text())
    checkpoints = {}
    for style in STYLES:
        info = preflight['configs'][style]
        assert sha(info['path'])==info['sha256']
        cfg = json.loads(Path(info['path']).read_text())
        target = ROOT/cfg['checkpoint_dir']/cfg['run_name']
        done = json.loads((FOLDER/(style+'_completion.json')).read_text())
        assert done['complete'] and sha(target/'last.pt')==done['last_sha256']
        assert sha(target/'optimizer.pt')==done['optimizer_sha256']
        payload = torch.load(target/'last.pt', map_location='cpu', weights_only=False)
        optimizer = torch.load(target/'optimizer.pt', map_location='cpu', weights_only=False)
        assert payload['config']==cfg['model'] and payload['answer_vocab']==preflight['answer_vocab']
        assert payload['extra']['global_step']==optimizer['global_step']==296
        assert len(payload['state'])==263 and finite_tree(payload['state']) and finite_tree(optimizer)
        assert len(done['history'])==4 and done['history'][-1]['val_n']==142
        checkpoints[style] = {'path':str(target/'last.pt'), 'sha256':done['last_sha256'], 'history':done['history']}
        del payload, optimizer
    gc.collect()
    return plan, qa, full, indices, lookup, checkpoints


@torch.no_grad()
def predict(model, ds, indices):
    before = hashes(model.trainable_state_dict())
    rows = []
    for offset in range(0,len(indices),4):
        batch = collate_fn([ds[i] for i in indices[offset:offset+4]])
        predictions = model.predict_batch({'images':batch['images'],'questions':batch['questions']})
        for j,pred in enumerate(predictions):
            rows.append({'image_id':batch['image_ids'][j], 'question':batch['questions'][j],
                         'gt':batch['answers'][j], 'answer_type':batch['answer_types'][j],
                         'pred':pred['answer'], 'p_yes':pred['p_yes']})
    assert hashes(model.trainable_state_dict())==before
    return {'records':rows, 'metrics':compute_metrics(rows,fracture_threshold=.5), 'weights_unchanged':True}


def evaluate(qa, full, indices, checkpoints, seed):
    for style in STYLES:
        info = checkpoints[style]
        set_seed(seed)
        model = BoneVQAModel.load(info['path'],device='cpu').eval()
        assert model.cfg['textual_prompt_style']==style and not model.cfg['use_region_in_prompt']
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt'))
        result = {'QA142':predict(model,qa,list(range(len(qa)))), 'fracture321':predict(model,full,indices)}
        for key,value in result['QA142']['metrics'].items():
            assert abs(value-info['history'][-1]['val_'+key])<=1e-6, (style,key)
        if style=='clinical':
            original = model.cfg['textual_prompt_style']
            try:
                model.cfg['textual_prompt_style']='clinical_paraphrase'
                result['secondary_inference_paraphrase142']=predict(model,qa,list(range(len(qa))))
            finally:
                model.cfg['textual_prompt_style']=original
        assert sha(info['path'])==info['sha256']
        del model
        gc.collect()
        yield style,result


def replay():
    _,qa,full,indices,_,checkpoints = validate_ready()
    errors = {}
    for style,result in evaluate(qa,full,indices,checkpoints,90211):
        saved = json.loads((OUT/(style+'.json')).read_text())
        assert saved==result, 'Fresh-constructor public predictions/metrics must replay exactly'
        errors[style] = 0.0
    dump(OUT/'fresh_process_replay.json', {'passed':True,'constructor_seed':90211,
         'all_records_and_metrics_exact':True,'max_probability_error':errors,
         'checkpoint_sha256':{s:x['sha256'] for s,x in checkpoints.items()},
         'training_updates':0,'calibration_or_test_used':False})


def bootstrap(endpoints, primary, lookup, keys):
    reference = endpoints[primary]
    identities = [(r['image_id'],r['question'],r['gt']) for r in reference]
    for records in endpoints.values():
        assert identities==[(r['image_id'],r['question'],r['gt']) for r in records]
    groups = sorted({lookup[r['image_id']] for r in reference})
    positions = {g:[i for i,r in enumerate(reference) if lookup[r['image_id']]==g] for g in groups}
    differences = {a:{k:[] for k in keys} for a in endpoints if a!=primary}
    rng = np.random.default_rng(20261020)
    for _ in range(500):
        take = [i for g in rng.choice(groups,len(groups),replace=True) for i in positions[g]]
        metrics = {a:compute_metrics([rows[i] for i in take],fracture_threshold=.5) for a,rows in endpoints.items()}
        # Clinical one-class resamples have undefined AUROC; explicitly omit.
        if not all(k in m for m in metrics.values() for k in keys):
            continue
        for arm in differences:
            for key in keys:
                differences[arm][key].append(metrics[primary][key]-metrics[arm][key])
    return {'component_count':len(groups),'primary_minus_controls':{
        a:{k:{'interval95':np.quantile(v,[.025,.975]).tolist(),'valid_resamples':len(v)}
           for k,v in values.items()} for a,values in differences.items()}}


def run():
    plan,qa,full,indices,lookup,checkpoints = validate_ready()
    assert not (OUT/'progress.json').exists(), 'Preserve prior evaluation, no automatic retry'
    state = {'status':'running','pid':os.getpid(),'started_utc':now()}
    dump(OUT/'progress.json',state)
    started = time.perf_counter()
    try:
        results = {}
        for style,result in evaluate(qa,full,indices,checkpoints,90210):
            results[style] = result
            dump(OUT/(style+'.json'),result)
            state.update(completed_style=style)
            dump(OUT/'progress.json',state)
        subprocess.run([sys.executable,__file__,'verify'],check=True)
        baseline = json.loads((BASE/'independent_evaluation.json').read_text())
        qa_endpoints = {s:r['QA142']['records'] for s,r in results.items()}
        qa_endpoints['raw_B_seed42'] = baseline['pilot142_QA']['fracatlas_B_roco_scale_pilot_v3/correct_image']['records']
        clinical_endpoints = {s:r['fracture321']['records'] for s,r in results.items()}
        clinical_endpoints['raw_B_seed42'] = baseline['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']['records']
        qa_boot = bootstrap(qa_endpoints,'clinical',lookup,['overall_acc','open_acc'])
        clinical_boot = bootstrap(clinical_endpoints,'clinical',lookup,
            ['fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity'])
        assert qa_boot['component_count']==32 and clinical_boot['component_count']==314
        report = {'scope':'Predeclared three T LAST296 pilots; inspected development only',
            'evaluation_plan_sha256':sha(OUT/'evaluation_plan.json'),
            'checkpoint_sha256':{s:x['sha256'] for s,x in checkpoints.items()},
            'results':{s:{k:v['metrics'] for k,v in r.items()} for s,r in results.items()},
            'raw_B_reference':{'QA142':baseline['pilot142_QA']['fracatlas_B_roco_scale_pilot_v3/correct_image']['metrics'],
                               'fracture321':baseline['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']['metrics']},
            'QA_component_bootstrap':qa_boot,'clinical_component_bootstrap':clinical_boot,
            'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'prediction_sha256':{s:sha(OUT/(s+'.json')) for s in STYLES},
            'elapsed_seconds_including_replay':time.perf_counter()-started,
            'raw_threshold':.5,'training_updates':0,'calibration_or_test_used':False,
            'long_training_authorized':False,'Final_Solution_selected':False,
            'limitations':['Single seed and repeatedly inspected model-validation, not independent confirmation',
                'Clinical versus irrelevant exact token lengths; paraphrase lengths differ as audited',
                'No best-epoch, best-template, threshold or seed selection; robustness not production selection']}
        dump(OUT/'report.json',report)
        state.update(status='complete_verified_development_only_NOTfull_step3',finished_utc=now(),report_sha256=sha(OUT/'report.json'))
        dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error))
        dump(OUT/'progress.json',state)
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','verify'])
    args = parser.parse_args()
    {'prepare':prepare,'run':run,'verify':replay}[args.mode]()
