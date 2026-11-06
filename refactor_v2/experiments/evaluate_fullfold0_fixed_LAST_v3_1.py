"""Current two-arm LAST1438 evaluation. Prepare is metadata-only; no auto-launch.

Never reuse historical LAST296 checkpoints, pilot frequency tables or bootstrap
seed. Failures are preserved, not retried. Independent artifact audit remains
required after fresh-process prediction replay.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
PARENT = ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
OUT = ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
MODES = ('token_mean', 'sequence_uniform')
PROTOCOL_SHA = 'b7a02dae0455abc5535242981364125d87f3f5681e1019ff794e28031e89ab4f'
CANONICAL = 'Is there a fracture in this X-ray?'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def bindings():
    assert sha(PARENT/'protocol.json') == PROTOCOL_SHA
    protocol = read(PARENT/'protocol.json')
    deps = {**protocol['dependency_sha256'],
            **read(PARENT/'runtime_binding.json')['dependency_sha256']}
    for name, digest in deps.items():
        assert sha(name) == digest, name
    configs = {}
    for mode in MODES:
        info = protocol['configs'][mode]
        assert sha(info['path']) == info['sha256']
        configs[mode] = read(info['path'])
    return protocol, configs


def prepare():
    _, configs = bindings()
    assert not OUT.exists(), 'Preserve prior preparation, no overwrite'
    cfg = configs['token_mean']
    paths = cfg['data']['loader_kwargs']['sealed_records_by_split']
    assert paths['calibration'] is None and paths['test'] is None
    rows = read(paths['model_val'])
    assert len(rows) == 1432 and sum(r['answer_type']=='OPEN' for r in rows) == 628
    selected = [r for r in rows if r['question']==CANONICAL]
    assert len(selected) == len({r['image_id'] for r in selected}) == 321
    assert sum(r['answer'].lower()=='yes' for r in selected) == 57
    deps = {str(Path(__file__).resolve()): sha(__file__)}
    for name in ['protocol.json', 'runtime_binding.json', 'scoped_execution_release.json']:
        deps[str(PARENT/name)] = sha(PARENT/name)
    for path in (ROOT/'refactor_v2/src').rglob('*.py'):
        deps[str(path)] = sha(path)
    for path in [paths['train'], paths['model_val'], cfg['data']['sealed_answer_vocab'],
                 cfg['train']['component_batching']['manifest'],
                 cfg['train']['component_batching']['seal']]:
        deps[str(path)] = sha(path)
    OUT.mkdir()
    dump(OUT/'evaluation_plan.json', {
        'screen_protocol_sha256': PROTOCOL_SHA, 'dependencies_sha256': deps,
        'modes': list(MODES), 'checkpoint': 'LAST1438_epoch2_only',
        'full_QA': 1432, 'OPEN_QA': 628, 'canonical_images': 321,
        'known_components': 314, 'order': 'sealed original modelval order batch4',
        'public_input_keys': ['images', 'questions'], 'raw_threshold': .5,
        'primary_contrast': 'sequence_uniform minus token_mean OPEN exact',
        'bootstrap_resamples': 500, 'bootstrap_seed': 20261021,
        'constructor_seeds': [90210, 90211], 'training_seed': 42,
        'calibration_test_allowed': False, 'training_updates': 0,
        'independent_confirmation': False, 'automatic_launch': False,
        'fresh_process_exact_replay_required': True,
        'independent_artifact_audit_required': True})
    print('Prepared metadata-only evaluation plan; zero model instantiations')


def ready():
    plan = read(OUT/'evaluation_plan.json')
    for path, digest in plan['dependencies_sha256'].items():
        assert sha(path) == digest, path
    _, configs = bindings()
    state = read(PARENT/'progress.json')
    assert state['status'] == 'both_trained_pending_independent_quality_evaluation'
    assert state['completed_modes'] == list(MODES)
    commands = subprocess.check_output(['ps', '-axo', 'command='], text=True)
    assert not any('run_fullfold0_B_uniform_screen_v3_1.py worker ' in line
                   or ' -m src.train ' in line for line in commands.splitlines())
    targets = {}
    for mode in MODES:
        cfg = configs[mode]
        target = ROOT/cfg['checkpoint_dir']/cfg['run_name']
        done = read(PARENT/(mode+'_completion.json'))
        assert done['complete'] and done['primary_is_LAST1438']
        assert sha(target/'last.pt') == done['last_sha256']
        assert sha(target/'optimizer.pt') == done['optimizer_sha256']
        history = read(target/'history.json')
        assert len(history)==2 and history[-1]['epoch']==2
        assert all(h['val_n']==1432 and h['skipped_no_supervision_batches']==0 for h in history)
        audit = read(PARENT/(mode+'_worker_audit.json'))
        assert audit['passed'] and audit['max_updates']==1438
        targets[mode] = (target/'last.pt', done['last_sha256'], history[-1])
    return plan, configs, targets


def evaluate(seed, track):
    import torch
    from src.data import build_dataset, collate_fn
    from src.train import build_split
    from src.protocol import records_of
    from src.models.bonevqa import BoneVQAModel
    from src.utils import compute_metrics, set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment()
    _, configs, targets = ready()
    cfg = configs['token_mean']
    ds = build_split(build_dataset, cfg['data']['dataset'], 'model_val', None, True,
                     cfg['data']['loader_kwargs'])
    expected = records_of(ds)
    assert len(expected)==1432
    for mode in MODES:
        path, digest, history = targets[mode]
        started = time.perf_counter()
        set_seed(seed)
        model = BoneVQAModel.load(str(path), device='cpu').eval()
        assert len(model.answer_vocab)==125
        assert not any(model.cfg[k] for k in ('use_visual_prompt', 'use_lens',
            'use_latent_prompt', 'use_textual_prompt', 'use_region_in_prompt'))
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        before = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                  for k,v in model.trainable_state_dict().items()}
        rows = []
        for offset in range(0, len(ds), 4):
            batch = collate_fn([ds[i] for i in range(offset, min(offset+4, len(ds)))])
            with torch.no_grad():
                predictions = model.predict_batch({'images': batch['images'], 'questions': batch['questions']})
            assert len(predictions)==len(batch['questions'])
            for j,prediction in enumerate(predictions):
                row = {'image_id': batch['image_ids'][j], 'question': batch['questions'][j],
                       'gt': batch['answers'][j], 'answer_type': batch['answer_types'][j],
                       'pred': prediction['answer']}
                if 'p_yes' in prediction:
                    row['p_yes'] = prediction['p_yes']
                if row['question']==CANONICAL:
                    assert 'p_yes' in row
                rows.append(row)
            if track:
                dump(OUT/'model_progress.json', {'mode': mode, 'completed_QA': len(rows), 'total_QA': 1432})
        assert [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows] == [
            (r['image_id'],r['question'],r['answer'],r['answer_type']) for r in expected]
        assert before == {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                          for k,v in model.trainable_state_dict().items()}
        clinical = [r for r in rows if r['question']==CANONICAL]
        assert len(clinical)==321 and sha(path)==digest
        metrics = compute_metrics(rows, fracture_threshold=.5)
        discrepancy = {k: metrics[k]-v for k,v in ((k[4:],v) for k,v in history.items()
            if k.startswith('val_') and k!='val_loss')}
        result = {'records': rows, 'metrics': metrics,
                  'canonical_metrics': compute_metrics(clinical, fracture_threshold=.5),
                  'trainer_metric_differences': discrepancy, 'weights_unchanged': True,
                  'checkpoint_sha256': digest}
        if track:
            dump(OUT/(mode+'_timing.json'), {'elapsed_seconds': time.perf_counter()-started})
        del model
        gc.collect()
        yield mode, result


def bootstrap(results, cfg):
    import numpy as np
    from src.context_controls import load_component_lookup
    from src.utils import compute_metrics
    b = cfg['train']['component_batching']
    lookup, _ = load_component_lookup(b['manifest'], b['seal'], ROOT)
    arms = {m: r['records'] for m,r in results.items()}
    identities = lambda rows: [(r['image_id'],r['question'],r['gt'],r['answer_type']) for r in rows]
    assert identities(arms[MODES[0]])==identities(arms[MODES[1]])
    groups = sorted({lookup[r['image_id']] for r in arms[MODES[0]]})
    assert len(groups)==314
    positions = {g: [i for i,r in enumerate(arms[MODES[0]]) if lookup[r['image_id']]==g] for g in groups}
    qa_keys = ['open_acc','overall_acc','closed_acc']
    clinical_keys = ['fracture_auroc','fracture_auprc','fracture_sensitivity',
        'fracture_specificity','fracture_balanced_accuracy','fracture_brier','fracture_ece10']
    draws = {k: [] for k in qa_keys+clinical_keys}
    rng = np.random.default_rng(20261021)
    for _ in range(500):
        indices = [i for g in rng.choice(groups,len(groups),replace=True) for i in positions[g]]
        metrics = {}
        for mode,rows in arms.items():
            sample = [rows[i] for i in indices]
            metrics[mode] = compute_metrics(sample, fracture_threshold=.5)
        for key in draws:
            if all(key in m for m in metrics.values()):
                value = metrics[MODES[1]][key]-metrics[MODES[0]][key]
                if np.isfinite(value):
                    draws[key].append(value)
    return {'seed':20261021, 'resamples':500, 'known_components':314,
        'direction':'sequence_uniform minus token_mean; Brier/ECE lower is better',
        'paired_intervals':{k:{'interval95':np.quantile(v,[.025,.975]).tolist(),
            'valid_resamples':len(v)} for k,v in draws.items()}}


def replay():
    for mode,result in evaluate(90211, False):
        assert result==read(OUT/(mode+'.json')), 'Fresh-process exact replay failed; preserve artifacts'
    dump(OUT/'fresh_process_replay.json', {'passed':True, 'all1432records_probabilities_metrics_exact':True,
        'constructor_seed':90211, 'training_updates':0, 'calibration_test_used':False})


def run():
    _,configs,_ = ready()
    assert not (OUT/'progress.json').exists(), 'Preserve failure, no automatic retry'
    state = {'status':'running', 'pid':os.getpid(), 'completed_modes':[]}
    dump(OUT/'progress.json',state)
    started = time.perf_counter()
    try:
        results = {}
        for mode,result in evaluate(90210, True):
            dump(OUT/(mode+'.json'),result)
            results[mode]=result
            state['completed_modes'].append(mode)
            dump(OUT/'progress.json',state)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        uncertainty = bootstrap(results, configs['token_mean'])
        report = {'scope':'Fixed LAST1438 development-informed pair, not clinical confirmation',
            'plan_sha256':sha(OUT/'evaluation_plan.json'),
            'results':{m:{k:v for k,v in r.items() if k!='records'} for m,r in results.items()},
            'prediction_sha256':{m:sha(OUT/(m+'.json')) for m in MODES},
            'replay_sha256':sha(OUT/'fresh_process_replay.json'), 'paired_bootstrap':uncertainty,
            'elapsed_seconds_including_replay_statistics':time.perf_counter()-started,
            'independent_artifact_audit_pending':True, 'training_seed':42,
            'calibration_test_used':False, 'step3_complete':False, 'production_released':False}
        dump(OUT/'report.json',report)
        state.update(status='replayed_pending_independent_artifact_audit',report_sha256=sha(OUT/'report.json'))
        dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error))
        dump(OUT/'progress.json',state)
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','replay'])
    action = parser.parse_args().mode
    {'prepare':prepare,'run':run,'replay':replay}[action]()
