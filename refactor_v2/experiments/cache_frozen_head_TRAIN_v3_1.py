"""TRAIN-only frozen x_f cache, exact prior public-score and fresh-feature replay.

No optimizer, validation inference, or training release. Failures are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
OUT = ROOT/'refactor_v2/reports/frozen_head_TRAIN_cache_v3_1'
PRIOR = ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists(), 'Preserve previous artifacts'
    prior = read(PRIOR/'protocol.json')
    assert read(PRIOR/'independent_artifact_audit.json')['passed']
    deps = dict(prior['dependencies_sha256'])
    for p in [Path(__file__).resolve(), PRIOR/'protocol.json', PRIOR/'token_mean.json',
              PRIOR/'report.json', PRIOR/'independent_artifact_audit.json',
              ROOT/'refactor_v2/reports/FROZEN_FEATURE_HEAD_PILOT_DRAFT_V3.json']:
        deps[str(p)] = sha(p)
    target = prior['targets']['token_mean']
    deps[target['config']['path']] = target['config']['sha256']
    for p, digest in deps.items():
        assert sha(p) == digest, p
    OUT.mkdir()
    dump(OUT/'protocol.json', {
        'scope': 'TRAIN-only cache qualification, not head optimization release',
        'dependencies_sha256': deps, 'target': target,
        'TRAIN_manifest': prior['TRAIN_manifest'],
        'indices': prior['selected_indices'], 'image_ids': prior['selected_image_ids'],
        'public_input_keys': ['images', 'questions'], 'batch_size': 4,
        'runtime': 'Original CPU float32, eight intra-op threads; eval/no_grad',
        'cache': 'Exact fracture_head input x_f; canonical TRAIN existing labels only',
        'required': 'All2568 prior public scores/decisions exact; full fresh-process feature tensor replay exact; weights unchanged',
        'constructor_seeds': [90210, 90211], 'raw_threshold': .5,
        'optimizer_updates': 0, 'modelval_inference': False,
        'calibration_test_inference': False, 'quality_training_released': False,
        'no_automatic_retry': True})
    print('Prepared TRAIN2568-only frozen-feature extraction, no optimizer release', flush=True)


def validate():
    p = read(OUT/'protocol.json')
    for path, digest in p['dependencies_sha256'].items():
        assert sha(path) == digest, path
    assert sha(p['target']['path']) == p['target']['sha256']
    audit = read(OUT/'preflight.json')
    assert audit['passed'] and audit['protocol_sha256'] == sha(OUT/'protocol.json')
    return p


def extract(seed, progress):
    import torch
    from src.data import build_dataset, collate_fn
    from src.train import build_split
    from src.protocol import records_of
    from src.models.bonevqa import BoneVQAModel
    from src.utils import set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment()
    p = validate()
    cfg = read(p['target']['config']['path'])
    ds = build_split(build_dataset, cfg['data']['dataset'], 'train', None, True, cfg['data']['loader_kwargs'])
    raw = read(p['TRAIN_manifest'])
    assert records_of(ds) == raw
    indices = p['indices']
    assert len(indices) == 2568 and [raw[i]['image_id'] for i in indices] == p['image_ids']
    assert sum(raw[i]['answer'] == 'yes' for i in indices) == 458
    expected = read(PRIOR/'token_mean.json')['records']
    set_seed(seed)
    model = BoneVQAModel.load(p['target']['path'], device='cpu').eval()
    assert not any(model.cfg[k] for k in ('use_visual_prompt', 'use_lens', 'use_latent_prompt', 'use_textual_prompt', 'use_region_in_prompt'))
    assert model.cfg['fracture_yes_threshold'] == .5 and model.cfg.get('fracture_platt_slope') is None
    def hashes():
        return {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in model.trainable_state_dict().items()}
    before = hashes()
    assert len(before) == 263
    features, scores, labels, pixels = [], [], [], []
    capture = []
    handle = model.fracture_head.register_forward_pre_hook(lambda module, args: capture.append(args[0].detach().cpu().clone()))
    for offset in range(0, len(indices), 4):
        batch = collate_fn([ds[i] for i in indices[offset:offset+4]])
        assert batch['image_ids'] == p['image_ids'][offset:offset+4]
        assert all(q == 'Is there a fracture in this X-ray?' for q in batch['questions'])
        capture.clear()
        with torch.no_grad():
            predictions = model.predict_batch({'images': batch['images'], 'questions': batch['questions']})
        assert len(capture) == 1 and capture[0].shape[0] == len(predictions)
        assert torch.isfinite(capture[0]).all()
        features.append(capture[0])
        for j, pred in enumerate(predictions):
            old = expected[offset+j]
            assert old['image_id'] == batch['image_ids'][j] and old['gt'] == batch['answers'][j]
            assert pred['p_yes'] == old['p_yes'] and pred['answer'] == old['pred'], (offset+j, pred, old)
            scores.append(pred['p_yes'])
            labels.append(int(batch['answers'][j] == 'yes'))
            im = batch['images'][j]
            pixels.append(hashlib.sha256(im.mode.encode()+str(im.size).encode()+im.tobytes()).hexdigest())
        if progress and (offset % 128 == 0 or offset+4 >= len(indices)):
            dump(OUT/'model_progress.json', {'completed': offset+len(predictions), 'total': 2568})
            print(f'Cached {offset+len(predictions)}/2568 TRAIN images', flush=True)
    handle.remove()
    assert hashes() == before and sha(p['target']['path']) == p['target']['sha256']
    return {'features': torch.cat(features), 'labels': torch.tensor(labels, dtype=torch.float32),
            'scores': scores, 'image_ids': p['image_ids'], 'indices': indices,
            'decoded_image_sha256': pixels, 'head_state': {k: v.detach().cpu().clone() for k, v in model.fracture_head.state_dict().items()},
            'head_dropout': model.cfg['dropout'], 'all263_trainable_hashes': before,
            'checkpoint_sha256': p['target']['sha256'], 'protocol_sha256': sha(OUT/'protocol.json'),
            'role': 'TRAIN_ONLY', 'optimizer_updates': 0}


def replay():
    import torch
    old = torch.load(OUT/'TRAIN_features.pt', map_location='cpu')
    fresh = extract(90211, False)
    assert old.keys() == fresh.keys()
    for key in old:
        if torch.is_tensor(old[key]):
            assert torch.equal(old[key], fresh[key]), key
        elif key == 'head_state':
            assert old[key].keys() == fresh[key].keys()
            assert all(torch.equal(old[key][k], fresh[key][k]) for k in old[key])
        else:
            assert old[key] == fresh[key], key
    dump(OUT/'fresh_process_replay.json', {'passed': True, 'all2568_features_scores_labels_pixels_head_and_weight_hashes_exact': True,
        'cache_sha256': sha(OUT/'TRAIN_features.pt'), 'optimizer_updates': 0})


def run():
    import torch
    validate()
    assert not (OUT/'progress.json').exists()
    state = {'status': 'running', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}
    dump(OUT/'progress.json', state)
    started = time.perf_counter()
    try:
        data = extract(90210, True)
        torch.save(data, OUT/'TRAIN_features.pt')
        dump(OUT/'extraction_timing.json', {'seconds': time.perf_counter()-started, 'shape': list(data['features'].shape)})
        del data
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'replay'], check=True)
        dump(OUT/'report.json', {'protocol_sha256': sha(OUT/'protocol.json'), 'cache_sha256': sha(OUT/'TRAIN_features.pt'),
            'replay_sha256': sha(OUT/'fresh_process_replay.json'), 'elapsed_seconds_including_replay': time.perf_counter()-started,
            'all2568_original_public_scores_exact': True, 'optimizer_updates': 0, 'modelval_calibration_test_inference': False,
            'quality_training_released': False, 'independent_artifact_audit_pending': True})
        state.update(status='replayed_pending_artifact_audit', report_sha256=sha(OUT/'report.json'))
        dump(OUT/'progress.json', state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry', failure=repr(error))
        dump(OUT/'progress.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run', 'replay'])
    {'prepare': prepare, 'run': run, 'replay': replay}[parser.parse_args().mode]()
