"""Separate metadata/source and, after training, checkpoint-only preflight.

No neural model construction, prediction or optimizer updates. Do not load
checkpoint tensors while the quality worker is active. Preserve audit results.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path('/Users/uyen/Downloads/BONEVQA')
PARENT = ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
OUT = ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
MODES = ('token_mean', 'sequence_uniform')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def stable(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                    separators=(',',':')).encode()).hexdigest()


def defaults():
    tree = ast.parse((ROOT/'refactor_v2/src/models/bonevqa.py').read_text())
    nodes = [n for n in tree.body if isinstance(n,ast.AnnAssign)
             and isinstance(n.target,ast.Name) and n.target.id=='DEFAULT_MODEL_CFG']
    assert len(nodes)==1
    return ast.literal_eval(nodes[0].value)


def audit(checkpoints=False):
    destination = OUT/('checkpoint_preflight.json' if checkpoints else 'staged_source_preflight.json')
    assert not destination.exists(), 'Preserve prior audit; no overwrite'
    plan = read(OUT/'evaluation_plan.json')
    protocol = read(PARENT/'protocol.json')
    assert sha(PARENT/'protocol.json')==plan['screen_protocol_sha256']==(
        'b7a02dae0455abc5535242981364125d87f3f5681e1019ff794e28031e89ab4f')
    for path,digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest,path
    binding = read(PARENT/'runtime_binding.json')
    for path,digest in binding['dependency_sha256'].items():
        assert sha(path)==digest,path
    assert plan['modes']==list(MODES) and plan['bootstrap_seed']==20261021
    assert plan['bootstrap_resamples']==500 and plan['checkpoint']=='LAST1438_epoch2_only'
    assert plan['public_input_keys']==['images','questions'] and plan['raw_threshold']==.5
    assert plan['constructor_seeds']==[90210,90211] and plan['training_seed']==42
    assert not plan['calibration_test_allowed'] and not plan['automatic_launch']
    source = ROOT/'refactor_v2/experiments/evaluate_fullfold0_fixed_LAST_v3_1.py'
    ast.parse(source.read_text())
    configs = {mode:read(protocol['configs'][mode]['path']) for mode in MODES}
    for mode,cfg in configs.items():
        assert sha(protocol['configs'][mode]['path'])==protocol['configs'][mode]['sha256']
        assert cfg['train']['epochs']==2 and cfg['seed']==42
        assert cfg['data']['loader_kwargs']['sealed_records_by_split']['calibration'] is None
        assert cfg['data']['loader_kwargs']['sealed_records_by_split']['test'] is None
    cfg = configs[MODES[0]]
    paths = cfg['data']['loader_kwargs']['sealed_records_by_split']
    training,val = read(paths['train']),read(paths['model_val'])
    assert len(training)==11502 and len(val)==1432
    assert sum(r['answer_type']=='OPEN' for r in val)==628
    canonical = [r for r in val if r['question']=='Is there a fracture in this X-ray?']
    assert len(canonical)==len({r['image_id'] for r in canonical})==321
    assert sum(r['answer'].lower()=='yes' for r in canonical)==57
    batching = cfg['train']['component_batching']
    manifest_path = Path(batching['manifest'])
    seal = read(batching['seal'])
    assert seal['artifacts'][manifest_path.relative_to(ROOT).as_posix()]==sha(manifest_path)
    groups = read(manifest_path)['components']
    lookup = {}
    for group,images in enumerate(groups):
        for image in images:
            assert image not in lookup
            lookup[image]=group
    tg={lookup[r['image_id']] for r in training}
    vg={lookup[r['image_id']] for r in val}
    assert len(vg)==314 and not tg&vg
    checked = {}
    if checkpoints:
        state = read(PARENT/'progress.json')
        assert state['status']=='both_trained_pending_independent_quality_evaluation'
        assert state['completed_modes']==list(MODES)
        commands = subprocess.check_output(['ps','-axo','command='],text=True)
        assert not any('run_fullfold0_B_uniform_screen_v3_1.py worker ' in row
                       or ' -m src.train ' in row for row in commands.splitlines())
        import torch
        torch.set_num_threads(8)
        def finite(value):
            if torch.is_tensor(value):
                return bool(torch.isfinite(value).all())
            if isinstance(value,dict):
                return all(finite(v) for v in value.values())
            if isinstance(value,(tuple,list)):
                return all(finite(v) for v in value)
            if isinstance(value,float):
                return math.isfinite(value)
            return True
        vocab = read(cfg['data']['sealed_answer_vocab'])
        assert isinstance(vocab,list) and len(vocab)==125
        counts = {a:sum(r['question']=='Is there a fracture in this X-ray?'
            and r['answer_type']=='CLOSED' and r['answer'].lower()==a for r in training)
                  for a in ('yes','no')}
        assert counts=={'yes':458,'no':2110}
        for mode,c in configs.items():
            target=ROOT/c['checkpoint_dir']/c['run_name']
            done=read(PARENT/(mode+'_completion.json'))
            assert done['complete'] and done['primary_is_LAST1438']
            assert sha(target/'last.pt')==done['last_sha256']
            assert sha(target/'optimizer.pt')==done['optimizer_sha256']
            payload=torch.load(target/'last.pt',map_location='cpu',weights_only=False)
            optimizer=torch.load(target/'optimizer.pt',map_location='cpu',weights_only=False)
            expected=defaults()
            updates=copy.deepcopy(c['model'])
            updates['fracture_positive_weight']=counts['no']/counts['yes']
            for k,v in updates.items():
                if isinstance(v,dict) and isinstance(expected.get(k),dict):
                    expected[k].update(v)
                else:
                    expected[k]=v
            assert payload['config']==expected
            assert payload['answer_vocab']==vocab
            assert len(payload['state'])==263 and finite(payload['state']) and finite(optimizer)
            assert payload['extra']['global_step']==optimizer['global_step']==1438
            assert payload['extra']['epoch']==2
            meta=payload['manifest']
            assert meta['schema_version']==2 and meta['architecture_hash']==stable(expected)
            assert meta['answer_vocab_hash']==stable(vocab)
            assert meta['preprocessing_version']=='bonevqa-raster-letterbox-v2'
            assert meta['tensor_shapes']=={k:list(v.shape) for k,v in payload['state'].items()}
            extra=payload['extra']['fullfold0_uniform_screen']
            assert extra['mode']==mode and extra['screen_protocol_sha256']==sha(PARENT/'protocol.json')
            assert extra['config_sha256']==protocol['configs'][mode]['sha256']
            assert extra['initial_sha256']==protocol['configs'][mode]['initial_sha256']
            assert extra['runtime_binding_sha256']==sha(PARENT/'runtime_binding.json')
            assert extra['scoped_release_sha256']==sha(PARENT/'scoped_execution_release.json')
            history=read(target/'history.json')
            assert len(history)==2 and all(h['val_n']==1432 and h['skipped_no_supervision_batches']==0 for h in history)
            worker=read(PARENT/(mode+'_worker_audit.json'))
            assert worker['passed'] and worker['statistics']['TRAIN_rows']==23004
            assert worker['statistics']['native_eval_rows']==2864
            checked[mode]={'LAST_sha256':done['last_sha256'],'optimizer_sha256':done['optimizer_sha256'],
                           'finite_state_and_exact_resolved_config':True,'global_step':1438}
            del payload,optimizer
    report={'passed':True,'scope':'checkpoint preflight' if checkpoints else 'metadata/source preparation only',
        'auditor_sha256':sha(__file__),'plan_sha256':sha(OUT/'evaluation_plan.json'),
        'evaluator_sha256':sha(source),'plan_dependencies_checked':len(plan['dependencies_sha256']),
        'runtime_dependencies_checked':len(binding['dependency_sha256']),
        'modelval_QA':1432,'OPEN_QA':628,'canonical_images':321,'known_components':314,
        'TRAIN_modelval_component_overlap':0,'checkpoint_checks':checked,
        'neural_model_instantiations':0,'model_predictions':0,'optimizer_updates':0,
        'neural_replay_complete':False,'quality_report_audit_complete':False}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['staged','checkpoints'])
    audit(parser.parse_args().mode=='checkpoints')
