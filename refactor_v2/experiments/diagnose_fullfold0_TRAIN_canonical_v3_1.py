"""Frozen LAST TRAIN-fit diagnostic; zero optimization or reserved inference."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
PARENT=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
QUALITY=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
OUT=ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'
MODES=('token_mean','sequence_uniform')
QUESTION='Is there a fracture in this X-ray?'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    assert not OUT.exists(),'Preserve previous preparation'
    quality=read(QUALITY/'independent_artifact_audit.json')
    assert quality['passed'] and quality['report_sha256']==sha(QUALITY/'report.json')
    protocol=read(PARENT/'protocol.json')
    cfg=read(protocol['configs'][MODES[0]]['path'])
    path=Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train'])
    rows=read(path)
    indices=[i for i,r in enumerate(rows) if r['question']==QUESTION]
    selected=[rows[i] for i in indices]
    assert len(rows)==11502 and len(selected)==len({r['image_id'] for r in selected})==2568
    assert sum(r['answer']=='yes' for r in selected)==458
    deps={str(Path(__file__).resolve()):sha(__file__),str(path):sha(path)}
    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])
    for p in [PARENT/'runtime_binding.json',PARENT/'protocol.json',QUALITY/'report.json',
              QUALITY/'independent_artifact_audit.json',QUALITY/'checkpoint_preflight.json']:
        deps[str(p)]=sha(p)
    targets={}
    for mode in MODES:
        c=read(protocol['configs'][mode]['path'])
        assert sha(protocol['configs'][mode]['path'])==protocol['configs'][mode]['sha256']
        target=ROOT/c['checkpoint_dir']/c['run_name']/'last.pt'
        digest=read(PARENT/(mode+'_completion.json'))['last_sha256']
        assert sha(target)==digest
        targets[mode]={'path':str(target),'sha256':digest,'config':protocol['configs'][mode]}
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Post-quality frozen TRAIN-fit diagnosis, not efficacy evidence',
        'dependencies_sha256':deps,'targets':targets,'TRAIN_manifest':str(path),
        'selected_indices':indices,'selected_image_ids':[r['image_id'] for r in selected],
        'canonical_question':QUESTION,'TRAIN_images':2568,'positives':458,'negatives':2110,
        'selection':'Every existing canonical TRAIN row in original order, no score/label subsampling',
        'input_keys':['images','questions'],'batch_size':4,'raw_threshold':.5,
        'checkpoint':'Both fixed LAST1438; no selection or weight update',
        'constructor_seeds':[90210,90211],'training_seed':42,
        'fresh_process_all_records_probabilities_exact_required':True,
        'metrics':['AUROC','AUPRC','sensitivity','specificity','BA','Brier','ECE10'],
        'modelval_comparison':'Read verified saved canonical321 summary only, no new modelval inference; descriptive different-cohort TRAIN-fit gap, not causal proof',
        'calibration_test_inference':False,'optimizer_updates':0,'threshold_searches':0,
        'no_automatic_retry':True,'new_quality_training_released':False})
    print('Prepared metadata-only TRAIN2568 diagnostic; zero models/updates')


def validate():
    plan=read(OUT/'protocol.json')
    for path,digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest,path
    for info in plan['targets'].values():
        assert sha(info['path'])==info['sha256']
    assert read(OUT/'independent_preflight.json')['passed']
    assert read(OUT/'independent_preflight.json')['protocol_sha256']==sha(OUT/'protocol.json')
    commands=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    for row in commands.splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if Path(parts[1].split()[0]).name.lower().startswith('python'):
            assert 'run_fullfold0_B_uniform_screen_v3_1.py worker ' not in parts[1]
            assert 'evaluate_fullfold0_fixed_LAST_v3_1.py run' not in parts[1]
    return plan


def evaluate(seed,track):
    import torch
    from src.data import build_dataset,collate_fn
    from src.train import build_split
    from src.protocol import records_of
    from src.models.bonevqa import BoneVQAModel
    from src.utils import compute_metrics,set_seed
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment();plan=validate()
    cfg=read(plan['targets'][MODES[0]]['config']['path'])
    ds=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
    raw=read(plan['TRAIN_manifest']);assert records_of(ds)==raw
    indices=plan['selected_indices']
    assert len(indices)==2568 and [raw[i]['image_id'] for i in indices]==plan['selected_image_ids']
    assert all(raw[i]['question']==QUESTION for i in indices)
    for mode in MODES:
        started=time.perf_counter();set_seed(seed)
        info=plan['targets'][mode]
        model=BoneVQAModel.load(info['path'],device='cpu').eval()
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        before={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
        rows=[]
        for offset in range(0,len(indices),4):
            batch=collate_fn([ds[i] for i in indices[offset:offset+4]])
            with torch.no_grad():
                predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
            assert len(predictions)==len(batch['questions'])
            for j,p in enumerate(predictions):
                assert 'p_yes' in p
                rows.append({'image_id':batch['image_ids'][j],'question':batch['questions'][j],
                    'gt':batch['answers'][j],'answer_type':batch['answer_types'][j],
                    'pred':p['answer'],'p_yes':p['p_yes']})
            if track:dump(OUT/'model_progress.json',{'mode':mode,'completed_images':len(rows),'total_images':2568})
        assert len(rows)==2568 and sha(info['path'])==info['sha256']
        assert before=={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}
        result={'records':rows,'metrics':compute_metrics(rows,fracture_threshold=.5),
            'checkpoint_sha256':info['sha256'],'weights_unchanged':True}
        if track:dump(OUT/(mode+'_timing.json'),{'elapsed_seconds':time.perf_counter()-started})
        del model;gc.collect();yield mode,result


def replay():
    for mode,result in evaluate(90211,False):
        assert result==read(OUT/(mode+'.json')),'Preserve failed replay, no retry'
    dump(OUT/'fresh_process_replay.json',{'passed':True,'all2568_records_probabilities_metrics_exact_each':True,
        'constructor_seed':90211,'optimizer_updates':0,'calibration_test_used':False})


def run():
    validate();assert not (OUT/'progress.json').exists()
    state={'status':'running','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'completed_modes':[]}
    dump(OUT/'progress.json',state);started=time.perf_counter()
    try:
        results={}
        for mode,result in evaluate(90210,True):
            results[mode]=result;dump(OUT/(mode+'.json'),result)
            state['completed_modes'].append(mode);dump(OUT/'progress.json',state)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'replay'],check=True)
        quality=read(QUALITY/'report.json')
        report={'scope':'Frozen TRAIN-fit diagnostic only; not quality gain or cause confirmation',
            'protocol_sha256':sha(OUT/'protocol.json'),
            'TRAIN_metrics':{m:r['metrics'] for m,r in results.items()},
            'saved_modelval_metrics':{m:quality['results'][m]['canonical_metrics'] for m in MODES},
            'prediction_sha256':{m:sha(OUT/(m+'.json')) for m in MODES},
            'replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'elapsed_seconds_including_replay':time.perf_counter()-started,
            'independent_artifact_audit_pending':True,'modelval_new_inference':False,
            'calibration_test_used':False,'optimizer_updates':0,'threshold_searches':0,
            'new_quality_training_released':False,'step3_complete':False}
        dump(OUT/'report.json',report)
        state.update(status='replayed_pending_independent_artifact_audit',report_sha256=sha(OUT/'report.json'))
        dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error))
        dump(OUT/'progress.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','replay'])
    {'prepare':prepare,'run':run,'replay':replay}[parser.parse_args().mode]()
