"""Fixed-checkpoint full model-validation coverage follow-up, not confirmation.

Original142QA screen remains unchanged. No fitting, selection, reserved cohorts,
or training. All1432sealed model_val QA/321images, three fixed LAST296 models.
"""
import argparse
from collections import Counter,defaultdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from experiments import evaluate_T_bounded_current_v3_3 as shared
from src.utils import normalize_answer,normalize_yes_no

core=shared.core
ROOT=Path.cwd();OUT=ROOT/'refactor_v2/reports/full_modelval_coverage_v3_1'
PARENT=ROOT/'refactor_v2/reports/B_sequence_loss_quality_v3_2'
MODES=('raw_B','sequence_uniform','sequence_rarity')
PATHS={
    'raw_B':('fracatlas_B_roco_scale_pilot_v3','94594edf64f8ef3fd7ae974e290b86900e186661d03c09e68d90e34c51ff3ddf'),
    'sequence_uniform':('B_sequence_uniform_seed42_bounded_v3_3','8a4dbbb7abf60ad55e4e0197063743c709a5f135b501c5e5e2c6cecb4328e009'),
    'sequence_rarity':('B_sequence_rarity_seed42_bounded_v3_3','2c818267913ce69e66c30803eef67553aba0f2b77d81aa97b1f994f53b5d9ec1')}
sha,dump=core.sha,core.dump


def checkpoint(mode):return ROOT/'refactor_v2/checkpoints'/PATHS[mode][0]/'last.pt'


def prepare():
    core.check_environment();assert not OUT.exists()
    qa,full,indices,lookup,config=core.cohorts()
    assert len(full)==1432
    parent=json.loads((PARENT/'independent_artifact_audit.json').read_text())
    assert parent['passed'] and parent['report_sha256']==sha(PARENT/'report.json')
    dependencies={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for p in (ROOT/'refactor_v2/experiments').glob('*.py'):dependencies[str(p.resolve())]=sha(p)
    for p in [config,PARENT/'report.json',PARENT/'independent_artifact_audit.json',
              core.BASE/'independent_evaluation.json']:
        dependencies[str(p.resolve())]=sha(p)
    for mode in MODES:assert sha(checkpoint(mode))==PATHS[mode][1]
    # Coverage verification only; GT never becomes public model input.
    full_ids=Counter((r['image_id'],r['question'],r['answer'],r['answer_type']) for r in core.records_of(full))
    assert all(full_ids[(r['image_id'],r['question'],r['answer'],r['answer_type'])]>0 for r in core.records_of(qa))
    OUT.mkdir()
    dump(OUT/'protocol.json',{'scope':'Post-screen full model-validation coverage; not independent confirmation or a new training release',
        'dependencies_sha256':dependencies,'full_modelval_manifest':core.manifest(full),
        'checkpoint_sha256':{m:PATHS[m][1] for m in MODES},'modes':list(MODES),
        'full_QA':1432,'images':321,'canonical_images':321,'components':314,
        'public_input_keys':['images','questions'],'order':'Original sealed full model_val order, batch4',
        'raw_threshold':.5,'calibration_or_test_inference_allowed':False,'training_updates':0,
        'original142_primary_screen_redefined':False,'checkpoint_epoch_seed_template_selection_allowed':False,
        'coverage_summaries':'Per existing GTtype/question and TRAIN normalized-answer frequency0/1to5/>5, after prediction only',
        'bootstrap_resamples':500,'bootstrap_seed':20261020,'contrast_direction':'sequence_rarity minus rawB and sequence_uniform',
        'full_QA_metrics':['overall_acc','open_acc'],'clinical_metrics':['fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity'],
        'constructor_seeds':[90210,90211],'fresh_process_same_full_cohort_exact_replay_required':True,
        'limits':['Repeated development inspection and one training seed, no untouched holdout',
            'Follow-up cannot retrospectively replace the original142primary endpoint',
            'Changing cohort batching can yield small numerical or decoded differences; report prior321score discrepancies instead of selecting results'],
        'long_training_authorized':False})
    print('Prepared fixed3checkpoint full1432modelval coverage, no model loaded')


def validate():
    core.check_environment();protocol=json.loads((OUT/'protocol.json').read_text())
    for p,digest in protocol['dependencies_sha256'].items():assert sha(p)==digest,p
    for mode in MODES:assert sha(checkpoint(mode))==protocol['checkpoint_sha256'][mode]
    _,full,indices,lookup,_=core.cohorts();assert core.manifest(full)==protocol['full_modelval_manifest']
    return protocol,full,indices,lookup


def correct(row):
    normalizer=normalize_yes_no if row['answer_type']=='CLOSED' else normalize_answer
    return normalizer(row['pred'])==normalizer(row['gt'])


def coverage(rows):
    _,datasets,_=core.data_and_schedule();training=core.records_of(datasets['train'])
    counts=Counter(normalize_answer(r['answer']) for r in training)
    questions={r['question'] for r in training};by_question={};by_frequency={}
    for row in rows:
        n=counts[normalize_answer(row['gt'])]
        label='TRAIN_answer_unseen' if n==0 else 'TRAIN_answer_count1to5' if n<=5 else 'TRAIN_answer_count>5'
        for target,key in [(by_question,row['question']),(by_frequency,label)]:
            summary=target.setdefault(key,{'n':0,'correct':0,'OPEN_n':0,'OPEN_correct':0})
            summary['n']+=1;summary['correct']+=int(correct(row))
            if row['answer_type']=='OPEN':summary['OPEN_n']+=1;summary['OPEN_correct']+=int(correct(row))
    return {'per_question':by_question,'TRAIN_answer_frequency_strata':by_frequency,
        'question_template_absent_from_pilot_TRAIN_rows':sum(r['question'] not in questions for r in rows)}


def evaluate(full,indices,seed,track):
    for mode in MODES:
        core.set_seed(seed);model=core.BoneVQAModel.load(str(checkpoint(mode)),device='cpu').eval()
        assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        assert model.cfg['fracture_yes_threshold']==.5 and model.cfg.get('fracture_platt_slope') is None
        before=core.hashes(model.trainable_state_dict());rows=[];started=time.perf_counter()
        for offset in range(0,len(full),4):
            batch=core.collate_fn([full[i] for i in range(offset,min(offset+4,len(full)))])
            with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
            assert len(predictions)==len(batch['questions'])
            rows.extend(shared.repaired.prediction_record(batch,j,p) for j,p in enumerate(predictions))
            if track:dump(OUT/'model_progress.json',{'mode':mode,'completed_QA':len(rows),'total_QA':len(full),'pid':os.getpid()})
        assert core.hashes(model.trainable_state_dict())==before and sha(checkpoint(mode))==PATHS[mode][1]
        clinical=[rows[i] for i in indices];assert len(clinical)==321
        result={'full1432':{'records':rows,'metrics':core.compute_metrics(rows,fracture_threshold=.5)},
            'canonical321':{'records':clinical,'metrics':core.compute_metrics(clinical,fracture_threshold=.5)},
            'coverage':coverage(rows),'weights_unchanged':True}
        if track:dump(OUT/(mode+'_firstpass_timing.json'),{'elapsed_seconds':time.perf_counter()-started,'QA':1432})
        del model;gc.collect();yield mode,result


def replay():
    _,full,indices,_=validate()
    for mode,result in evaluate(full,indices,90211,False):assert result==json.loads((OUT/(mode+'.json')).read_text())
    dump(OUT/'fresh_process_replay.json',{'passed':True,'all1432_records_probabilities_metrics_and_coverage_exact_each_mode':True,
        'constructor_seed':90211,'checkpoint_sha256':{m:PATHS[m][1] for m in MODES},'training_updates':0,'calibration_or_test_used':False})


def run():
    protocol,full,indices,lookup=validate();assert not (OUT/'progress.json').exists()
    state={'status':'running_full_modelval_coverage','pid':os.getpid(),'started_utc':core.now(),'completed_modes':[]}
    dump(OUT/'progress.json',state);started=time.perf_counter()
    try:
        results={}
        for mode,result in evaluate(full,indices,90210,True):
            results[mode]=result;dump(OUT/(mode+'.json'),result);state['completed_modes'].append(mode);dump(OUT/'progress.json',state)
        state['status']='fresh_process_replay_running';dump(OUT/'progress.json',state)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify'],check=True)
        qa_endpoints={m:r['full1432']['records'] for m,r in results.items()}
        clinical_endpoints={m:r['canonical321']['records'] for m,r in results.items()}
        old=json.loads((core.BASE/'independent_evaluation.json').read_text())
        parent={m:json.loads((PARENT/(m+'.json')).read_text())['fracture321']['records'] for m in MODES if m!='raw_B'}
        parent['raw_B']=old['full321_fracture']['fracatlas_B_roco_scale_pilot_v3']['records']
        overlap={}
        for mode in MODES:
            original={r['image_id']:r for r in parent[mode]};now=clinical_endpoints[mode]
            assert set(original)=={r['image_id'] for r in now}
            overlap[mode]={'max_probability_difference_from_prior321_batching':max(abs(r['p_yes']-original[r['image_id']]['p_yes']) for r in now),
                'answer_changes_from_prior321_batching':sum(r['pred']!=original[r['image_id']]['pred'] for r in now)}
        qa_boot=core.bootstrap(qa_endpoints,'sequence_rarity',lookup,protocol['full_QA_metrics'])
        clinical_boot=core.bootstrap(clinical_endpoints,'sequence_rarity',lookup,protocol['clinical_metrics'])
        assert qa_boot['component_count']==clinical_boot['component_count']==314
        report={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),
            'results':{m:{'full1432':r['full1432']['metrics'],'canonical321':r['canonical321']['metrics'],'coverage':r['coverage']} for m,r in results.items()},
            'full_QA_component_bootstrap':qa_boot,'clinical_component_bootstrap':clinical_boot,
            'prior_clinical_batching_overlap':overlap,'prediction_sha256':{m:sha(OUT/(m+'.json')) for m in MODES},
            'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'elapsed_seconds_including_replay':time.perf_counter()-started,
            'original142_primary_screen_redefined':False,'limits':protocol['limits'],
            'training_updates':0,'calibration_or_test_used':False,'long_training_authorized':False,
            'step3_complete':False,'Final_Solution_selected':False,'production_released':False}
        dump(OUT/'report.json',report);state.update(status='complete_verified_development_coverage_not_confirmation',report_sha256=sha(OUT/'report.json'));dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(OUT/'progress.json',state);raise


def launch():
    validate();assert not (OUT/'progress.json').exists() and not (OUT/'detached_launch.json').exists()
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if Path(parts[1].split()[0]).name.lower() in ('python','python3','python3.11'):
            assert ' -m src.train ' not in parts[1] and ' worker ' not in parts[1]
            assert 'evaluate_full_modelval_coverage_v3_1.py run' not in parts[1]
    with (OUT/'detached_driver.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,str(Path(__file__).resolve()),'run'],
            cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(OUT/'detached_launch.json',{'launch_pid':child.pid,'protocol_sha256':sha(OUT/'protocol.json'),'start_new_session':True,'training_updates':0,'no_automatic_retry':True})
    print('Detached full modelval coverage PID',child.pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify','launch']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'verify':replay,'launch':launch}[args.action]()
