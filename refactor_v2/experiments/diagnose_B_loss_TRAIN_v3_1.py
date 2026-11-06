"""Fixed-checkpoint, post-screen TRAIN diagnosis; no fitting or selection.

Both loss arms use all1173 existing TRAIN rows, same prior B diagnosis summaries.
Only public original image/raw question reaches inference. Fresh process replay
is required. This is not another independent-quality experiment.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments import audit_B_fullTRAIN_errors_v3_1 as core

ROOT=Path.cwd();OUT=ROOT/'refactor_v2/reports/B_loss_TRAIN_diagnosis_v3_1'
FOLDER=ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_3'
QUALITY=ROOT/'refactor_v2/reports/B_sequence_loss_quality_v3_2'
BASELINE=ROOT/'refactor_v2/reports/B_fullTRAIN_errors_v3_1'
MODES=('sequence_uniform','sequence_rarity')
sha,dump=core.sha,core.dump


def prepare():
    core.check_environment();assert not OUT.exists()
    quality=json.loads((QUALITY/'independent_artifact_audit.json').read_text())
    assert quality['passed'] and quality['report_sha256']==sha(QUALITY/'report.json')
    baseline=json.loads((BASELINE/'progress.json').read_text())
    assert baseline['status']=='complete_TRAIN_diagnosis_verified'
    assert baseline['report_sha256']==sha(BASELINE/'report.json')
    _,datasets,manifests=core.data_and_schedule()
    assert len(datasets['train'])==1173
    plan=json.loads((FOLDER/'plan.json').read_text());checkpoints={}
    for mode in MODES:
        info=plan['configs'][mode];assert sha(info['path'])==info['sha256']
        cfg=json.loads(Path(info['path']).read_text());target=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
        done=json.loads((FOLDER/(mode+'_completion.json')).read_text())
        assert sha(target)==done['last_sha256']
        checkpoints[mode]={'path':str(target),'sha256':done['last_sha256']}
    dependencies={str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for path in [Path(__file__),Path(core.__file__),FOLDER/'plan.json',QUALITY/'report.json',
        QUALITY/'independent_artifact_audit.json',BASELINE/'report.json',BASELINE/'TRAIN_predictions.json',
        BASELINE/'independent_artifact_audit.json',ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py']:
        dependencies[str(path.resolve())]=sha(path)
    OUT.mkdir()
    for mode in MODES:(OUT/mode).mkdir()
    dump(OUT/'protocol.json',{'scope':'Post-screen all1173TRAIN diagnosis only; no quality/causal claim',
        'checkpoints':checkpoints,'dependencies_sha256':dependencies,'actual_manifests':manifests,
        'modes':list(MODES),'TRAIN_QA':1173,'public_input_keys':['images','questions'],
        'first_constructor_seed':90210,'fresh_process_constructor_seed':90211,
        'summary':'Same original B fullTRAIN arithmetic: perquestion/type/actualroute, globalTRAINanswerfrequency<=5/>5, literal mismatches only',
        'baseline':'Previously verified fixed rawB LAST296 all1173TRAIN predictions, not reinferred',
        'design':'All rows, original order, batch4, original routing, threshold0.5, no loss/routing adapter',
        'hypothesis':'Check whether changed OPEN loss has observable TRAIN learning effects; global rarity stratum is diagnostic, not identical to question-conditioned loss weights',
        'modelval_calibration_test_inference_allowed':False,'training_updates':0,
        'weights_or_labels_or_templates_changed':False,'long_training_authorized':False,
        'exit_rule':'Record both modes regardless outcome; do not tune hyperparameters or claim generalization from TRAIN differences'})
    print('Prepared both-arm all1173TRAIN diagnosis; no model loaded')


def validate():
    core.check_environment();protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependencies_sha256'].items():assert sha(path)==digest,path
    for info in protocol['checkpoints'].values():assert sha(info['path'])==info['sha256']
    _,datasets,manifests=core.data_and_schedule();assert manifests==protocol['actual_manifests']
    import src.models.bonevqa as runtime
    from src.utils import is_closed_question
    assert runtime.is_closed_question is is_closed_question
    return protocol,datasets['train']


def predict(mode,protocol,dataset,seed,track):
    core.OUT=OUT/mode;core.CHECKPOINT=Path(protocol['checkpoints'][mode]['path'])
    core.EXPECTED_LAST=protocol['checkpoints'][mode]['sha256']
    return core.predict(dataset,seed,track)


def replay():
    protocol,dataset=validate()
    for mode in MODES:
        rows=predict(mode,protocol,dataset,90211,False)
        assert rows==json.loads((OUT/mode/'TRAIN_predictions.json').read_text())
    dump(OUT/'fresh_process_replay.json',{'passed':True,'all1173_answers_probabilities_routes_exact_each_mode':True,
        'constructor_seed':90211,'training_updates':0,'reserved_role_inference':False})


def run():
    protocol,dataset=validate();assert not (OUT/'progress.json').exists()
    state={'status':'running_TRAIN_diagnosis','pid':os.getpid(),'completed_modes':[]}
    dump(OUT/'progress.json',state);started=time.perf_counter()
    try:
        summaries={}
        for mode in MODES:
            state['mode']=mode;dump(OUT/'progress.json',state)
            rows=predict(mode,protocol,dataset,90210,True)
            dump(OUT/mode/'TRAIN_predictions.json',rows);summaries[mode]=core.summarize(rows)
            state['completed_modes'].append(mode);dump(OUT/'progress.json',state)
        state['status']='fresh_process_replay_running';dump(OUT/'progress.json',state)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'verify'],check=True)
        baseline=json.loads((BASELINE/'report.json').read_text())['summary']
        # Comparing summaries is descriptive TRAIN error attribution only.
        report={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),
            'summaries':summaries,'raw_B_TRAIN_summary':baseline,
            'prediction_sha256':{m:sha(OUT/m/'TRAIN_predictions.json') for m in MODES},
            'fresh_process_replay_sha256':sha(OUT/'fresh_process_replay.json'),
            'elapsed_seconds_including_replay':time.perf_counter()-started,
            'training_updates':0,'reserved_role_inference':False,'generalization_improvement_claimed':False,
            'long_training_authorized':False,'step3_complete':False}
        dump(OUT/'report.json',report)
        state.update(status='complete_verified_TRAIN_diagnosis_not_quality',report_sha256=sha(OUT/'report.json'))
        dump(OUT/'progress.json',state)
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));dump(OUT/'progress.json',state);raise


def launch():
    validate();assert not (OUT/'detached_launch.json').exists() and not (OUT/'progress.json').exists()
    rows=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    for row in rows:
        fields=row.strip().split(maxsplit=1)
        if len(fields)!=2 or int(fields[0])==os.getpid():continue
        if Path(fields[1].split()[0]).name.lower() in ('python','python3','python3.11'):
            assert ' -m src.train ' not in fields[1]
            assert not ('diagnose_B_loss_TRAIN_v3_1.py run' in fields[1] or ' worker ' in fields[1])
    with (OUT/'detached_driver.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,str(Path(__file__).resolve()),'run'],
            cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    dump(OUT/'detached_launch.json',{'launch_pid':child.pid,'protocol_sha256':sha(OUT/'protocol.json'),
        'start_new_session':True,'training_updates':0,'no_automatic_retry':True})
    print('Detached TRAIN-only diagnostic PID',child.pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','run','verify'])
    args=parser.parse_args();{'prepare':prepare,'launch':launch,'run':run,'verify':replay}[args.action]()
