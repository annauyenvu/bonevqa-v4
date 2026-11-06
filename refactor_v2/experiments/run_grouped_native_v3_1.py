"""Serial four-fold execution, strict frozen configs and previously checked caps."""
import sys,os,json,hashlib,subprocess,time
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA')
CON=ROOT/'refactor_v2/reports/grouped_native_executor_contract_v3_1'
OUT=ROOT/'refactor_v2/reports/grouped_native_training_v3_1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def prepare():
    assert not OUT.exists()
    gates=[CON/'report.json',CON/'model_integration.json',ROOT/'refactor_v2/reports/grouped_native_preparation_v3_2/independent_audit.json',ROOT/'refactor_v2/reports/grouped_native_export_gate_v3_1/report.json']
    assert read(gates[0])['passed_cap_fixtures']
    assert all(read(p)['passed'] for p in gates[1:])
    bindings=read(CON/'bindings.json');paths=gates+[CON/'bindings.json',Path(__file__).resolve(),ROOT/'refactor_v2/reports/GROUPED_EVALUATION_RULES_V3_1.json']
    paths+=list((ROOT/'refactor_v2/src').rglob('*.py'))
    for name in ('grouped_native_executor_contract_v3_1.py','private_TRAIN_sequence_loss_adapter_v3_1.py','sequence_answer_loss_candidate_v3_1.py'):
        paths.append(ROOT/'refactor_v2/experiments'/name)
    for b in bindings.values():
        assert sha(b['config'])==b['config_sha256'] and sha(b['initial_path'])==b['initial_sha256']
        assert not Path(b['checkpoint_dir']).exists()
        paths += [Path(b['config']),Path(b['initial_path'])]
        c=read(b['config'])
        paths += [Path(p) for p in c['data']['loader_kwargs']['sealed_records_by_split'].values() if p]
        paths += [Path(c['data']['sealed_answer_vocab'])]
    OUT.mkdir();dump(OUT/'release.json',{'dependencies':{str(p):sha(p) for p in paths},'folds':['1','2','3','4'],'bindings':bindings,'seed':42,'epochs':2,'total_updates':5768,'seconds_cap_per_fold':10800,'calibration_test_allowed':False,'no_automatic_retry':True,'scope':'Only these four native TRAIN/modelval runs; fixed LAST, no candidate search'})
def validate():
    p=read(OUT/'release.json')
    for file,h in p['dependencies'].items():assert sha(file)==h,file
    return p
def worker(f):
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    from experiments.grouped_native_executor_contract_v3_1 import integration
    from src.utils import normalize_answer
    import src.train as trainer
    check_environment();p=validate();assert f in p['folds']
    b=dict(p['bindings'][f]);assert not Path(b['checkpoint_dir']).exists()
    c=read(b['config']);rows=read(c['data']['loader_kwargs']['sealed_records_by_split']['train'])
    table={(r['question'],normalize_answer(r['answer'])):1. for r in rows if r['answer_type']=='OPEN'}
    b['checkpoint_writes_released']=True;b['release_sha256']=sha(OUT/'release.json')
    stats={'TRAIN_calls':0,'TRAIN_rows':0,'native_eval_rows':0}
    original=trainer.load_config
    def load(path):
        assert Path(path).resolve()==Path(b['config']).resolve() and sha(path)==b['config_sha256']
        loaded=original(path);assert loaded==c;return loaded
    trainer.load_config=load;sys.argv=['src.train','--config',b['config'],'--device','cpu'];start=time.perf_counter()
    with integration(table,stats,b):trainer.main()
    assert stats=={'TRAIN_calls':b['max_TRAIN_calls'],'TRAIN_rows':b['max_TRAIN_rows'],'native_eval_rows':b['max_eval_rows']}
    folder=Path(b['checkpoint_dir']);history=read(folder/'history.json');assert len(history)==2 and history[-1]['epoch']==2
    dump(OUT/f'fold_{f}_completion.json',{'passed_execution':True,'elapsed_seconds':time.perf_counter()-start,'statistics':stats,'last_sha256':sha(folder/'last.pt'),'optimizer_sha256':sha(folder/'optimizer.pt'),'independent_quality_audit_pending':True})
def run():
    p=validate();assert not (OUT/'progress.json').exists()
    commands=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    for line in commands.splitlines():
        parts=line.strip().split(maxsplit=1)
        if len(parts)==2 and int(parts[0])!=os.getpid() and Path(parts[1].split()[0]).name.startswith('python'):
            assert ' -m src.train ' not in parts[1] and 'run_grouped_native_v3_1.py' not in parts[1]
    os.close(os.open(OUT/'ACTIVE.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
    state={'status':'running','pid':os.getpid(),'completed':[]};dump(OUT/'progress.json',state)
    try:
        for f in p['folds']:
            state['active_fold']=f;dump(OUT/'progress.json',state)
            with (OUT/f'fold_{f}.log').open('x') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'worker',f],stdout=log,stderr=subprocess.STDOUT,check=True)
            state['completed'].append(f);dump(OUT/'progress.json',state)
        state['status']='trained_pending_independent_evaluation';dump(OUT/'progress.json',state)
    except Exception as e:
        state.update(status='failed_preserved_no_retry',error=repr(e));dump(OUT/'progress.json',state);raise
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2])
    else:{'prepare':prepare,'run':run}[sys.argv[1]]()
