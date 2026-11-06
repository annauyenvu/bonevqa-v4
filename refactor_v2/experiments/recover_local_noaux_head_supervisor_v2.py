"""Replace only a verified idle supervisor; never touch an active training worker."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
def read(p):return json.loads(p.read_text())
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
assert not (O/'execution_completed.json').exists()
assert not (O/'postprocess/progress.json').exists(),'A head child already started; no idle replacement allowed'
assert not any(O.glob('fold_*/*/head_refit'))
dest=O/'head_batch_consistency_v2';dest.mkdir(exist_ok=False)
for name in ('preparation.json','head_analysis_release.json'):
    for p,h in read(O/name)['dependencies'].items():assert digest(Path(p))==h,p
old=read(O/'postprocess/started.json')['pid']
command=subprocess.check_output(['ps','-p',str(old),'-o','command='],text=True).strip()
assert 'local_noaux_head_analysis_v1.py supervise' in command,command
assert old not in (read(O/'coordinator.json')['pid'],read(O/'active_child.json')['pid'])
paths=[ROOT/'refactor_v2/experiments/local_noaux_head_analysis_v1.py',ROOT/'refactor_v2/experiments/local_noaux_head_analysis_v2.py',ROOT/'refactor_v2/reports/LOCAL_NOAUX_HEAD_BATCH_REVIEW_V2_20261028.md']
dump(dest/'release.json',{'dependencies':{str(p):digest(p) for p in paths},'synthetic_diagnostic':{'seed':42,'rows':803,'features':768,'compute':'FP32CPU1','changed_probabilities':109,'maximum_absolute_difference':1.1920928955078125e-7},'only_change':'detached_head_prediction_batch4_instead_of_full_cohort','head_training_started_before_change':False,'scientific_budget_unchanged':True,'wall':time.time()})
os.kill(old,signal.SIGTERM)
for _ in range(30):
    alive=subprocess.run(['ps','-p',str(old),'-o','command='],capture_output=True,text=True)
    if alive.returncode:break
    time.sleep(1)
else:raise RuntimeError('Old idle supervisor did not terminate; no duplicate started')
dump(dest/'idle_supervisor_stopped.json',{'pid':old,'verified_command':command,'no_child_was_started':True,'wall':time.time()})
log=(dest/'supervisor.log').open('x')
new=subprocess.Popen([sys.executable,'-u',str(ROOT/'refactor_v2/experiments/local_noaux_head_analysis_v2.py'),'supervise'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
hold=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(new.pid)],start_new_session=True)
dump(dest/'launcher.json',{'pid':new.pid,'idle_hold_pid':hold.pid,'old_idle_pid':old,'quality_coordinator_unchanged':read(O/'coordinator.json')['pid'],'active_quality_child_unchanged':read(O/'active_child.json')['pid'],'wall':time.time()})
print(json.dumps({'new_idle_supervisor_pid':new.pid,'active_training_unchanged':True}))
