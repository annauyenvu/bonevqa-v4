"""Register one postprocessor and one external wall guard, without restarting training."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
assert (O/'coordinator.json').exists()
assert not (O/'postprocess_launcher.json').exists()
paths=[ROOT/'refactor_v2/experiments/local_noaux_head_analysis_v1.py',ROOT/'refactor_v2/experiments/guard_local_noaux_wall_v1.py',ROOT/'refactor_v2/reports/LOCAL_NOAUX_REFINEMENT_PROTOCOL_20261027.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (O/'head_analysis_release.json').open('x') as f:json.dump({'dependencies':{str(p):sha(p) for p in paths},'created':time.time(),'head_epochs':10,'head_batch':64,'threshold_target_specificity':.90,'bootstrap_draws':2000,'bootstrap_seed':20261105,'preparation_sha256':sha(O/'preparation.json')},f,indent=2)
started={}
for name,script,arg in [('postprocess','local_noaux_head_analysis_v1.py',['supervise']),('wall_guard','guard_local_noaux_wall_v1.py',[])]:
    log=(O/(name+'_supervisor.log')).open('x')
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'refactor_v2/experiments'/script)]+arg,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    started[name]=p.pid
hold=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(started['postprocess'])],start_new_session=True)
started['idle_hold']=hold.pid
with (O/'postprocess_launcher.json').open('x') as f:json.dump(started,f,indent=2)
print(json.dumps(started))
