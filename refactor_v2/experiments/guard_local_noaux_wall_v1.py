"""External true-wall guard; never modifies any scientific/training source."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
def read(p):return json.loads(p.read_text())
def dump(p,v):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def live(pid,needle):
    p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
    return p.returncode==0 and needle in p.stdout
dest=O/'wall_guard';dest.mkdir(exist_ok=False);coordinator=read(O/'coordinator.json');deadline=coordinator['deadline']
dump(dest/'started.json',{'pid':os.getpid(),'deadline':deadline,'coordinator':coordinator})
while True:
    if (O/'failure.json').exists() or (O/'postprocess_failure.json').exists():
        dump(dest/'completed.json',{'stopped_on_preserved_failure':True,'wall':time.time()});break
    if (O/'postprocess/completed.json').exists():
        assert time.time()<=deadline;dump(dest/'completed.json',{'execution_and_analysis_within_true_wall':True,'wall':time.time(),'deadline':deadline});break
    if time.time()>deadline:
        stopped=[]
        for file,needle in (('active_child.json','local_noaux_refinement_v1.py'),('postprocess/progress.json','local_noaux_head_analysis_v1.py'),('coordinator.json','local_noaux_refinement_v1.py'),('postprocess/started.json','local_noaux_head_analysis_v1.py')):
            p=O/file
            if p.exists():
                pid=read(p)['pid']
                if live(pid,needle):os.kill(pid,signal.SIGTERM);stopped.append(pid)
        dump(dest/'deadline_exceeded.json',{'deadline':deadline,'wall':time.time(),'terminated_verified_study_pids':stopped,'preserve_partials':True});break
    dump(dest/'progress.json',{'wall':time.time(),'deadline':deadline,'seconds_remaining':deadline-time.time()})
    time.sleep(20)
