"""Bounded serial native Lens gate/replay; no automatic retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/lens_native_trainability_workflow_v1'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'started.json').write_text(json.dumps({'pid':os.getpid(),'wall_cap_seconds':1500,'rss_cap_bytes':20*1024**3}))
    start=time.monotonic()
    for phase,extra in [('updates',[]),('fresh_replay',['--replay'])]:
        with (OUT/(phase+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,str(ROOT/'experiments/gate_lens_native_trainability_v1.py'),*extra],
                                   stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                if time.monotonic()-start>1500:
                    child.terminate();raise RuntimeError('25-minute gate wall cap; outputs preserved')
                result=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True,check=True)
                rss=int(result.stdout.strip() or 0)*1024
                (OUT/'progress.json').write_text(json.dumps({'phase':phase,'pid':child.pid,'rss_bytes':rss,
                    'elapsed_seconds':time.monotonic()-start}))
                if rss>20*1024**3:
                    child.terminate();raise RuntimeError('20-GiB RSS resource cap; outputs preserved')
                time.sleep(5)
            assert child.returncode==0, f'{phase} failed: inspect preserved log'
    receipt=json.loads((ROOT/'reports/lens_native_trainability_gate_v1/replay.json').read_text())
    assert receipt['exact']
    (OUT/'completed.json').write_text(json.dumps({'exact_fresh_replay':True,'seconds':time.monotonic()-start}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
