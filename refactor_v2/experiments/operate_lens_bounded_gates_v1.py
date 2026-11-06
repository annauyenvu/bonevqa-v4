"""Wait for existing preparation, then two sequential arm gates and fresh replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/lens_bounded_gate_workflow_v1'


def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    (OUT/'started.json').write_text(json.dumps({'pid':os.getpid(),'cap_seconds':2400,'rss_cap_bytes':20*1024**3}))
    prep=ROOT/'reports/lens_bounded_preparation_v1'
    while not (prep/'preparation_seal.json').exists():
        assert not (prep/'failure.json').exists(),'Existing preparation failed; preserve and inspect'
        assert time.monotonic()-start<900,'Preparation wait cap'
        time.sleep(5)
    for arm in ['ce_only','lens_losses']:
        for phase,extra in [('updates',[]),('replay',['--replay'])]:
            with (OUT/(arm+'_'+phase+'.log')).open('x') as log:
                process=subprocess.Popen([sys.executable,str(ROOT/'experiments/gate_lens_bounded_optimizer_v1.py'),arm,*extra],stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    if time.monotonic()-start>2400:
                        process.terminate();raise RuntimeError('40-minute gate workflow cap')
                    result=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True)
                    rss=int(result.stdout.strip() or 0)*1024
                    (OUT/'progress.json').write_text(json.dumps({'arm':arm,'phase':phase,'pid':process.pid,'rss_bytes':rss,'seconds':time.monotonic()-start}))
                    if rss>20*1024**3:
                        process.terminate();raise RuntimeError('20-GiB process RSS guard')
                    time.sleep(5)
                assert process.returncode==0,f'{arm} {phase} failed; no retry, inspect logs'
    (OUT/'completed.json').write_text(json.dumps({'both_arm_accumulated_scheduled_gates_and_replays':True,'seconds':time.monotonic()-start}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
