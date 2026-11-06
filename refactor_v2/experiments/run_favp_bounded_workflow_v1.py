"""Wait existing fresh preparation, release once, then bounded pilot+analysis."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/favp_bounded_workflow_v1'
PREP=ROOT/'reports/favp_bounded_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False)
    deadline=time.monotonic()+600
    pid=54103
    (OUT/'coordinator.json').write_text(json.dumps({'pid':os.getpid(),'preparation_pid':pid,
        'quality_started':False,'visual_review_required_after_analysis':True},indent=2))
    while True:
        assert not (PREP/'failure.json').exists(),'Preparation failed; no automatic retry'
        try:os.kill(pid,0)
        except ProcessLookupError:break
        assert time.monotonic()<deadline,'Preparation wait budget exceeded'
        time.sleep(5)
    assert (PREP/'preparation_seal.json').exists()
    for phase,script,timeout in [('release','release_favp_bounded_pilot_v1.py',300),
                                 ('quality_and_analysis','operate_favp_bounded_quality_v1.py',14800)]:
        (OUT/'progress.json').write_text(json.dumps({'pid':os.getpid(),'phase':phase}))
        with (OUT/f'{phase}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'experiments'/script)],cwd=ROOT.parent,
                env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    (OUT/'completed.json').write_text(json.dumps({'pilot_analysis_generated':True,'visual_review_final_interpretation_closure_pending':True}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
