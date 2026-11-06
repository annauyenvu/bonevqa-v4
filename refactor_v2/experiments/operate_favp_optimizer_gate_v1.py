"""Exclusive sequential real AdamW gate then fresh-process replay; no retries."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/favp_pilot_optimizer_workflow_v1'


def main():
    OUT.mkdir(exist_ok=False)
    assert (ROOT/'reports/favp_pilot_inputs_v1/receipt.json').exists()
    (OUT/'started.json').write_text(json.dumps({'pid':os.getpid(),'quality_released':False}))
    for name,args in [('gate',[]),('replay',['--replay'])]:
        with (OUT/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'experiments/gate_favp_pilot_optimizer_v1.py'),*args],
                cwd=ROOT.parent,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    (OUT/'completed.json').write_text(json.dumps({'optimizer_gate_and_replay':True,'quality_released':False}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
