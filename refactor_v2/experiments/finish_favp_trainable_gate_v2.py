"""Wait for existing version-3 native-route gate then replay once; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT/'reports/favp_trainable_route_gate_v3'
OUT = ROOT/'reports/favp_trainable_gate_supervisor_v2'


def main():
    OUT.mkdir(exist_ok=False)
    pid=53682
    deadline=time.monotonic()+2100
    (OUT/'started.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'worker_pid':pid,
        'wait_budget_seconds':2100,'replay_budget_seconds':600,'action':'fresh replay only; no training retry'},indent=2))
    while True:
        assert not (GATE/'failure.json').exists(),'Gate failure preserved; inspect before any retry'
        try:
            os.kill(pid,0)
        except ProcessLookupError:
            break
        assert time.monotonic()<deadline,'Wait budget exceeded; do not duplicate/kill worker'
        time.sleep(10)
    assert (GATE/'update.json').exists(),'Worker exited without update receipt'
    assert not (GATE/'replay.json').exists() and not (GATE/'replay_failure.json').exists()
    with (OUT/'replay.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'experiments/gate_favp_trainable_route_v3.py'),'--replay'],
            cwd=ROOT.parent,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    replay=json.loads((GATE/'replay.json').read_text())
    assert replay['exact_logits'] and replay['exact_loss'] and replay['strict_mutable_reload']
    (OUT/'completed.json').write_text(json.dumps({'update_and_replay_accepted':True,'quality_released':False},indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:
                json.dump({'error':repr(exc)},f)
        raise

