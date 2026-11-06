"""Bounded supervisor for an already running projector gate; never retrains."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / 'reports/favp_projector_update_v1'
OUT = ROOT / 'reports/favp_projector_supervisor_v1'


def main():
    OUT.mkdir(exist_ok=False)
    worker_pid = 52676
    deadline = time.monotonic() + 7200
    (OUT / 'started.json').write_text(json.dumps({'worker_pid': worker_pid, 'supervisor_pid': os.getpid(),
        'wait_budget_seconds': 7200, 'replay_budget_seconds': 1800, 'action': 'replay only; no training retry'}, indent=2))
    while True:
        assert not (GATE / 'failure.json').exists(), 'Training gate failed; preserve and inspect before retry'
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline, 'Wait budget exceeded; do not kill or duplicate original worker'
        time.sleep(15)
    assert (GATE / 'update.json').exists(), 'Worker exited without update receipt'
    assert not (GATE / 'replay.json').exists(), 'Replay already exists; do not duplicate'
    assert not (GATE / 'replay_failure.json').exists(), 'Preserve replay failure; no automatic retry'
    with (OUT / 'replay.log').open('x') as log:
        subprocess.run([sys.executable, '-u', str(ROOT / 'experiments/check_favp_projector_update_v1.py'), '--replay'],
                       cwd=ROOT.parent, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=1800)
    receipt = json.loads((GATE / 'replay.json').read_text())
    assert receipt['exact_logits'] and receipt['exact_loss'] and receipt['strict_projector_reload']
    (OUT / 'completed.json').write_text(json.dumps({'replay_accepted': True, 'quality_released': False}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            with (OUT / 'failure.json').open('x') as f:
                json.dump({'error': repr(exc)}, f)
        raise
