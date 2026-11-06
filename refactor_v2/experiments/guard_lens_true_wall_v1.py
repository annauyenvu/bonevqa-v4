"""External real-wall deadline guard; never edits a released training dependency."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
QUALITY=ROOT/'reports/lens_bounded_quality_v1'
OUT=ROOT/'reports/lens_bounded_wall_guard_v1'
WORKER=str(ROOT/'experiments/lens_bounded_quality_worker_v1.py')
COORDINATOR=str(ROOT/'experiments/operate_lens_bounded_quality_v1.py')


def process_rows():
    result=subprocess.run(['ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True)
    rows=[]
    for line in result.stdout.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)==3:rows.append((int(fields[0]),int(fields[1]),fields[2]))
    return rows


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    coordinator_path=QUALITY/'coordinator.json'
    coordinator=json.loads(coordinator_path.read_text())
    release=json.loads((QUALITY/'execution_release.json').read_text())
    assert release['quality_released'] and coordinator['cap_seconds']==release['wall_cap_seconds']==14400
    pid=coordinator['pid'];created=os.stat(coordinator_path).st_birthtime
    deadline=created+release['wall_cap_seconds']
    assert any(p==pid and COORDINATOR in command for p,_,command in process_rows()),'Coordinator identity not active'
    binding={'pid':os.getpid(),'coordinator_pid':pid,'coordinator_created_epoch':created,'deadline_epoch':deadline,
        'deadline_local':datetime.datetime.fromtimestamp(deadline).isoformat(),
        'anchor':'macOS coordinator.json birth time, created once immediately at quality start',
        'coordinator_receipt_sha256':sha(coordinator_path),'release_sha256':sha(QUALITY/'execution_release.json'),
        'script_sha256':sha(__file__),'reason':'clamshell suspension excluded from local monotonic timer; enforce original four-hour real-wall budget independently without editing sealed code'}
    (OUT/'started.json').write_text(json.dumps(binding,indent=2))
    while True:
        now=time.time()
        assert sha(coordinator_path)==binding['coordinator_receipt_sha256']
        if (QUALITY/'execution_completed.json').exists():
            finished=os.stat(QUALITY/'execution_completed.json').st_birthtime
            if finished>deadline:
                raise RuntimeError('Execution completion after original real-wall deadline; budget violation, do not close as compliant')
            (OUT/'completed.json').write_text(json.dumps({'execution_completed_within_true_wall':True,
                'true_wall_seconds':finished-created,'deadline_epoch':deadline,'source_sha256':sha(QUALITY/'execution_completed.json')}))
            return
        rows=process_rows()
        active=any(p==pid and COORDINATOR in cmd for p,_,cmd in rows)
        if not active:
            (OUT/'stopped.json').write_text(json.dumps({'reason':'coordinator no longer active without execution_completed; inspect existing failure','epoch':now}))
            return
        if now>=deadline:
            stopped=[]
            # Only verified immediate children of this exact coordinator, never broad-name termination.
            for child,parent,command in rows:
                if parent==pid and WORKER in command:
                    try:os.kill(child,signal.SIGTERM);stopped.append(child)
                    except ProcessLookupError:pass
            (OUT/'deadline_exceeded.json').write_text(json.dumps({'epoch':now,'deadline_epoch':deadline,
                'terminated_verified_worker_pids':stopped,'partial_outputs_preserved':True,'automatic_retry':False},indent=2))
            # Let the existing coordinator record its own failed-child receipt first.
            time.sleep(10)
            if any(p==pid and COORDINATOR in cmd for p,_,cmd in process_rows()):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            return
        (OUT/'progress.json').write_text(json.dumps({'epoch':now,'true_wall_elapsed_seconds':now-created,
            'remaining_seconds':deadline-now,'deadline_local':binding['deadline_local']}))
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
