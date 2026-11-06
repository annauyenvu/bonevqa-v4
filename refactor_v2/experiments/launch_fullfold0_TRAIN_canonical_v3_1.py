"""Single detached TRAIN-fit inference/replay job; no optimization."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'
SOURCE=ROOT/'refactor_v2/experiments/diagnose_fullfold0_TRAIN_canonical_v3_1.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    audit=json.loads((OUT/'independent_preflight.json').read_text())
    assert audit['passed'] and audit['protocol_sha256']==sha(OUT/'protocol.json')
    assert audit['worker_sha256']==sha(SOURCE)
    assert not (OUT/'progress.json').exists() and not (OUT/'detached_launch.json').exists()
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():continue
        if Path(parts[1].split()[0]).name.lower().startswith('python'):
            assert not any(s in parts[1] for s in (' -m src.train ',
                'run_fullfold0_B_uniform_screen_v3_1.py worker ',
                'evaluate_fullfold0_fixed_LAST_v3_1.py run',
                'diagnose_fullfold0_TRAIN_canonical_v3_1.py run',
                'diagnose_fullfold0_TRAIN_canonical_v3_1.py replay'))
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    assert os.environ['OMP_NUM_THREADS']=='8'
    with (OUT/'detached_diagnostic.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,str(SOURCE),'run'],
            cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'launch_pid':child.pid,
        'protocol_sha256':sha(OUT/'protocol.json'),'preflight_sha256':sha(OUT/'independent_preflight.json'),
        'optimizer_updates':0,'calibration_test_inference':False,'no_automatic_retry':True,
        'idle_guard_only_not_sleep_immunity':True}
    (OUT/'detached_launch.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
