"""One-shot detached fixed LAST evaluation after separate checkpoint audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'
SOURCE=ROOT/'refactor_v2/experiments/evaluate_fullfold0_fixed_LAST_v3_1.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    audit=json.loads((OUT/'checkpoint_preflight.json').read_text())
    assert audit['passed'] and audit['evaluator_sha256']==sha(SOURCE)
    assert audit['plan_sha256']==sha(OUT/'evaluation_plan.json')
    assert set(audit['checkpoint_checks'])=={'token_mean','sequence_uniform'}
    assert not (OUT/'detached_launch.json').exists() and not (OUT/'progress.json').exists()
    for row in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        parts=row.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid():
            continue
        executable=Path(parts[1].split()[0]).name.lower()
        if executable.startswith('python'):
            assert not any(term in parts[1] for term in (
                'run_fullfold0_B_uniform_screen_v3_1.py worker ',
                'evaluate_fullfold0_fixed_LAST_v3_1.py run',
                'evaluate_fullfold0_fixed_LAST_v3_1.py replay',' -m src.train '))
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    assert os.environ['OMP_NUM_THREADS']=='8'
    with (OUT/'detached_evaluation.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/caffeinate','-i',sys.executable,str(SOURCE),'run'],
            cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'started_utc':datetime.now(timezone.utc).isoformat(),'guard_pid':child.pid,
        'checkpoint_audit_sha256':sha(OUT/'checkpoint_preflight.json'),
        'plan_sha256':sha(OUT/'evaluation_plan.json'),'source_sha256':sha(SOURCE),
        'training_updates_authorized':0,'calibration_test_allowed':False,
        'no_automatic_retry':True,'idle_guard_only_not_sleep_immunity':True}
    (OUT/'detached_launch.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
