"""Detach only the independently gated capped runner from chat cancellation.

No power-setting changes or duplicate/retry loop. The capped runner retains all
scientific/data/source guards and stops on failure. Runtime remains offline.
"""
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from experiments import run_B_sequence_loss_pilots_v3_3 as recovery

core=recovery.core
if __name__=='__main__':
    core.idle();core.validate_plan()
    verification=core.OUT/'independent_execution_verification.json'
    verified=json.loads(verification.read_text())
    assert verified['passed'] and verified['plan_sha256']==core.sha(core.OUT/'plan.json')
    assert not (core.OUT/'progress.json').exists()
    assert not (core.OUT/'detached_launch.json').exists()
    environment=os.environ.copy()
    environment.update(HF_HOME=str(core.ROOT/'.evaluation_runtime/hf_cache'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_ENABLE_MPS_FALLBACK='1',
        OMP_NUM_THREADS='8',TMP='/private/tmp',TEMP='/private/tmp',
        PYTHONPATH=str(core.ROOT/'.evaluation_runtime/deps')+':'+str(core.ROOT/'refactor_v2'),
        BONEVQA_DATA=str(core.ROOT/'.evaluation_runtime/data'))
    command=['/usr/bin/caffeinate','-i','/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python',
        str(Path(recovery.__file__).resolve()),'run']
    with (core.OUT/'detached_driver.log').open('x') as log:
        child=subprocess.Popen(command,cwd=core.ROOT,env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    core.dump(core.OUT/'detached_launch.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'launch_pid':child.pid,'command':command,'independent_verification_sha256':core.sha(verification),
        'launcher_sha256':core.sha(Path(__file__).resolve()),'start_new_session':True,
        'scope':'Exact two serial gated296update pilots; no automatic retry or long training',
        'offline':True,'power_settings_changed':False})
    print('Detached gated runner launch PID',child.pid)
