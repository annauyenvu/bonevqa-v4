"""Detach exact independently verified scoped pair, no retry or power-setting edits."""
import os
import json
import subprocess
from pathlib import Path
from experiments import run_fullfold0_B_uniform_screen_v3_1 as worker


if __name__=='__main__':
    worker.idle();worker.validate();out=worker.OUT
    check=json.loads((out/'prelaunch_independent_verification.json').read_text())
    assert check['passed'] and check['screen_protocol_sha256']==worker.sha(out/'protocol.json')
    assert check['runtime_binding_sha256']==worker.sha(out/'runtime_binding.json')
    assert check['trainer_integration_sha256']==worker.sha(out/'trainer_integration_verification.json')
    assert not (out/'progress.json').exists() and not (out/'detached_launch.json').exists()
    command=['/usr/bin/caffeinate','-i','/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python',str(Path(worker.__file__).resolve()),'run']
    with (out/'detached_driver.log').open('x') as log:
        child=subprocess.Popen(command,cwd=worker.ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    worker.dump(out/'detached_launch.json',{'started_utc':worker.now(),'launch_pid':child.pid,
        'command':command,'launcher_sha256':worker.sha(Path(__file__).resolve()),
        'prelaunch_verification_sha256':worker.sha(out/'prelaunch_independent_verification.json'),
        'start_new_session':True,'offline_CPU_threads':8,'power_settings_changed':False,
        'scope':'Only fixed fold0 two serial1438-update arms; no automatic retry/fivefold or other long stage'})
    print('Detached exact fullfold0 pair PID',child.pid,flush=True)
