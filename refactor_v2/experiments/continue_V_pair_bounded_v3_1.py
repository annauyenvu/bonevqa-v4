"""Wait for the existing real arm, verify it, then run only its sealed control.

No model imports while waiting; no retries, new arms, or budget expansion.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone


def main():
    root=Path.cwd()
    folder=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    output=folder/'V_pair_continuation_progress.json'
    assert not output.exists(), 'Existing continuation; do not duplicate'
    assert not (folder/'V_location_placebo').exists(), 'Existing control; do not duplicate'
    state={'status':'waiting_for_existing_V_real','pid':os.getpid(),
           'started_utc':datetime.now(timezone.utc).isoformat(),
           'scope':'verify existing V_real then ONE sealed V_location_placebo296step arm only',
           'long_training_released':False}
    def save():output.write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        while True:
            progress=json.loads((folder/'V_real/progress.json').read_text())
            if progress['status'].startswith('failed'):
                raise RuntimeError('V_real failed; preserve and stop')
            if progress['status']=='complete_pending_independent_evaluation' and not (folder/'ACTIVE.lock').exists():
                break
            # A stale running record is not evidence of active work.
            os.kill(progress['pid'],0)
            time.sleep(10)
        state['status']='independent_V_real_verification';save()
        with (folder/'V_real_independent_driver.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'refactor_v2/experiments/verify_vl_bounded_completed_arm_v3_1.py','--arm','V_real'],stdout=log,stderr=subprocess.STDOUT)
            state['verification_pid']=child.pid;save()
            if child.wait()!=0:raise RuntimeError('Independent V_real verification failed; no control launched')
        report=json.loads((folder/'V_real/independent_LAST296_verification.json').read_text())
        assert report['passed'] and report['global_step']==296 and not report['calibration_or_test_used']
        assert not (folder/'ACTIVE.lock').exists()
        state['status']='running_sealed_V_location_placebo';save()
        with (folder/'V_location_placebo_driver.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py','--run','V_location_placebo'],stdout=log,stderr=subprocess.STDOUT)
            state['control_pid']=child.pid;save()
            if child.wait()!=0:raise RuntimeError('V_location_placebo failed; preserve and stop')
        state.update(status='V_pair_training_complete_control_pending_independent_evaluation',finished_utc=datetime.now(timezone.utc).isoformat());save()
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));save();raise


if __name__=='__main__':main()
