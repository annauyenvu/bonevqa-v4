"""Finish only the already sealed V/L pilot set, serially, with verification.

Wait for the existing V control. No training is duplicated or automatically
retried; every failure stops. No expansion beyond the existing five-arm seal.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone


def main():
    folder=Path.cwd()/'refactor_v2/reports/vl_bounded_quality_v3_1'
    path=folder/'remaining_VL_continuation_progress.json'
    assert not path.exists(), 'Existing continuation; inspect instead'
    assert all(not (folder/a).exists() for a in ('L_real','L_original_teacher','L_location_placebo'))
    state={'status':'waiting_existing_V_location_placebo','pid':os.getpid(),
           'started_utc':datetime.now(timezone.utc).isoformat(),
           'scope':'finish verified Vpair and ONLY three existing sealed L148+148 arms; no retries',
           'long_training_released':False}
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    def command(label,args):
        state.update(status=label);save()
        with (folder/(label+'_driver.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid);save()
            if child.wait()!=0:raise RuntimeError(label+' failed; preserve and stop')
    save()
    try:
        while True:
            progress=json.loads((folder/'V_location_placebo/progress.json').read_text())
            if progress['status'].startswith('failed'):raise RuntimeError('Existing V control failed')
            if progress['status']=='complete_pending_independent_evaluation' and not (folder/'ACTIVE.lock').exists():
                try:os.kill(progress['pid'],0)
                except ProcessLookupError:break
            else:os.kill(progress['pid'],0)
            time.sleep(10)
        command('verify_V_location_placebo',['refactor_v2/experiments/verify_vl_bounded_completed_arm_v3_1.py','--arm','V_location_placebo'])
        command('compare_verified_V_pair',['refactor_v2/experiments/compare_vl_bounded_verified_v3_1.py','--group','V'])
        for arm in ('L_real','L_original_teacher','L_location_placebo'):
            assert not (folder/'ACTIVE.lock').exists() and not (folder/arm).exists()
            command('run_'+arm,['refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py','--run',arm])
            command('verify_'+arm,['refactor_v2/experiments/verify_vl_bounded_completed_arm_v3_1.py','--arm',arm])
            report=json.loads((folder/arm/'independent_LAST296_verification.json').read_text())
            assert report['passed'] and report['global_step']==296 and not report['calibration_or_test_used']
            state.setdefault('completed_verified_L_arms',[]).append(arm);save()
        command('compare_verified_L_controls',['refactor_v2/experiments/compare_vl_bounded_verified_v3_1.py','--group','L'])
        state.update(status='five_bounded_arms_verified_development_comparisons_complete_NOTfull_step3',
                     finished_utc=datetime.now(timezone.utc).isoformat(),clinical_efficacy_or_production_release=False);save()
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));save();raise


if __name__=='__main__':main()
