"""One persistent gate→prepare→evaluate→release→quality→analysis workflow."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from experiments.lapa_integrated_gate_v1 import REPORTS
from experiments.lapa_native_pilot_v2 import ARMS

OUT=REPORTS/'lapa_native_workflow_v1'
SCRIPTS=Path(__file__).parent


def run(name,*args):
    with (OUT/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,str(SCRIPTS/(name+'.py')),*args],stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'coordinator.json').write_text(json.dumps({'pid':os.getpid(),'started_unix':time.time()}))
    gates=REPORTS/'lapa_native_scheduled_gates_v2'
    deadline=time.monotonic()+1800
    while not all((gates/a/'receipt.json').exists() for a in ARMS):
        failures=list(gates.rglob('*failure.json'))
        if failures:raise RuntimeError(f'Gate failure: {failures}')
        if time.monotonic()>deadline:raise RuntimeError('Gate wait exceeded30min; inspect jobs')
        time.sleep(15)
    run('prepare_lapa_native_pilot_v1')
    run('check_lapa_native_evaluator_v1')
    run('release_lapa_native_quality_v1')
    run('operate_lapa_native_quality_v1')
    for arm in ARMS:run('audit_lapa_native_outputs_v1',arm)
    run('bootstrap_lapa_native_v1')
    run('report_lapa_native_v1')
    (OUT/'analysis_generated.json').write_text(json.dumps({'visual_review_pending':True,'final_interpretation_pending':True,'closure_verified':False}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(error)},f)
        raise
