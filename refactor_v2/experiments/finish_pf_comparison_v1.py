"""Wait for existing coordinator, then analyze saved outputs once."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'refactor_v2/reports/pf_comparison_quality_v1'
SCRIPTS=Path(__file__).parent


def run(name,*args):
    with (OUT/(name+'.log')).open('a') as log:
        subprocess.run([sys.executable,str(SCRIPTS/(name+'.py')),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    with (OUT/'analysis_supervisor.json').open('x') as f:json.dump({'pid':os.getpid(),'started':time.time()},f)
    while not (OUT/'execution_completed.json').exists():
        if (OUT/'operation_failure.json').exists():raise RuntimeError('Execution failure; inspect before retry')
        time.sleep(30)
    for arm in ('author_real','masked_real','residual_real','residual_rewired'):
        if not (OUT/arm/'independent_metric_audit.json').exists():run('audit_pf_comparison_outputs_v1',arm)
    if not (OUT/'paired_component_bootstrap.json').exists():run('bootstrap_pf_comparison_v1')
    if not (OUT/'REFERENCE_PILOT_RESULTS.md').exists():run('report_pf_comparison_v1')
    with (OUT/'analysis_generated.json').open('x') as f:json.dump({'saved_output_analyses_generated':True,'visual_review_pending':True,'closure_verified':False},f)


if __name__=='__main__':
    try:main()
    except Exception as e:
        with (OUT/'analysis_failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise

