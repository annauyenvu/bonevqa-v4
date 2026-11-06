"""Recover analysis after repeated audit-log filename collision; no model runs."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'refactor_v2/reports/lapa_native_analysis_v2'
STUDY=ROOT/'refactor_v2/reports/lapa_native_quality_v1'
SCRIPTS=Path(__file__).parent


def run(script,*args):
    name=script+('_'+args[0] if args else '')+'.log'
    with (OUT/name).open('x') as log:
        subprocess.run([sys.executable,str(SCRIPTS/(script+'.py')),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(exist_ok=False)
    assert json.loads((STUDY/'execution_completed.json').read_text())['full_replays']==5
    for arm in ['carrier','gm_lf','pf_author','pf_residual','pf_residual_rewired']:
        if not (STUDY/arm/'independent_metric_audit.json').exists():run('audit_lapa_native_outputs_v1',arm)
    run('bootstrap_lapa_native_v1')
    run('report_lapa_native_v1')
    (OUT/'completed.json').write_text(json.dumps({'analyses_generated':True,'visual_review_pending':True,
        'previous_failure':'workflow reused exclusive audit log filename across arms; preserved in lapa_native_workflow_v1'}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
