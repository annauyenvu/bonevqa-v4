"""Wait existing gates, release fixed quality, run both arms/replays/audit/plot once."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/lens_bounded_workflow_v1'


def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic()
    (OUT/'started.json').write_text(json.dumps({'pid':os.getpid(),'python':sys.executable}))
    gates=ROOT/'reports/lens_bounded_gate_workflow_v1'
    while not (gates/'completed.json').exists():
        assert not (gates/'failure.json').exists(),'Scheduled gate failed; no automatic retry'
        assert time.monotonic()-start<2700,'45-minute existing-gate wait cap'
        (OUT/'progress.json').write_text(json.dumps({'phase':'waiting_existing_gates','seconds':time.monotonic()-start}))
        time.sleep(10)
    commands=[('release','release_lens_bounded_quality_v1.py',sys.executable),
        ('quality_and_replays','operate_lens_bounded_quality_v1.py',sys.executable),
        ('independent_analysis','audit_lens_bounded_outputs_v1.py',sys.executable),
        ('plot','plot_lens_bounded_pilot_v1.py','/Users/uyen/Downloads/BoneMedVQA/.venv/bin/python')]
    for phase,file,python in commands:
        (OUT/'progress.json').write_text(json.dumps({'phase':phase,'seconds':time.monotonic()-start}))
        with (OUT/(phase+'.log')).open('x') as log:
            subprocess.run([python,str(ROOT/'experiments'/file)],stdout=log,stderr=subprocess.STDOUT,check=True,
                           timeout=14500 if phase=='quality_and_replays' else 900)
    (OUT/'analysis_generated.json').write_text(json.dumps({'execution_audits_bootstrap_and_plot_complete':True,
        'actual_visual_review_final_interpretation_closure_pending':True,'seconds':time.monotonic()-start}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
