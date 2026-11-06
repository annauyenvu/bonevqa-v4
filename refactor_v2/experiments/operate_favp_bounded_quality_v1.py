"""Exclusive quality worker→fresh replay→independent analysis; fixed four hours."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments.favp_visual_bridge_v2 import ROOT

OUT=ROOT/'reports/favp_bounded_quality_v1'


def main():
    started=time.perf_counter();deadline=started+14400
    assert json.loads((OUT/'execution_release.json').read_text())['quality_released']
    with (OUT/'coordinator.json').open('x') as f:json.dump({'pid':os.getpid(),'wall_budget_seconds':14400,'phase':'quality'},f)
    stages=[('quality','favp_bounded_quality_worker_v1.py',[]),('replay','favp_bounded_quality_worker_v1.py',['--replay'])]
    for name,script,args in stages:
        (OUT/'coordinator_progress.json').write_text(json.dumps({'pid':os.getpid(),'phase':name,'elapsed_seconds':time.perf_counter()-started}))
        remaining=deadline-time.perf_counter();assert remaining>0
        with (OUT/f'{name}.log').open('x') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'experiments'/script),*args],cwd=ROOT.parent,
                env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
    assert json.loads((OUT/'completed.json').read_text())['updates']==64
    assert json.loads((OUT/'replay.json').read_text())['exact_tokens_answers_metrics']
    (OUT/'execution_completed.json').write_text(json.dumps({'updates':64,'full_selected_replay':True,'seconds':time.perf_counter()-started}))
    with (OUT/'analysis.log').open('x') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'experiments/analyze_favp_bounded_pilot_v1.py')],cwd=ROOT.parent,
            env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    (OUT/'analysis_generated.json').write_text(json.dumps({'visual_review_and_final_interpretation_pending':True,
        'do_not_claim_closure':True,'seconds':time.perf_counter()-started}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        with (OUT/'operation_failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
