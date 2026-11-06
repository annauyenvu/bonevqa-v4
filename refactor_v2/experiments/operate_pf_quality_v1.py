"""Exclusive, sequential bounded PF quality and independent full replays."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from experiments.pf_comparison_v1 import ARMS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'refactor_v2/reports/pf_comparison_quality_v1'
WORKER=Path(__file__).with_name('pf_quality_worker_v1.py')


def main():
    assert json.loads((OUT/'execution_release.json').read_text())['quality_released']
    with (OUT/'coordinator.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time()},f)
    for arm in ARMS:
        for mode in ('quality','replay'):
            command=[sys.executable,str(WORKER),arm]+(['--replay'] if mode=='replay' else [])
            tick=time.perf_counter()
            with (OUT/f'{arm}_{mode}.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (OUT/f'{arm}_{mode}_timing.json').open('x') as f:json.dump({'seconds':time.perf_counter()-tick},f)
    with (OUT/'execution_completed.json').open('x') as f:json.dump({'arms':4,'updates_per_arm':292,'full_replays':4,'analysis_complete':False},f)


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            with (OUT/'operation_failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
