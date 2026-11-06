"""Sequential released pilot plus full fresh-process replays; preserve failures."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'refactor_v2/reports/lapa_shared_quality_v1'
WORKER=Path(__file__).with_name('lapa_quality_worker_v1.py')


def main():
    assert (OUT/'execution_release.json').exists()
    # Exclusive coordinator marker prevents accidental duplicate starts.
    with (OUT/'coordinator.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time()},f)
    for arm in ('carrier','train_cs0','train_cs1','random_cs0','random_cs1'):
        for mode in ('quality','replay'):
            args=[sys.executable,str(WORKER),arm]+(['--replay'] if mode=='replay' else [])
            tick=time.perf_counter()
            with (OUT/f'{arm}_{mode}.log').open('x') as log:
                subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            with (OUT/f'{arm}_{mode}_timing.json').open('x') as f:json.dump({'seconds':time.perf_counter()-tick},f)
    with (OUT/'execution_completed.json').open('x') as f:json.dump({'arms':5,'updates_per_arm':292,'full_replays':5,'analysis_complete':False},f)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():
            with (OUT/'operation_failure.json').open('x') as f:json.dump({'error':repr(error),'preserve_artifacts':True},f)
        raise
