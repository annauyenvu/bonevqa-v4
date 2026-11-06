"""Exclusive sequential native-backbone pilot and independent full replays."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from experiments.lapa_native_pilot_v2 import ARMS
from experiments.lapa_integrated_gate_v1 import REPORTS

OUT=REPORTS/'lapa_native_quality_v1'
HERE=Path(__file__).parent


def main():
    assert json.loads((OUT/'execution_release.json').read_text())['quality_released']
    with (OUT/'coordinator.json').open('x') as f:json.dump({'pid':os.getpid(),'started_unix':time.time()},f)
    for arm in ARMS:
        for mode in ('quality','replay'):
            tick=time.perf_counter()
            with (OUT/f'{arm}_{mode}.log').open('x') as log:
                subprocess.run([sys.executable,str(HERE/'lapa_native_quality_worker_v1.py'),arm]+(['--replay'] if mode=='replay' else []),
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            (OUT/f'{arm}_{mode}_timing.json').write_text(json.dumps({'seconds':time.perf_counter()-tick}))
    (OUT/'execution_completed.json').write_text(json.dumps({'arms':5,'updates_per_arm':292,'full_replays':5,'analysis_complete':False}))


if __name__=='__main__':
    try:main()
    except Exception as e:
        with (OUT/'operation_failure.json').open('x') as f:json.dump({'error':repr(e)},f)
        raise
