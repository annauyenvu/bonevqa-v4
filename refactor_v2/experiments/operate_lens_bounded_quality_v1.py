"""Two sequential bounded arms with independent processes and exact fresh replay."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiments.lens_bounded_route_v1 import ROOT,QUALITY,sha


def main():
    assert not (QUALITY/'coordinator.json').exists()
    release=json.loads((QUALITY/'execution_release.json').read_text());assert release['quality_released']
    start=time.monotonic()
    (QUALITY/'coordinator.json').write_text(json.dumps({'pid':os.getpid(),'cap_seconds':14400}))
    for arm in release['arms']:
        for phase,extra in [('quality',[]),('replay',['--replay'])]:
            assert all(sha(Path(p))==h for p,h in release['dependencies'].items())
            with (QUALITY/(arm+'_'+phase+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,str(ROOT/'experiments/lens_bounded_quality_worker_v1.py'),arm,*extra],stdout=log,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    reason=None
                    if time.monotonic()-start>release['wall_cap_seconds']:reason='4h quality+replay wall cap'
                    if shutil.disk_usage(ROOT).free<release['disk_reserve_bytes']:reason='8-GiB disk reserve'
                    result=subprocess.run(['ps','-o','rss=','-p',str(child.pid)],capture_output=True,text=True)
                    rss=int(result.stdout.strip() or 0)*1024
                    if rss>release['rss_cap_bytes']:reason='20-GiB process RSS cap'
                    (QUALITY/'coordinator_progress.json').write_text(json.dumps({'arm':arm,'phase':phase,'pid':child.pid,
                        'rss_bytes':rss,'seconds':time.monotonic()-start}))
                    if reason:
                        child.terminate();raise RuntimeError(reason+'; partial outputs preserved')
                    time.sleep(5)
                assert child.returncode==0,f'{arm} {phase} failed; inspect log and failure receipt, no automatic retry'
    assert all(sha(Path(p))==h for p,h in release['dependencies'].items())
    a=json.loads((QUALITY/'ce_only/PRE.json').read_text());b=json.loads((QUALITY/'lens_losses/PRE.json').read_text())
    assert a['rows']==b['rows'] and a['metrics']==b['metrics'],'Matched PRE outputs differ'
    (QUALITY/'execution_completed.json').write_text(json.dumps({'all_two_arms_and_fresh_replays_complete':True,
        'matched_PRE_exact':True,'execution_dependencies_unchanged':len(release['dependencies']),
        'seconds':time.monotonic()-start,'analysis_pending':True},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if QUALITY.exists():
            with (QUALITY/'operation_failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
