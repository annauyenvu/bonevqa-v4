"""Exclusive isolated environment resolution/install/import workflow, no models."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OUT=ROOT/'reports/lens_runtime_v2'
VENV=PROJECT/'.evaluation_runtime/lens_venv_v2'


def main():
    OUT.mkdir(exist_ok=False)
    assert not VENV.exists(),'Do not overwrite an existing environment'
    stages=[('venv',[sys.executable,'-m','venv',str(VENV)],120),
        ('resolution',[str(VENV/'bin/python'),'-m','pip','--isolated','install','--dry-run','--report',str(OUT/'resolution.json'),
            '--only-binary=:all:','--index-url','https://pypi.org/simple','--cache-dir',str(PROJECT/'.evaluation_runtime/pip_cache_lens_v1'),
            '-r',str(ROOT/'experiments/lens_runtime_requirements_v2.txt')],300),
        ('installation',[sys.executable,'-u',str(ROOT/'experiments/install_lens_runtime_v2.py')],1200)]
    for name,command,timeout in stages:
        (OUT/'workflow_progress.json').write_text(json.dumps({'phase':name}))
        with (OUT/f'{name}_workflow.log').open('x') as log:
            subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    (OUT/'workflow_completed.json').write_text(json.dumps({'runtime_imports_accepted':True,'models_loaded':False}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'workflow_failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
