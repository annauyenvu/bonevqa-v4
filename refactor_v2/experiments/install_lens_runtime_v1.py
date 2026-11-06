"""Install resolved hashed binary wheels only into the new isolated Lens venv."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OUT=ROOT/'reports/lens_runtime_v1'
PYTHON=PROJECT/'.evaluation_runtime/lens_venv_v1/bin/python'


def inspect(entry):
    info=entry['download_info'];url=info['url'];parsed=urllib.parse.urlparse(url)
    assert parsed.scheme=='https' and parsed.hostname=='files.pythonhosted.org'
    filename=urllib.parse.unquote(parsed.path).split('/')[-1]
    assert filename.endswith('.whl') and ('arm64' in filename or 'universal2' in filename or 'none-any' in filename),filename
    name=entry['metadata']['name'];version=entry['metadata']['version']
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+_-]*',version)
    digest=info['archive_info']['hashes']['sha256'];assert re.fullmatch(r'[0-9a-f]{64}',digest)
    req=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'BoneVQA-Lens-runtime-audit/1'})
    with urllib.request.urlopen(req,timeout=45) as response:
        size=int(response.headers['Content-Length']);assert size>0
    return {'name':name,'version':version,'filename':filename,'bytes':size,'sha256':digest,'url':url}


def main():
    assert not (OUT/'installation_started.json').exists(),'Do not duplicate installation'
    resolution=OUT/'resolution.json';report=json.loads(resolution.read_text())
    with ThreadPoolExecutor(max_workers=6) as pool:files=list(pool.map(inspect,report['install']))
    total=sum(f['bytes'] for f in files);assert total<1536*1024**2,'Unexpected bulk download plan'
    assert shutil.disk_usage(PROJECT).free>8*1024**3+6*total+1024**3
    lock=OUT/'requirements_locked.txt'
    lock.write_text(''.join(f"{f['name']}=={f['version']} --hash=sha256:{f['sha256']}\n" for f in files))
    (OUT/'wheel_plan.json').write_text(json.dumps({'files':files,'download_bytes':total,
        'resolution_sha256':hashlib.sha256(resolution.read_bytes()).hexdigest(),
        'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'old_runtime_modified':False},indent=2))
    (OUT/'installation_started.json').write_text(json.dumps({'python':str(PYTHON),'packages':len(files),'download_bytes':total}))
    with (OUT/'install.log').open('x') as log:
        subprocess.run([str(PYTHON),'-m','pip','--isolated','install','--require-hashes','--no-deps','--only-binary=:all:',
            '--index-url','https://pypi.org/simple','--cache-dir',str(PROJECT/'.evaluation_runtime/pip_cache_lens_v1'),
            '-r',str(lock)],cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
    with (OUT/'pip_check.log').open('x') as log:
        subprocess.run([str(PYTHON),'-m','pip','check'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
    code="""import json,importlib.metadata as m
from transformers import AutoProcessor,Idefics3ForConditionalGeneration,SamModel,SamProcessor
import torch,peft,accelerate,datasets
print(json.dumps({'versions':{n:m.version(n) for n in ['torch','torchvision','transformers','peft','accelerate','datasets','safetensors','numpy','Pillow']},'Idefics3_import':True,'SAM_import':True,'cuda_available':torch.cuda.is_available()}))
"""
    result=subprocess.run([str(PYTHON),'-c',code],capture_output=True,text=True,check=True,timeout=120)
    (OUT/'import.log').write_text(result.stderr)
    acceptance=json.loads(result.stdout.strip())
    acceptance.update({'runtime_imports_accepted':True,'old_runtime_modified':False,'models_loaded':False,
                       'download_bytes':total,'packages':len(files),'scope':'isolated binary-wheel runtime installation/import only; no model or training gate'})
    (OUT/'receipt.json').write_text(json.dumps(acceptance,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
