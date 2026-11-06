from common import *
import shutil
setup(42)
if (P/'LOCK.json').exists():raise RuntimeError('Already sealed; use explicit resume process after inspecting status')
assert read(P/'smoke_result.json')['success']
# Lock deterministic evaluator changes before any test score or main training is observed.
write(P/'determinism_amendment.json',{'training_lens_views':'global seeded Python random instead of new Random()','training_lens_alpha':'seeded global NumPy normal, same mean/std/clipping','evaluation_lens_alpha':.5,'evaluation_lens_views':'fixed first two modes as main','dataloader':'independent torch.Generator seeded by prespecified training seed; workers=0','mps_determinism':'torch.use_deterministic_algorithms(True), fail on unsupported nondeterministic operation','sam_fallback':'explicit CPU linear interpolation with per-image and operator counts; PYTORCH_ENABLE_MPS_FALLBACK=1','scope':'Same implementation for all 30 primary runs and sanity repeat; historical results not pooled'})
lock=read(P/'inputs_lock.json');files=lock['files']
for f in P.glob('*'):
 if f.is_file() and f.name!='LOCK.json':files[str(f.relative_to(ROOT))]=sha(f)
for f in CODE.rglob('*.py'):files[str(f.relative_to(ROOT))]=sha(f)
for f in (LEGACY/'deps/open_clip').rglob('*.py'):files[str(f)]=sha(f)
files.update(lock['pretrained']);write(P/'LOCK.json',{'locked_at':datetime.datetime.now().isoformat(),'files':files})
# Materialize the exact 30 run configs now, before looking at any test results.
from engine import run_config
for seed in SEEDS:
 for regime in REGIMES:
  for variant in CONFIGS:
   d=ROOT/regime/variant/f'seed_{seed}';d.mkdir(parents=True,exist_ok=True);write(d/'config.json',run_config(regime,variant,seed));(d/'seed.txt').write_text(str(seed)+'\n');shutil.copy2(P/'environment.txt',d/'environment.txt')
   for split in SPECS:(d/f'{split}_manifest_hash.txt').write_text(sha(P/f'{split}_manifest.json')+'\n')
with (ROOT/'logs/pipeline.log').open('a') as log:
 p=subprocess.Popen([sys.executable,str(CODE/'orchestrate.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONHASHSEED':'42'})
write(ROOT/'pipeline_pid.json',{'pid':p.pid,'python':sys.executable,'protocol_sha256':sha(P/'LOCK.json')});print('SEALED and STARTED',p.pid,sha(P/'LOCK.json'),flush=True)
