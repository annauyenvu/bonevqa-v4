from pathlib import Path
import os,sys,json,hashlib,random,platform,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1];LEGACY=ROOT.parent/'.evaluation_runtime';CODE=ROOT/'code';P=ROOT/'protocol'
os.environ['HF_HOME']=str(LEGACY/'hf_cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['BONEVQA_DATA']=str(LEGACY/'data');os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1';os.environ['BONEVQA_STRICT_SAM']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['MPLCONFIGDIR']=str(ROOT/'mpl')
os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
sys.path[:0]=[str(CODE),str(LEGACY/'deps')];os.chdir(ROOT)
SEEDS=[42,123,456,789,2026];REGIMES=['auto','detector','clinician'];CONFIGS=['vlp','none'];SPECS={'train':(6000,1342,3340),'val':(300,69,166),'test':(2773,612,1546)}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def jhash(x):return hashlib.sha256(canonical(x)).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)
def status(stage,state='running',**kw):write(ROOT/'STATUS.json',{'stage':stage,'state':state,'updated':datetime.datetime.now().isoformat(),**kw})
def setup(seed=42):
 import numpy as np,torch
 if not torch.backends.mps.is_available():raise RuntimeError('Locked MPS environment unavailable; refusing CPU/CUDA substitution')
 if os.environ.get('PYTHONHASHSEED')!=str(seed):raise RuntimeError('PYTHONHASHSEED must be set before process start')
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.mps.manual_seed(seed);torch.set_num_threads(4)
 torch.use_deterministic_algorithms(True,warn_only=False)
 return torch.device('mps')
def environment():
 import importlib.metadata,torch
 return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'hardware':subprocess.check_output(['system_profiler','SPHardwareDataType'],text=True),'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'precision':'float32 all model weights and LLM in train and eval; no AMP','cpu_fallback':True,'strict_sam':True,'deterministic_algorithms':True,'num_threads':4,'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','transformers','peft','accelerate','timm','numpy','Pillow','huggingface-hub']},'open_clip_source':str(LEGACY/'deps/open_clip')}
def manifests():
 ans={s:read(P/f'{s}_manifest.json') for s in SPECS}
 for s,rs in ans.items():
  actual=(len(rs),len({r['image_id'] for r in rs}),sum(r['answer_type']=='CLOSED' for r in rs));assert actual==SPECS[s],(s,actual)
  assert len({(r['image_id'],r['question']) for r in rs})==len(rs)
 return ans

def verify_lock():
 lock=read(P/'LOCK.json')
 for rel,digest in lock['files'].items():
  p=ROOT/rel if not Path(rel).is_absolute() else Path(rel)
  if sha(p)!=digest:raise RuntimeError('Protocol input/source hash changed: '+str(p))
 if environment()!=read(P/'environment.json'):raise RuntimeError('Software/hardware environment differs from lock')
 return lock
