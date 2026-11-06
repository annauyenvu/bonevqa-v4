from common import *
import torch
from src.utils import load_config
setup(42)
if (P/'LOCK.json').exists():raise RuntimeError('Protocol already locked; no overwrite')
write(P/'environment.json',environment());(P/'environment.txt').write_text(json.dumps(environment(),indent=2))
source=LEGACY/'data/fracatlas'
for s,n in [('train',6000),('val',300),('test',2773)]:write(P/f'{s}_manifest.json',read(source/f'vqa_{s}.json')[:n])
ms=manifests()
for a,b in [('train','val'),('train','test'),('val','test')]:assert not {r['image_id'] for r in ms[a]}&{r['image_id'] for r in ms[b]}
base=load_config(str(LEGACY/'configs/ablation_fracatlas_sach.yaml'))
ck=LEGACY/'checkpoints/stage2_pmcvqa_full/best.pt';historical=LEGACY/'checkpoints/abl3_du3/best.pt'
payload=torch.load(historical,map_location='cpu',weights_only=False)
base['model']=payload['config'];base['model']['llm_dtype']='float32';base['model']['sam']['sam_med2d_ckpt']=None
base['model']['sam']['cache_dir']=str(ROOT/'unused_sam_cache');base['model']['sam']['enabled']=True
base['seed']=None;base['run_name']='locked_exp4';base['init_from']=str(ck);base['checkpoint_dir']=str(ROOT);base['data']['loader_kwargs']={};base['data']['root']=str(LEGACY/'data');base['train']['amp']=False;base['data']['num_workers']=0
write(P/'base_config.json',base);write(P/'answer_vocab.json',payload['answer_vocab']);del payload
print('Hashing image cohort and model artifacts',flush=True)
images={}
for rs in ms.values():
 for r in rs:images.setdefault(r['image_id'],r)
image_manifest=[]
for i,(iid,r) in enumerate(sorted(images.items())):
 path=LEGACY/'data/fracatlas/FracAtlas'/r['image'];image_manifest.append({'image_id':iid,'path':str(path),'sha256':sha(path),'width':r['width'],'height':r['height']})
 if i%250==0:print('image_hash',i,len(images),flush=True)
write(P/'image_content_manifest.json',image_manifest)
weights={}
for model in (LEGACY/'hf_cache/hub').glob('models--*'):
 for f in (model/'snapshots').glob('*/*'):
  if f.is_file() and f.suffix in ['.json','.txt','.bin','.safetensors']:weights[str(f)]=sha(f)
write(P/'pretrained_files.json',weights)
write(P/'protocol.json',{'seeds':SEEDS,'primary_matrix':[{'regime':r,'config':c,'seed':s} for s in SEEDS for r in REGIMES for c in CONFIGS],'expected_counts':SPECS,'primary_runs':30,'sanity_extra_runs':1,'sanity_rule':'First auto/vlp/seed42 run is prespecified primary; independent repeat excluded. Never select between repeats by score. Full matrix gated by evaluator identity and training sanity report.','primary_endpoint':'CLOSED exact accuracy 1546 QA, 612 image clusters','primary_contrasts':['vlp-none within '+r for r in REGIMES],'secondary_contrasts':['(clinician vlp-none)-(auto vlp-none)','(clinician vlp-none)-(detector vlp-none)'],'bootstrap':{'replicates':10000,'seed':20260921,'algorithm':'resample 5 seeds, then independently resample 612 images within each sampled seed slot; identical image and seed draws across paired configs and regimes'},'checkpoint_selection':'validation overall_acc, strictly better selects new best; ties keep earliest epoch','failure_rules':['crash','nonfinite loss or gradients','unexpected ROI processing error or absent/corrupt mask file','manifest/environment/config mismatch','checkpoint corruption','evaluator failure or wrong counts'],'rerun_policy':'No score-based restart. Technical failures quarantined and recorded; rerun same seed/recipe only. Pipeline stops for diagnosis.','historical':'supplementary only; never included in primary seed summaries','roi':'All cohort masks generated once in locked environment and immutable audited bank; same bank used by every seed, empty post-filter result is valid, processing exception is invalid. None bypasses use of masks, but audit still verifies regime bank.','precision':'float32, no AMP, MPS, fixed CPU fallback','initialization':'Every new run independently starts optimizer from same stage2 state; not resume existing stage3','evaluator':'fixed batch size4, original manifest order, greedy generation, no sampling, float32, deterministic algorithms strict','lens_rng_fix':'random.Random() replaced with global seeded random; separate seeded DataLoader generator','optional_interventions':'Deferred until primary matrix and analysis complete; not part of primary stopping rule'})
files={str(ck):sha(ck),str(LEGACY/'data/fracatlas/box_bo_phat_hien.json'):sha(LEGACY/'data/fracatlas/box_bo_phat_hien.json')}
for f in P.glob('*'):
 if f.is_file() and f.name!='LOCK.json':files[str(f.relative_to(ROOT))]=sha(f)
# Code snapshot is sealed separately immediately before workers start.
write(P/'inputs_lock.json',{'files':files,'pretrained':weights})
print('PREPARED',SPECS,'images',len(images),'stage2',files[str(ck)],flush=True)
