from common import *
import numpy as np,csv
from engine import csvwrite,check_predictions,normalize_yes_no

def hierarchical(nums,den,reps=10000,seed=20260921):
 """nums[seed,regime,config,image], den[image]; paired draws across all cells."""
 rng=np.random.default_rng(seed);nseed,nreg,ncfg,nimage=nums.shape;boots=np.empty((reps,nreg,ncfg))
 for start in range(0,reps,100):
  batch=min(100,reps-start);seeds=rng.integers(nseed,size=(batch,nseed));acc=np.zeros((batch,nreg,ncfg))
  for slot in range(nseed):
   images=rng.integers(nimage,size=(batch,nimage));counts=den[images].sum(1);assert np.all(counts>0)
   for r in range(nreg):
    for c in range(ncfg):acc[:,r,c]+=np.take_along_axis(nums[seeds[:,slot],r,c,:],images,axis=1).sum(1)/counts
  boots[start:start+batch]=100*acc/nseed
 return boots

def main():
 setup(42);verify_lock();dest=ROOT/'analysis';dest.mkdir(exist_ok=True);test=read(P/'test_manifest.json');ids=sorted({r['image_id'] for r in test});ix={x:i for i,x in enumerate(ids)};den=np.zeros(len(ids));nums=np.zeros((5,3,2,len(ids)));rows=[];full=[]
 for r in test:
  if r['answer_type']=='CLOSED':den[ix[r['image_id']]]+=1
 for si,s in enumerate(SEEDS):
  row={'seed':s}
  for ri,r in enumerate(REGIMES):
   for ci,c in enumerate(CONFIGS):
    run=ROOT/r/c/f'seed_{s}';qc=read(run/'QC.json');assert qc['valid'] and qc['seed']==s and qc['regime']==r and qc['config']==c and qc['roi_failed']==0
    for split in SPECS:assert qc['manifest_hashes'][split]==sha(P/f'{split}_manifest.json')
    assert sha(run/'checkpoint_best.pt')==qc['checkpoint_sha256']
    rr=read(run/'test_predictions.json');check_predictions(rr,'test');m=read(run/'metrics.json');full.append({'seed':s,'regime':r,'config':c,**{k:v for k,v in m.items() if isinstance(v,(int,float))}})
    for x in rr:
     if x['answer_type']=='CLOSED':nums[si,ri,ci,ix[x['image_id']]]+=normalize_yes_no(x['pred'])==normalize_yes_no(x['gt'])
    row[r+'_'+c]=100*nums[si,ri,ci].sum()/den.sum()
   row[r+'_delta_pp']=row[r+'_vlp']-row[r+'_none']
  rows.append(row)
 point=100*nums.sum(-1)/den.sum();boots=hierarchical(nums,den);delta=point[:,:,0]-point[:,:,1];bdelta=boots[:,:,0]-boots[:,:,1]
 tableA=[];tableB=[];secondary=[]
 for r,reg in enumerate(REGIMES):
  for c,cfg in enumerate(CONFIGS):
   x=point[:,r,c];tableA.append({'regime':reg,'config':cfg,'mean_closed_pct':float(x.mean()),'sd_seed_pp':float(x.std(ddof=1)),'min_pct':float(x.min()),'max_pct':float(x.max()),'n_seeds':5})
  x=delta[:,r];lo,hi=np.quantile(bdelta[:,r],[.025,.975]);tableB.append({'regime':reg,'mean_delta_pp':float(x.mean()),'sd_seed_delta_pp':float(x.std(ddof=1)),'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),'positive_seeds':int((x>0).sum()),'negative_seeds':int((x<0).sum()),'zero_seeds':int((x==0).sum()),'n_seeds':5})
 for r in [0,1]:
  x=delta[:,2]-delta[:,r];b=bdelta[:,2]-bdelta[:,r];lo,hi=np.quantile(b,[.025,.975]);secondary.append({'contrast':'clinician_minus_'+REGIMES[r],'mean_difference_of_differences_pp':float(x.mean()),'sd_seed_pp':float(x.std(ddof=1)),'ci95_low_pp':float(lo),'ci95_high_pp':float(hi),'positive_seeds':int((x>0).sum()),'individual_seed_values':x.tolist()})
 for name,rr in [('seed_matrix',rows),('per_seed_metrics',full),('EXP4_A_training_seed_stability',tableA),('EXP4_B_paired_VLP_effect',tableB),('secondary_difference_of_differences',secondary)]:csvwrite(dest/(name+'.csv'),rr);write(dest/(name+'.json'),rr)
 np.savez_compressed(dest/'bootstrap_replicates.npz',cell_accuracy_pct=boots,paired_delta_pp=bdelta)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,3,figsize=(11,4),sharey=True)
 for r,ax in enumerate(axs):
  for s,seed in enumerate(SEEDS):ax.plot([0,1],[point[s,r,1],point[s,r,0]],marker='o',label=str(seed))
  ax.set_xticks([0,1],['None','VLP']);ax.set_title(REGIMES[r]);ax.grid(alpha=.2)
 axs[0].set_ylabel('Closed accuracy (%)');axs[-1].legend(title='Training seed');fig.tight_layout();fig.savefig(dest/'paired_seed_plot.png',dpi=250);fig.savefig(dest/'paired_seed_plot.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4))
 for si,s in enumerate(SEEDS):ax.plot(range(3),delta[si],marker='o',label=str(s))
 ax.axhline(0,color='black',lw=1);ax.set_xticks(range(3),REGIMES);ax.set_ylabel('VLP − None (percentage points)');ax.legend(title='Seed');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/'seed_delta_plot.png',dpi=250);plt.close(fig)
 lines=['# EXP4 results','','Thirty fresh primary runs; identical locked environment. Historical checkpoints excluded.','', '| Regime | Config | Mean closed % | SD seed pp | Min–max |','|---|---|---:|---:|---|']
 for r in tableA:lines.append(f"| {r['regime']} | {r['config']} | {r['mean_closed_pct']:.2f} | {r['sd_seed_pp']:.2f} | {r['min_pct']:.2f}–{r['max_pct']:.2f} |")
 lines+=['','| Regime | Mean Δ pp | SD Δ pp | Hierarchical 95% CI | Positive seeds |','|---|---:|---:|---|---|']
 for r in tableB:lines.append(f"| {r['regime']} | {r['mean_delta_pp']:+.2f} | {r['sd_seed_delta_pp']:.2f} | [{r['ci95_low_pp']:.2f}, {r['ci95_high_pp']:.2f}] | {r['positive_seeds']}/5 |")
 lines+=['','10,000 hierarchical seed × image-cluster bootstrap replicates; seed 20260921. Shared seed/image draws preserve matching across cells and secondary contrasts. These are marginal intervals, not multiplicity-adjusted. No hyperparameters or restarts were selected using test scores.','']
 (dest/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
