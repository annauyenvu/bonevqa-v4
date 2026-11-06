"""Scientific figures and a sealed supplementary interpretation of completed V."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/vnext_local_tokens_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert read(O/'analysis_closure.json')['V_analysis_complete']
    for p,h in read(O/'analysis_closure.json')['artifacts'].items():assert sha(Path(p))==h
    d=read(O/'independent_analysis.json');assert not (O/'V_learning_curves.png').exists()
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for arm,curves in d['curves'].items():axes[0].plot(range(1,5),[r['AP'] for r in curves],marker='o',label=arm)
    axes[0].set(title='Canonical fracture AP learning curves',xlabel='Epoch (73 updates)',ylabel='Model-validation AP');axes[0].set_xticks(range(1,5));axes[0].grid(alpha=.2);axes[0].legend(fontsize=8)
    names=list(d['comparisons']);values=[d['comparisons'][a]['real_minus_control_AP'] for a in names];bounds=[d['comparisons'][a]['CI95'] for a in names]
    axes[1].errorbar(values,range(len(names)),xerr=[[x-b[0] for x,b in zip(values,bounds)],[b[1]-x for x,b in zip(values,bounds)]],fmt='o',capsize=4)
    axes[1].axvline(0,color='grey',linewidth=1);axes[1].set_yticks(range(len(names)),['real_aux minus '+a for a in names]);axes[1].set(title='Paired component bootstrap: 2,000 draws',xlabel='AP difference, exploratory 95% interval');axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('V local tokens: seed42, inspected development cohort, fixed LAST4')
    fig.savefig(O/'V_learning_curves.png',dpi=180);plt.close(fig)
    text='''# V pilot interpretation and technical limits

All five arms completed four epochs /292updates and exact full1432record fresh-process replays. The primary provisional real_aux promotion rule failed: AP0.694109 versus placebo_aux0.708043. Its paired AP difference is−0.013935, exploratory component95%CI[−0.077238,+0.044088]. Real_aux minus B is+0.025420, CI[−0.036780,+0.078863]. Real_aux exceeds grid_aux by+0.073156, CI[+0.010574,+0.141255], but this does not overcome the matched placebo comparison.

The auxiliary ablation is informative: real_noaux AP0.732725 versus real_aux0.694109. Real_aux minus real_noaux is−0.038616, CI[−0.078240,−0.000557]. These fixed-model intervals are exploratory and not adjusted for multiple comparisons. This single-seed result supports investigating loss interference, not claiming a generally harmful auxiliary objective. No extra epochs, seeds, thresholds or combinations were selected.

Real_noaux is higher than B in point AP but lacks same-noaux placebo/grid controls, so this matrix does not establish a semantic localization advantage for that variant. At raw0.5 it rescues1/8Bfalse-negatives and loses4Btrue-positives; different calibration prevents interpreting these counts as a same-specificity causal effect. Real_aux rescues1/8 and loses5.

Recorded gradients, parameter updates and disposable intervention tests passed for expected paths. The local subclass computes original fusion before computing local fusion, adding compute/dropout draws relative to B. Local spatial controls share this path. For real_noaux, auxiliary-only gradients are zero; any auxiliary-only AdamW decay changes are not learning evidence. Fixed-grid matches detector proposal count but not crop geometry and is not detector-free. The detector's broader TRAIN exposure remains part of the resource ledger.

The frozen pilot is complete. The authorized next priority is L original-encoder spatial supervision, then P diagnostics. Current V results do not authorize a combination or production claim. Existing model-validation data are development-informed; components are not verified patients.
'''
    with (O/'V_INTERPRETATION.md').open('x') as f:f.write(text)
    names=('V_learning_curves.png','V_INTERPRETATION.md','error_modes_final.json','gradient_route_audit_final.json','analysis_closure.json')
    with (O/'supplementary_closure.json').open('x') as f:json.dump({'artifacts':{str(O/n):sha(O/n) for n in names},'core_closure_unchanged':True,'figure_visual_review_pending':True,'no_new_training':True,'source_sha256':sha(Path(__file__))},f,indent=2)
    print('V supplementary figures and interpretation saved.')
if __name__=='__main__':main()
