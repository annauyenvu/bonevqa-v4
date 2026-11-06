"""Read-only scientific plot generation and fail-closed final artifact sealing."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=read(O/'analysis/results.json');dest=O/'analysis';assert not (dest/'comparison.png').exists()
    fig,axs=plt.subplots(2,2,figsize=(13,9));names=list(data['pooled']);labels=['B','Local','B + head','Local + head']
    fields=[('AP','Raw fracture AP'),('BA','BA at calibration specificity target 0.90'),('sensitivity','Sensitivity at calibration operating point'),('specificity','Specificity at calibration operating point')]
    for ax,(field,title) in zip(axs.flat,fields):
        vals=[data['pooled'][n]['raw' if field=='AP' else 'spec90'][field] for n in names]
        ax.bar(labels,vals,color=['#596a7c','#237c94','#8596a8','#6caab9']);ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(vals):ax.text(i,v+.015,f'{v:.4f}',ha='center',fontsize=10)
    fig.suptitle('Fixed five-fold B versus local tokens without auxiliary loss\nDevelopment-informed data; seed 42; no patient-independent claim',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(dest/'comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for arm,color in [('B','#596a7c'),('local_real_noaux','#237c94')]:
        for ax,field,mode in [(axs[0],'AP','raw'),(axs[1],'BA','spec90')]:
            vals=[data['folds'][str(f)]['original_'+arm][mode][field] for f in range(5)]
            ax.plot(range(5),vals,'o-',label=arm,color=color);ax.set_xticks(range(5));ax.set_xlabel('Outer fold');ax.grid(alpha=.2)
    axs[0].set_title('Raw AP by fold');axs[1].set_title('BA at calibration operating point');axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(dest/'fold_comparison.png',dpi=180);plt.close(fig)
    with (dest/'plot_receipt.json').open('x') as f:json.dump({'source_results_sha256':sha(dest/'results.json'),'figures':{str(p):sha(p) for p in (dest/'comparison.png',dest/'fold_comparison.png')},'actual_visual_review_pending':True},f,indent=2)
def seal():
    assert not (O/'closure_seal.json').exists()
    assert not any((O/p).exists() for p in ('failure.json','postprocess_failure.json','wall_guard/deadline_exceeded.json'))
    guard=read(O/'wall_guard/completed.json');assert guard.get('execution_and_analysis_within_true_wall') is True
    for release in ('preparation.json','head_analysis_release.json'):
        for p,h in read(O/release)['dependencies'].items():assert sha(Path(p))==h,p
    assert read(O/'postprocess/completed.json')['analysis_generated']
    report=read(O/'analysis/results.json');assert report['audit']['fivefold_coverage_images']==4013 and report['audit']['QA']==17998
    for p,h in report['audit']['dependencies'].items():assert sha(Path(p))==h,p
    for fold in map(str,range(5)):
        for arm in ('B','local_real_noaux')+(('local_placebo_noaux',) if fold=='0' else ()):
            d=O/f'fold_{fold}'/arm;done=read(d/'completed.json');replay=read(d/'replay.json');assert replay['passed'] and done['frozen_unchanged']
            assert sha(d/'LAST.pt')==done['checkpoint_sha256']==replay['checkpoint_sha256']
            if arm!='local_placebo_noaux':
                h=d/'head_refit';fit=read(h/'fit.json');assert fit['updates']==10*math.ceil(fit['TRAIN_images']/64)
                assert fit['all_model_parameters_unchanged_during_fit'] and sha(h/'HEAD_LAST.pt')==read(h/'replay.json')['HEAD_sha256']
    review=read(O/'analysis/figure_visual_review.json');assert review['visually_reviewed'] is True
    figures=read(O/'analysis/plot_receipt.json')['figures']
    assert review['figure_sha256']==figures
    for p,h in figures.items():assert sha(Path(p))==h
    final=O/'FINAL_INTERPRETATION.md';assert final.exists() and len(final.read_text())>1500
    files=[p for p in O.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.png') and not p.name.endswith('.tmp')]
    with (O/'closure_seal.json').open('x') as f:json.dump({'complete_bounded_fivefold_refinement':True,'promotion_rule_passed':report['promotion_rule_passed'],'clinical_validation_claim':False,'artifacts':{str(p):sha(p) for p in files}},f,indent=2)
    print('Study closed; no additional training authorized by closure')
if __name__=='__main__':globals()[sys.argv[1]]()
