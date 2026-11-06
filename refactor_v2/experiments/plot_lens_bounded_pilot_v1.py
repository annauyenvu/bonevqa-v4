"""Plot plain saved audit/loss JSON using the unchanged historical plotting runtime."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/lens_bounded_quality_v1'


def main():
    audit=json.loads((OUT/'independent_audit.json').read_text())
    assert audit['all_saved_token_target_metric_audits_passed']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    labels=['PRE','Align1','QA1','QA2 LAST']
    for arm in ['ce_only','lens_losses']:
        values=audit['stages'][arm]
        stages=['PRE','stage1','qa_epoch1','LAST']
        axes[0,0].plot(range(4),[values[s]['strict_accuracy'] for s in stages],marker='o',label=arm)
        axes[0,1].plot(range(4),[values[s]['leading_binary_accuracy'] for s in stages],marker='o',label=arm)
        complete=json.loads((OUT/arm/'completed.json').read_text())
        axes[1,0].plot([1,2,3],[r['mean_CE'] for r in complete['epochs']],marker='o',label=arm)
        axes[1,1].plot([1,2,3],[r['mean_contrastive'] for r in complete['epochs']],marker='o',label=arm)
    baseline=audit['TRAIN_only_majority_constant']['strict_accuracy']
    for axis in axes[0]:
        axis.axhline(baseline,color='gray',linestyle='--',label='TRAIN majority constant')
        axis.set(ylim=(-.03,1.03),xticks=range(4),xticklabels=labels);axis.legend(fontsize=8)
    axes[0,0].set_title('Primary strict full-answer EM (8 reused cases)')
    axes[0,1].set_title('Secondary anchored leading yes/no accuracy')
    axes[1,0].set_title('TRAIN causal CE (region then QA targets)')
    axes[1,1].set_title('TRAIN contrastive term (DCL then InfoNCE)')
    for axis in axes[1]:axis.set(xticks=[1,2,3],xticklabels=labels[1:]);axis.legend(fontsize=8)
    fig.suptitle('Adapted native-asset Lens — one seed, not independent clinical evidence')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
