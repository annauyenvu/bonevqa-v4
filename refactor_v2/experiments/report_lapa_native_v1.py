"""Render completed fixed-budget comparison; visual review remains separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'reports/lapa_native_quality_v1'
ARMS=['carrier','gm_lf','pf_author','pf_residual','pf_residual_rewired']


def main():
    assert (OUT/'execution_completed.json').exists()
    uncertainty=json.loads((OUT/'paired_component_bootstrap.json').read_text())
    assert not (OUT/'REFERENCE_PILOT_RESULTS.md').exists()
    fig,axes=plt.subplots(1,3,figsize=(14,4));lines=[];timings={}
    for arm in ARMS:
        audit=json.loads((OUT/arm/'independent_metric_audit.json').read_text());assert audit['saved_logits_recount_passed']
        assert json.loads((OUT/arm/'replay.json').read_text())['exact']
        epochs=[json.loads((OUT/arm/f'epoch{i}.json').read_text()) for i in range(1,5)]
        for axis,key in zip(axes,['AP','accuracy','BA']):
            axis.plot(range(1,5),[d['metrics'][key] for d in epochs],marker='o',label=arm)
            axis.set_title(key);axis.set_xlabel('Epoch');axis.set_xticks(range(1,5));axis.grid(alpha=.2)
        m=epochs[-1]['metrics'];timings[arm]=sum(d['train_seconds']+d['eval_seconds'] for d in epochs)
        lines.append(f"| {arm} | {m['AP']:.6f} | {m['accuracy']:.6f} | {m['sensitivity']:.6f} | {m['specificity']:.6f} | {m['BA']:.6f} |")
    axes[0].legend(fontsize=7);fig.suptitle('Native-pretrained frozen-encoder LaPA pilot: seed42, LAST4')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
    text=['# Native-pretrained frozen-encoder LaPA pilot','',
          'Five matched arms, seed42, four epochs/292 updates each; full1432-output fresh replay and independent recount. Actual pretrained M3AE image/text/MM, frozen image/text encoders, new prompt/head parameters and reconstructed SLAKE409. Modality-key mapping and source optimizer runtime adaptations disclosed. Not full paper fine-tuning or original graph577 reproduction.',
          '', '| Arm | Fracture AP | Answer accuracy | Sensitivity | Specificity | BA |',
          '| --- | ---: | ---: | ---: | ---: | ---: |',*lines,'',
          '## Prespecified paired contrasts','']
    for pair,values in uncertainty['comparisons'].items():
        for metric,d in values.items():text.append(f"- {pair}, {metric}: delta {d['difference']:.6f}, component 95% interval [{d['CI95'][0]:.6f}, {d['CI95'][1]:.6f}], {d['valid_draws']} valid draws.")
    text+=['','Intervals are exploratory and unadjusted for multiplicity, conditional on one training seed. They do not measure seed uncertainty. Components are not verified patients; validation has informed development.',
           '', 'The answer endpoint is a 41-class TRAIN-vocabulary classifier. Unsupported answers are counted incorrect. Do not rank these accuracy/AP values against historical generative Native B as a matched comparison. The encoder-freezing policy, new prompt/head initialization, restricted endpoint and reconstructed graph limit claims about the original method.',
           '', '## Recorded train plus per-epoch validation seconds','',json.dumps(timings,indent=2),
           '', 'Full replay, initialization, hashing and analysis add overhead. No threshold/epoch selection or extra runs are authorized by this report. Figure visual review and final dependency verification remain required before closure.']
    with (OUT/'REFERENCE_PILOT_RESULTS.md').open('x') as f:f.write('\n'.join(text)+'\n')


if __name__=='__main__':main()


