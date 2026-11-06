"""Independent saved-token/target audit, fixed component bootstrap and plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import LlamaTokenizer
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.prepare_lapa_pilot_v1 import sha
from src.utils import normalize_answer

OUT=ROOT/'reports/favp_bounded_quality_v1'


def main():
    assert (OUT/'execution_completed.json').exists()
    assert not (OUT/'independent_audit.json').exists()
    assert json.loads((OUT/'replay.json').read_text())['exact_tokens_answers_metrics']
    selection=json.loads((ROOT/'reports/favp_pilot_inputs_v1/selection_and_schedules.json').read_text())
    source_records=json.loads((ROOT/'reports/step4_localization_v3_1/model_val.json').read_text())
    tokenizer=LlamaTokenizer.from_pretrained(ROOT/'references/native_assets_favp_v1/vicuna-7b',use_fast=False,local_files_only=True)
    sources={};audits={};datasets={}
    for stage,updates in [('PRE',0),('epoch1',32),('epoch2',64)]:
        path=OUT/f'{stage}.json';data=json.loads(path.read_text());assert data['updates']==updates
        rows=data['rows'];assert len(rows)==8
        correct=[]
        for row,selected in zip(rows,selection['subset']['model_val']):
            target=source_records[selected['record_index']]
            assert row['image_id']==target['image_id']==selected['image_id']
            assert row['question']==target['question'] and row['target']==normalize_answer(target['answer'])
            assert len(row['tokens'])<=20
            decoded=tokenizer.decode(row['tokens'],skip_special_tokens=True)
            answer=decoded.split('###')[0].replace('<s>','').split('Assistant: ')[-1].strip()
            assert decoded==row['raw_decoded'] and answer==row['answer'] and normalize_answer(answer)==row['normalized_answer']
            correct.append(row['normalized_answer']==row['target'])
        recount={'records':8,'accuracy':sum(correct)/8,'unsupported':sum(r['normalized_answer'] not in ('yes','no') for r in rows),
            'empty':sum(not r['normalized_answer'] for r in rows),'truncated':sum(r['truncated'] for r in rows),
            'target_yes':sum(r['target']=='yes' for r in rows),'target_no':sum(r['target']=='no' for r in rows)}
        assert recount==data['metrics']
        datasets[stage]=data;audits[stage]=recount;sources[str(path)]=sha(path)
    release=json.loads((OUT/'execution_release.json').read_text())
    assert all(sha(Path(p))==h for p,h in release['dependencies'].items())
    (OUT/'independent_audit.json').write_text(json.dumps({'all_three_saved_output_audits_passed':True,'stages':audits,
        'sources':sources,'execution_dependencies_unchanged':len(release['dependencies'])},indent=2))
    lookup_path=ROOT/'reports/step4_localization_v3_1/component_lookup.json'
    lookup=json.loads(lookup_path.read_text());rows=datasets['PRE']['rows']
    ids=[r['image_id'] for r in rows];groups=sorted({lookup[i] for i in ids})
    blocks=[np.array([j for j,i in enumerate(ids) if lookup[i]==g]) for g in groups]
    initial=np.array([r['normalized_answer']==r['target'] for r in rows],dtype=float)
    final=np.array([r['normalized_answer']==r['target'] for r in datasets['epoch2']['rows']],dtype=float)
    rng=np.random.default_rng(20261103);draws=[]
    for _ in range(2000):
        ii=np.concatenate([blocks[j] for j in rng.integers(0,len(groups),len(groups))])
        draws.append(float((final[ii]-initial[ii]).mean()))
    bootstrap={'seed':20261103,'draws':2000,'components':len(groups),'records':8,
        'LAST_minus_PRE_accuracy':float((final-initial).mean()),'CI95':np.quantile(draws,[.025,.975]).tolist(),
        'scope':'exploratory small engineering pilot; single seed/development-informed validation; components not patients; no clinical efficacy inference',
        'component_lookup_sha256':sha(lookup_path),'sources':sources}
    (OUT/'paired_component_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot([0,1,2],[audits[s]['accuracy'] for s in ['PRE','epoch1','epoch2']],marker='o')
    axes[0].set(xlabel='Epoch (0 = PRE)',ylabel='Strict generation accuracy (8 examples)',ylim=(-.03,1.03),xticks=[0,1,2])
    axes[1].plot([1,2],[datasets[s]['mean_train_loss'] for s in ['epoch1','epoch2']],marker='o')
    axes[1].set(xlabel='Epoch',ylabel='Mean TRAIN causal loss',xticks=[1,2])
    fig.suptitle('Adapted native-asset FAVP engineering pilot — not original reproduction')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# Bounded adapted FAVP pilot results','',
        'One seed,32TRAINcanonical examples/two epochs/64updates;8development-informed validation images. New projection, unavailable source stage handoff, CPU numerical/runtime adaptations. Native SAM accepted no proposals in this TRAIN cohort or selected validation subset; original-only fallback, no local-branch benefit claim.','',
        '| Stage | Strict accuracy | Non-yes/no | Truncated |','|---|---:|---:|---:|']
    for stage in ['PRE','epoch1','epoch2']:
        r=audits[stage];lines.append(f"| {stage} | {r['accuracy']:.3f} | {r['unsupported']} | {r['truncated']} |")
    lines+=['',f"LAST−PRE accuracy {bootstrap['LAST_minus_PRE_accuracy']:.3f}; exploratory paired component95%CI {bootstrap['CI95']}, {len(groups)}components/2000draws. Not seed uncertainty or patient-independent evidence.",'',
        'All saved-token decoding/target/metric audits and complete selected fresh-process replay passed. No checkpoint/threshold/seed search or automatic promotion. Actual figure review, final interpretation and closure remain pending.']
    (OUT/'PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
