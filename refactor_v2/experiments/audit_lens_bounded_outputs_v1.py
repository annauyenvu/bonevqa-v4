"""Independent saved-token audit and fixed component uncertainty, no model inference."""
import json
from pathlib import Path
import re
import numpy as np
from transformers import AutoTokenizer
from experiments.lens_native_common_v1 import ROOT,sha

QUALITY=ROOT/'reports/lens_bounded_quality_v1'


def normalized(text):return re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()


def recount(rows):
    target=[normalized(r['target']) for r in rows]
    predictions=[normalized(r['text']) for r in rows]
    parsed=[]
    for row in rows:
        matched=re.match(r'^\s*(yes|no)\b',row['text'],re.I)
        parsed.append(matched.group(1).lower() if matched else None)
    tp=sum(t=='yes' and p=='yes' for t,p in zip(target,parsed));tn=sum(t=='no' and p=='no' for t,p in zip(target,parsed))
    fn=sum(t=='yes' and p!='yes' for t,p in zip(target,parsed));fp=sum(t=='no' and p!='no' for t,p in zip(target,parsed))
    sensitivity=tp/(tp+fn) if tp+fn else None;specificity=tn/(tn+fp) if tn+fp else None
    return {'records':len(rows),'strict_accuracy':sum(t==p for t,p in zip(target,predictions))/len(rows),
        'leading_binary_accuracy':sum(t==p for t,p in zip(target,parsed))/len(rows),'secondary_unsupported':parsed.count(None),
        'truncated':sum(r['truncated'] for r in rows),'empty':sum(not r['text'] for r in rows),'tp':tp,'tn':tn,
        'fp_or_unsupported_negative':fp,'fn_or_unsupported_positive':fn,'sensitivity':sensitivity,'specificity':specificity,
        'balanced_accuracy':(sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None}


def main():
    assert (QUALITY/'execution_completed.json').exists()
    assert not (QUALITY/'independent_audit.json').exists()
    release=json.loads((QUALITY/'execution_release.json').read_text())
    assert all(sha(Path(p))==h for p,h in release['dependencies'].items())
    prep=ROOT/'reports/lens_bounded_preparation_v1'
    config=json.loads((prep/'config.json').read_text())
    targets=json.loads((ROOT/'reports/step4_localization_v3_1/model_val.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'references/native_assets_lens_v1/SmolVLM-256M-Instruct',local_files_only=True)
    datasets={};audit={};sources={}
    for arm in ['ce_only','lens_losses']:
        directory=QUALITY/arm
        assert json.loads((directory/'replay.json').read_text())['exact_tokens_text_targets_metrics']
        completed=json.loads((directory/'completed.json').read_text())
        assert completed['frozen_invariance'] and completed['updates']==sum(config['updates_per_stage'].values())
        assert completed['changed_tensor_count']==492
        assert [r['microbatches'] for r in completed['epochs']]==[14,14,14]
        assert sha(directory/'LAST.pt')==completed['LAST_sha256']
        datasets[arm]={};audit[arm]={}
        for stage in ['PRE','stage1','qa_epoch1','LAST']:
            path=directory/(stage+'.json');data=json.loads(path.read_text());rows=data['rows'];assert len(rows)==8
            for saved,selected in zip(rows,config['model_validation']):
                target=targets[selected['record_index']]
                assert saved['image_id']==selected['image_id']==target['image_id']
                assert saved['question']==target['question'] and saved['target']==target['answer']
                assert 0<len(saved['tokens'])<=32
                assert tokenizer.decode(saved['tokens'],skip_special_tokens=True).strip()==saved['text']
                assert saved['truncated']==(len(saved['tokens'])==32 and saved['tokens'][-1]!=tokenizer.eos_token_id)
                image=ROOT.parent/'.evaluation_runtime/data/fracatlas/FracAtlas'/target['image']
                assert sha(image)==saved['image_sha256']
            values=recount(rows);assert values==data['metrics'],(arm,stage,values,data['metrics'])
            datasets[arm][stage]=data;audit[arm][stage]=values;sources[str(path)]=sha(path)
        for epoch in completed['epochs']:
            terms=epoch['micro_terms'];assert all(np.isfinite(r['total']) and np.isfinite(r['CE']) and np.isfinite(r['contrastive']) for r in terms)
            assert sum(r['total'] for r in terms)/len(terms)==epoch['mean_loss']
    assert datasets['ce_only']['PRE']['rows']==datasets['lens_losses']['PRE']['rows']
    selected=json.loads((ROOT/'reports/lens_bounded_inputs_v1/receipt.json').read_text())
    training=json.loads((ROOT/'reports/step4_localization_v3_1/train.json').read_text())
    ids=set(selected['selected_training_ids']);counts={'yes':0,'no':0}
    for case in selected['cases']:
        if case['image']['image_id'] in ids:
            answer=normalized(training[case['candidate']['record_index']]['answer']);assert answer in counts
            counts[answer]+=1
    constant=max(['no','yes'],key=lambda k:counts[k]) # fixed tie order, no validation data used to choose
    rows=datasets['ce_only']['PRE']['rows'];constant_score=np.array([normalized(r['target'])==constant for r in rows],float)
    (QUALITY/'independent_audit.json').write_text(json.dumps({'all_saved_token_target_metric_audits_passed':True,
        'stages':audit,'sources':sources,'execution_dependencies_unchanged':len(release['dependencies']),
        'TRAIN_only_majority_constant':{'answer':constant,'counts':counts,'strict_accuracy':float(constant_score.mean())},
        'script_sha256':sha(Path(__file__))},indent=2))
    lookup_path=ROOT/'reports/step4_localization_v3_1/component_lookup.json';lookup=json.loads(lookup_path.read_text())
    image_ids=[r['image_id'] for r in rows];groups=sorted({lookup[i] for i in image_ids})
    blocks=[np.array([j for j,i in enumerate(image_ids) if lookup[i]==group]) for group in groups]
    scores={}
    for arm in ['ce_only','lens_losses']:
        for stage in ['PRE','LAST']:
            scores[arm+'_'+stage]=np.array([normalized(r['text'])==normalized(r['target']) for r in datasets[arm][stage]['rows']],float)
    differences={'Lens_minus_CE_LAST':scores['lens_losses_LAST']-scores['ce_only_LAST'],
        'Lens_LAST_minus_PRE':scores['lens_losses_LAST']-scores['lens_losses_PRE'],
        'CE_LAST_minus_PRE':scores['ce_only_LAST']-scores['ce_only_PRE'],
        'Lens_LAST_minus_TRAIN_majority':scores['lens_losses_LAST']-constant_score}
    rng=np.random.default_rng(20261104);draws={k:[] for k in differences}
    for _ in range(2000):
        indices=np.concatenate([blocks[j] for j in rng.integers(0,len(groups),len(groups))])
        for key,values in differences.items():draws[key].append(float(values[indices].mean()))
    bootstrap={'seed':20261104,'draws':2000,'components':len(groups),'records':8,
        'contrasts':{k:{'estimate':float(v.mean()),'CI95':np.quantile(draws[k],[.025,.975]).tolist()} for k,v in differences.items()},
        'component_lookup_sha256':sha(lookup_path),'sources':sources,
        'scope':'primary strict normalized answer match; small reused development cohort, one seed, components not patients; no clinical or original-method superiority claim'}
    (QUALITY/'paired_component_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    lines=['# Bounded native-asset Lens pilot — adapted engineering comparison','',
        '28 eligible TRAIN cases from32 fixed candidates;4 no-mask cases excluded without replacement. Reused8 development-informed model-validation cases. Same fresh source-sized LoRA, same three-view training, one alignment+two QA epochs,12updates/42microbatches per arm.1024 processor longest edge vs published2048; existing CSV region alignment adaptation. No organ fallback or annotation ROI.','',
        '| Arm | Stage | Strict full-answer EM | Leading yes/no (secondary) | Truncated |',
        '|---|---|---:|---:|---:|']
    for arm in ['ce_only','lens_losses']:
        for stage in ['PRE','stage1','qa_epoch1','LAST']:
            metric=audit[arm][stage]
            lines.append(f"| {arm} | {stage} | {metric['strict_accuracy']:.3f} | {metric['leading_binary_accuracy']:.3f} | {metric['truncated']} |")
    lines+=['',f'TRAIN-only majority constant `{constant}`: {counts}; strict accuracy {constant_score.mean():.3f}.',
        '',f"Primary Lens−CE LAST {bootstrap['contrasts']['Lens_minus_CE_LAST']}; paired2000 component draws, {len(groups)}groups, seed20261104.",
        '', 'This isolates contrastive losses conditional on the same views, not the whole Lens mechanism, combined BoneVQA ablations, independent clinical efficacy or an exact original-paper reproduction. No best-epoch/seed/threshold search. Exact fresh generation replays and independent saved-token/target/metric checks passed. Actual figure review, final interpretation and closure remain pending.']
    (QUALITY/'PILOT_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
