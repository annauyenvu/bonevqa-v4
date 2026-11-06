"""Predeclared 24-TRAIN-row extension of the preserved four-row route probe.

Six batch4 probes; deterministic question/type/component coverage selection,
never prediction-based selection, training or new validation inference.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

from src.context_controls import load_component_lookup
from src.utils import normalize_answer,normalize_yes_no
from experiments import diagnose_T_prompt_routes_TRAIN_v3_1 as prior

ROOT=prior.ROOT
OUT=ROOT/'refactor_v2/reports/T_routes_stratified24_TRAIN_v3_1'


def select_rows():
    base,datasets,manifests=prior.data_and_schedule()
    rows=prior.records_of(datasets['train'])
    binding=base['train']['component_batching']
    lookup,_=load_component_lookup(binding['manifest'],binding['seal'],ROOT)
    used=set();selected={}
    # Round-robin sorted question templates, preserving source row order within
    # each template. Reserve OPEN first so both decoder and heads are exercised.
    for kind in ('OPEN','CLOSED'):
        buckets={q:[i for i,r in enumerate(rows) if r['answer_type']==kind and r['question']==q]
                 for q in sorted({r['question'] for r in rows if r['answer_type']==kind})}
        chosen=[]
        while len(chosen)<12:
            made_progress=False
            for bucket in buckets.values():
                while bucket and lookup[rows[bucket[0]]['image_id']] in used:
                    bucket.pop(0)
                if bucket:
                    i=bucket.pop(0);chosen.append(i);used.add(lookup[rows[i]['image_id']]);made_progress=True
                if len(chosen)==12:break
            assert made_progress, 'Cannot satisfy declared coverage; do not silently change design'
        selected[kind]=chosen
    batches=[selected['CLOSED'][i:i+2]+selected['OPEN'][i:i+2] for i in range(0,12,2)]
    indices=[i for batch in batches for i in batch]
    assert len(set(indices))==len({lookup[rows[i]['image_id']] for i in indices})==24
    assert all([rows[i]['answer_type'] for i in batch]==['CLOSED','CLOSED','OPEN','OPEN'] for batch in batches)
    return datasets['train'],manifests,batches,lookup


def prepare():
    prior.check_environment();assert not OUT.exists()
    ds,manifests,batches,lookup=select_rows()
    previous=json.loads((prior.OUT/'report.json').read_text())
    assert previous['fresh_constructor_replay_exact'] and previous['results']['weights_unchanged']
    assert prior.sha(prior.CHECKPOINT)==previous['checkpoint_sha256']
    rows=prior.records_of(ds);indices=[i for b in batches for i in b]
    dependencies={str(p.resolve()):prior.sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for path in [Path(__file__),Path(prior.__file__),prior.OUT/'report.json',prior.OUT/'protocol.json',
                 ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py',
                 ROOT/'refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py',
                 ROOT/'refactor_v2/experiments/vl_bounded_proposal_contract_v3_1.py']:
        dependencies[str(path.resolve())]=prior.sha(path)
    OUT.mkdir()
    prior.dump(OUT/'protocol.json',{'scope':'24 deterministic TRAIN rows; explanatory routing diagnostic only',
        'selection':'OPEN then CLOSED, sorted-question round robin, earliest unused component, no model predictions',
        'selected_indices_by_batch':batches,'selected_TRAIN_records':[rows[i] for i in indices],
        'component_count':24,'unique_questions':len({rows[i]['question'] for i in indices}),
        'CLOSED_rows':12,'OPEN_rows':12,'batch_size':4,'batch_count':6,
        'dependency_sha256':dependencies,'checkpoint_sha256':prior.sha(prior.CHECKPOINT),
        'actual_manifests':manifests,'conditions_biomedical_then_LLM':prior.CONDITIONS,
        'public_input_keys':['images','questions'],'GT_inference_input':False,
        'fresh_constructors_each_batch_seeds':[90210,90211],
        'same_biomedical_core_exact_and_coherent_endpoint_and_restore_checks_required':True,
        'training_updates':0,'modelval_calibration_test_inference_allowed':False,
        'threshold_template_architecture_or_loss_selection_allowed':False,
        'long_training_authorized':False})
    print('Prepared six batch4 TRAIN probes:24distinct components,12CLOSED12OPEN; no model loaded')


def summarize(results,records):
    from src.utils import compute_metrics
    summary={};base=results['clinical__clinical']
    def correct(record,prediction):
        normalize=normalize_yes_no if record['answer_type']=='CLOSED' else normalize_answer
        return normalize(prediction['answer'])==normalize(record['answer'])
    for condition,predictions in results.items():
        assert len(predictions)==len(records)==24
        transitions={kind:{k:0 for k in ('both_correct','correct_to_wrong','wrong_to_correct','both_wrong','answer_text_changes')}
                     for kind in ('CLOSED','OPEN')}
        scored=[]
        for r,original,changed in zip(records,base,predictions):
            a,b=correct(r,original),correct(r,changed);d=transitions[r['answer_type']]
            d['both_correct' if a and b else 'correct_to_wrong' if a else 'wrong_to_correct' if b else 'both_wrong']+=1
            d['answer_text_changes']+=int(original['answer']!=changed['answer'])
            item={'image_id':r['image_id'],'question':r['question'],'gt':r['answer'],
                  'answer_type':r['answer_type'],'pred':changed['answer']}
            if 'p_yes' in changed:item['p_yes']=changed['p_yes']
            scored.append(item)
        summary[condition]={'metrics':compute_metrics(scored,fracture_threshold=.5),
                            'transitions_vs_bothclinical_by_GT_type':transitions}
    return summary


def run():
    prior.check_environment();assert not (OUT/'progress.json').exists()
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any(' -m src.train ' in r or ('Python' in r and ('evaluate_T_bounded_' in r or 'diagnose_T_prompt_routes_TRAIN_v3_1.py run' in r)) for r in commands.splitlines())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert prior.sha(path)==digest
    ds,manifests,batches,_=select_rows()
    assert batches==protocol['selected_indices_by_batch'] and manifests==protocol['actual_manifests']
    assert prior.sha(prior.CHECKPOINT)==protocol['checkpoint_sha256']
    state={'status':'running_TRAIN_only_zero_updates','pid':os.getpid(),'completed_batches':0}
    prior.dump(OUT/'progress.json',state);started=time.perf_counter();results={};all_predictions={}
    try:
        for number,indices in enumerate(batches):
            first=prior.measure(ds,indices,90210);second=prior.measure(ds,indices,90211)
            assert first==second
            results[str(number)]={'TRAIN_indices':indices,'results':first,'fresh_constructor_replay_exact':True}
            prior.dump(OUT/('batch_'+str(number)+'.json'),results[str(number)])
            for condition,value in first['conditions'].items():
                all_predictions.setdefault(condition,[]).extend(value['predictions'])
            state['completed_batches']=number+1;prior.dump(OUT/'progress.json',state);gc.collect()
        summary=summarize(all_predictions,protocol['selected_TRAIN_records'])
        assert prior.sha(prior.CHECKPOINT)==protocol['checkpoint_sha256']
        report={'scope':protocol['scope'],'protocol_sha256':prior.sha(OUT/'protocol.json'),
            'checkpoint_sha256':prior.sha(prior.CHECKPOINT),'summary':summary,
            'batch_report_sha256':{str(i):prior.sha(OUT/('batch_'+str(i)+'.json')) for i in range(6)},
            'all6_batches_fresh_constructor_replay_exact':True,
            'weights_cfg_methods_and_normal_public_predictions_restored_each_pass':True,
            'elapsed_seconds':time.perf_counter()-started,'training_updates':0,
            'modelval_calibration_or_test_inference_used':False,'new_clinical_labels':False,
            'long_training_authorized':False,'step3_complete':False,
            'limitations':['24 TRAIN rows are not a validation or clinical efficacy estimate',
                'Mixed branch prompts are internal interventions, NOT a production/candidate protocol',
                'Exact string mismatch is not a new clinical-equivalence or correctness label',
                'No model/architecture/loss/threshold/template change authorized by this diagnostic alone']}
        prior.dump(OUT/'report.json',report)
        prior.dump(OUT/'progress.json',{'status':'complete_TRAIN_diagnostic_only','report_sha256':prior.sha(OUT/'report.json')})
        print(json.dumps(report,indent=2))
    except Exception as error:
        state.update(status='failed_preserved_no_automatic_retry',failure=repr(error));prior.dump(OUT/'progress.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().mode]()
