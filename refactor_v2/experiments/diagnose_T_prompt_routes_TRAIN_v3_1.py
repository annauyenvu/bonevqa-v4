"""Four fixed TRAIN rows, factorial text-route intervention, zero updates.

Intentionally mixed branch prompts are internal diagnostics, NOT a new public
protocol/candidate. Original sources, checkpoints and coherent runtime remain.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import time

import torch
from src.data.prompts import textual_prompt
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.utils import set_seed
from experiments.run_T_bounded_current_v3_1 import data_and_schedule,hashes,FOLDER,ROOT
from experiments.run_vl_bounded_pilot_v3_1 import check_environment
from experiments.vl_bounded_proposal_contract_v3_1 import sha

OUT = ROOT/'refactor_v2/reports/T_prompt_routes_TRAIN_v3_1'
QUALITY = ROOT/'refactor_v2/reports/T_independent_quality_current_v3_3'
CHECKPOINT = ROOT/'refactor_v2/checkpoints/T_clinical_seed42_bounded_current_v3_1/last.pt'
CONDITIONS = [('clinical','clinical'),('clinical_paraphrase','clinical'),
              ('clinical','clinical_paraphrase'),('clinical_paraphrase','clinical_paraphrase')]


def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    check_environment();assert not OUT.exists()
    quality = json.loads((QUALITY/'report.json').read_text())
    audit = json.loads((QUALITY/'independent_artifact_audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==sha(QUALITY/'report.json')
    assert sha(CHECKPOINT)==quality['checkpoint_sha256']['clinical']
    preflight = json.loads((FOLDER/'preflight_report.json').read_text())
    indices = preflight['TRAIN_gate_indices'];assert indices==[0,3,9,11]
    _,datasets,manifests = data_and_schedule()
    selected = [records_of(datasets['train'])[i] for i in indices]
    assert {r['answer_type'] for r in selected}=={'CLOSED','OPEN'}
    OUT.mkdir()
    dependencies = {str(p.resolve()):sha(p) for p in (ROOT/'refactor_v2/src').rglob('*.py')}
    for p in [Path(__file__),FOLDER/'preflight_report.json',QUALITY/'report.json',
              QUALITY/'independent_artifact_audit.json',
              ROOT/'refactor_v2/experiments/run_T_bounded_current_v3_1.py',
              ROOT/'refactor_v2/experiments/run_vl_bounded_pilot_v3_1.py',
              ROOT/'refactor_v2/experiments/vl_bounded_proposal_contract_v3_1.py']:
        dependencies[str(p.resolve())]=sha(p)
    dump(OUT/'protocol.json',{'scope':'Four previously fixed TRAIN rows; factorial route intervention, not candidate inference',
        'dependency_sha256':dependencies,'checkpoint_sha256':sha(CHECKPOINT),
        'TRAIN_indices':indices,'selected_TRAIN_records':selected,'actual_manifests':manifests,
        'conditions_biomedical_then_LLM':CONDITIONS,'public_input_keys':['images','questions'],
        'normal_unhooked_bothclinical_and_bothparaphrase_controls_required':True,
        'same_biomedical_route_must_preserve_core_heads_exactly_across_LLM_style':True,
        'fresh_constructor_replay_seeds':[90210,90211],
        'ground_truth_as_inference_input':False,'training_updates':0,
        'modelval_calibration_test_inference_allowed':False,'production_or_long_training_release':False})


def slim(predictions):
    return [{k:p[k] for k in ('answer','answer_type','confidence','p_yes') if k in p} for p in predictions]


@torch.no_grad()
def measure(dataset,indices,seed):
    from src.data import collate_fn
    batch = collate_fn([dataset[i] for i in indices])
    public = {'images':batch['images'],'questions':batch['questions']}
    set_seed(seed);model=BoneVQAModel.load(str(CHECKPOINT),device='cpu').eval()
    original_cfg=copy.deepcopy(model.cfg);before=hashes(model.trainable_state_dict())
    assert not any(model.cfg[k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_region_in_prompt'))
    assert model.cfg.get('closed_llm_weight',0)==0 and not model.cfg.get('closed_via_llm',False)
    normal={}
    for style in ('clinical','clinical_paraphrase'):
        model.cfg['textual_prompt_style']=style
        normal[style]=slim(model.predict_batch(public))
    results={};captures={}
    for bio,llm in CONDITIONS:
        name=bio+'__'+llm;model.cfg['textual_prompt_style']=llm
        texts={textual_prompt(q,None,style=llm):textual_prompt(q,None,style=bio) for q in batch['questions']}
        expected_bio=[textual_prompt(q,None,style=bio) for q in batch['questions']]
        observed_bio=[];observed_llm=[];captured=[]
        original_core=model._encode_core;original_inputs=model._build_llm_inputs
        def change_bio(module,args):
            assert all(t in texts for t in args[0])
            replacement=[texts[t] for t in args[0]];observed_bio.extend(replacement)
            return (replacement,*args[1:])
        def capture_core(*args,**kwargs):
            core=original_core(*args,**kwargs)
            captured.append({'text_tokens':core['txt']['tokens'].detach().cpu().clone(),
                'text_mask':core['txt']['mask'].detach().cpu().clone(),
                'fusion':core['fus']['x_f'].detach().cpu().clone(),
                'closed_logits':core['logits_closed'].detach().cpu().clone(),
                'type_logit':core['type_logit'].detach().cpu().clone(),
                'fracture_logit':core['fracture_logit'].detach().cpu().clone()})
            return core
        def capture_llm(prefix,questions,answers):
            assert answers is None, 'No teacher forcing or GT targets in this diagnostic'
            assert all(q in texts for q in questions)
            observed_llm.extend(questions)
            return original_inputs(prefix,questions,answers)
        handle=model.text.register_forward_pre_hook(change_bio)
        model._encode_core=capture_core;model._build_llm_inputs=capture_llm
        try:
            predictions=slim(model.predict_batch(public))
        finally:
            handle.remove();model._encode_core=original_core;model._build_llm_inputs=original_inputs
        assert observed_bio==expected_bio and len(captured)==1
        assert observed_llm, 'Fixed TRAIN probe must actually exercise LLM routing'
        captures[name]=captured[0]
        assert all(torch.isfinite(v).all() for v in captured[0].values())
        results[name]={'predictions':predictions,'biomedical_texts':observed_bio,
            'actual_LLM_input_texts':observed_llm,'all_captured_states_finite':True}
    assert results['clinical__clinical']['predictions']==normal['clinical']
    assert results['clinical_paraphrase__clinical_paraphrase']['predictions']==normal['clinical_paraphrase']
    fixed_bio_checks={}
    for bio in ('clinical','clinical_paraphrase'):
        a=captures[bio+'__clinical'];b=captures[bio+'__clinical_paraphrase']
        assert all(torch.equal(a[k],b[k]) for k in a), 'Changing only LLM text must not alter upstream text/fusion/heads'
        fixed_bio_checks[bio]={'all_text_tokens_masks_fusion_and_heads_exact':True}
    a=captures['clinical__clinical'];b=captures['clinical_paraphrase__clinical']
    differences={k:float((a[k].float()-b[k].float()).abs().max()) for k in ('fusion','closed_logits','type_logit','fracture_logit')}
    differences['text_CLS']=float((a['text_tokens'][:,0].float()-b['text_tokens'][:,0].float()).abs().max())
    base=results['clinical__clinical']['predictions']
    answer_changes={name:[i for i,(p,q) in enumerate(zip(base,value['predictions'])) if p['answer']!=q['answer']]
                    for name,value in results.items()}
    model.cfg.clear();model.cfg.update(original_cfg)
    assert hashes(model.trainable_state_dict())==before and model.cfg==original_cfg
    assert slim(model.predict_batch(public))==normal['clinical'], 'Restore normal inference exactly'
    del model;gc.collect()
    return {'conditions':results,'fixed_biomedical_route_checks':fixed_bio_checks,
        'biomedical_paraphrase_only_max_abs_core_change':differences,
        'public_answer_changed_indices_vs_bothclinical':answer_changes,
        'normal_endpoint_controls_and_restored_predictions_exact':True,'weights_unchanged':True}


def run():
    check_environment();assert not (OUT/'progress.json').exists()
    commands=subprocess.check_output(['ps','-axo','command'],text=True)
    assert not any(' -m src.train ' in r or ('Python' in r and 'evaluate_T_bounded_current_' in r) for r in commands.splitlines())
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest
    assert sha(CHECKPOINT)==protocol['checkpoint_sha256']
    _,datasets,manifests=data_and_schedule();assert manifests==protocol['actual_manifests']
    dump(OUT/'progress.json',{'status':'running_TRAIN_only_zero_updates'})
    started=time.perf_counter()
    try:
        first=measure(datasets['train'],protocol['TRAIN_indices'],90210)
        second=measure(datasets['train'],protocol['TRAIN_indices'],90211)
        assert first==second, 'Fresh constructor diagnostic replay must be exact'
        assert sha(CHECKPOINT)==protocol['checkpoint_sha256']
        report={'scope':protocol['scope'],'protocol_sha256':sha(OUT/'protocol.json'),
            'checkpoint_sha256':sha(CHECKPOINT),'results':first,'fresh_constructor_replay_exact':True,
            'elapsed_seconds':time.perf_counter()-started,'training_updates':0,'GT_inference_inputs':False,
            'modelval_calibration_or_test_inference_used':False,'long_training_authorized':False,
            'limitations':['Four TRAIN rows cannot attribute the full142QA development accuracy loss',
                'Mixed branch prompts are internal intervention, not a candidate/production protocol',
                'No new architecture, loss, threshold or template selected; no clinical quality claim']}
        dump(OUT/'report.json',report)
        dump(OUT/'progress.json',{'status':'complete_TRAIN_diagnostic_only','report_sha256':sha(OUT/'report.json')})
        print(json.dumps(report,indent=2))
    except Exception as error:
        dump(OUT/'progress.json',{'status':'failed_preserved_no_automatic_retry','failure':repr(error)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().mode]()
