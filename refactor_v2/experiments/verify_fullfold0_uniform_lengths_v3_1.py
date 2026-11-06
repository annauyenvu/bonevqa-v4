"""No neural model: verify hashes, deterministic TRAIN selection and gate artifacts."""
import json
import math
from pathlib import Path
from transformers import AutoTokenizer
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump


def main():
    f=ROOT/'refactor_v2/reports/fullfold0_uniform_lengths_v3_1';out=f/'independent_artifact_verification.json'
    assert not out.exists()
    p=json.loads((f/'protocol.json').read_text());r=json.loads((f/'report.json').read_text());progress=json.loads((f/'progress.json').read_text())
    assert r['passed'] and sha(f/'report.json')==progress['report_sha256']
    assert sha(f/'protocol.json')==r['protocol_sha256']
    for path,h in p['dependencies_sha256'].items():assert sha(Path(path))==h
    prior=ROOT/'refactor_v2/reports/fullfold0_uniform_timing_gate_v3_3/protocol.json'
    for path,h in json.loads(prior.read_text())['dependencies_sha256'].items():assert sha(Path(path))==h
    for path,h in p['selection']['tokenizer_files_sha256'].items():assert sha(Path(path))==h
    assert sha(ROOT/'refactor_v2/checkpoints/revised_roco_alignment_v3/best.pt')==p['upstream_sha256']
    base=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/fold_0'
    rows=json.loads((base/'vqa_train.json').read_text());groups=json.loads((base/'detector_split_v3.json').read_text())['component_by_image']
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct',local_files_only=True)
    lengths={i:len(tokenizer(' '+v['answer'],add_special_tokens=False)['input_ids']) for i,v in enumerate(rows) if v['answer_type']=='OPEN'}
    selected=[];used=set()
    for answer in ('yes','no'):
        i=next(i for i,v in enumerate(rows) if v['question']=='Is there a fracture in this X-ray?' and v['answer']==answer and groups[v['image_id']] not in used)
        selected.append(i);used.add(groups[rows[i]['image_id']])
    for sign in (1,-1):
        i=min((i for i in lengths if groups[rows[i]['image_id']] not in used),key=lambda i:(sign*lengths[i],i))
        selected.append(i);used.add(groups[rows[i]['image_id']])
    assert selected==p['TRAIN_indices'] and len(used)==4
    assert [rows[i]['image_id'] for i in selected]==p['TRAIN_image_ids']
    actual=[lengths[i]+1 for i in selected[2:]]
    assert actual==p['selection']['expected_supervised_lengths_including_EOS']==[2,19]
    assert len(r['all263_initial_tensor_hashes'])==263 and r['matched_initial_states_and_native_first_forward_controls_exact']
    for mode,arm in r['results'].items():
        assert len(arm['steps'])==2 and arm['public_inference_poison_target_bypass_passed'] and arm['all_trainable_states_finite']
        for step in arm['steps']:
            assert step['actual_target_lengths']==actual and math.isfinite(step['loss']) and step['seconds']>0
            assert all(v==0 for v in step['native_token_gradient_max_errors'].values())
            for key in ('native_uniform_gradient_max_differences','answer_gradient_max'):
                assert all(math.isfinite(v) and v>0 for v in step[key].values())
            assert math.isfinite(step['native_LM']) and math.isfinite(step['uniform_LM'])
        assert all(math.isfinite(v) and v>0 for v in arm['selected_updates'].values())
    a=r['results']['token_mean']['steps'][0];b=r['results']['sequence_uniform']['steps'][0]
    assert a['native_LM']==b['native_LM'] and a['uniform_LM']==b['uniform_LM']
    assert abs(a['native_LM']-a['uniform_LM'])>1e-6
    assert abs((b['loss']-a['loss'])-(a['uniform_LM']-a['native_LM']))<1e-6
    assert r['disposable_optimizer_updates']==4 and r['quality_training_updates']==0 and r['all_smoke_weights_discarded']
    assert not list(f.rglob('*.pt')) and r['model_checkpoints_saved']==0
    assert not r['modelval_calibration_outer_inference_used'] and not r['long_training_authorized']
    audit={'passed':True,'script_sha256':sha(Path(__file__).resolve()),'report_sha256':sha(f/'report.json'),
        'protocol_sha256':sha(f/'protocol.json'),'all_dependency_upstream_and_prior_protocol_hashes_verified':True,
        'TRAIN_selection_tokenizer_and_actual_supervised_lengths_recomputed':[2,19],
        'reported_selected_native_token_gradient_errors_all_zero':True,
        'reported_native_uniform_gradient_differences_finite_positive':True,
        'initial_total_loss_difference_matches_LM_reduction_difference':True,
        'no_saved_smoke_weights':True,'model_instantiations':0,'training_updates':0,
        'independent_neural_training_or_gradient_replay':False,'quality_gain_claimed':False,
        'long_training_authorized':False,'step3_complete':False}
    dump(out,audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
