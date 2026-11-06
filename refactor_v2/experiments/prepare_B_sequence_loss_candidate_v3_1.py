"""Metadata and numeric gates only; no model, optimizer or training release."""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
from src.utils import normalize_answer
from experiments.sequence_answer_loss_candidate_v3_1 import numeric_unit_gate
from experiments.run_T_bounded_current_v3_1 import ROOT,BASE,data_and_schedule
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.run_vl_bounded_pilot_v3_1 import check_environment


def main():
    check_environment();folder=ROOT/'refactor_v2/reports/B_sequence_loss_candidate_v3_1';assert not folder.exists()
    errors=ROOT/'refactor_v2/reports/B_fullTRAIN_errors_v3_1'
    audit=json.loads((errors/'independent_artifact_audit.json').read_text())
    assert audit['passed'] and audit['report_sha256']==sha(errors/'report.json')
    base,datasets,manifests=data_and_schedule()
    from src.protocol import records_of
    rows=records_of(datasets['train']);opened=[r for r in rows if r['answer_type']=='OPEN']
    assert len(rows)==1173 and len(opened)==534
    frequencies=Counter((r['question'],normalize_answer(r['answer'])) for r in opened)
    question_n=Counter(r['question'] for r in opened)
    question_k=Counter(q for q,a in frequencies)
    clipped={key:min(2.,max(.5,math.sqrt(question_n[key[0]]/(question_k[key[0]]*n)))) for key,n in frequencies.items()}
    denominator={q:sum(n*clipped[(qq,a)] for (qq,a),n in frequencies.items() if qq==q)/question_n[q] for q in question_n}
    weights={key:value/denominator[key[0]] for key,value in clipped.items()}
    for q,n in question_n.items():
        assert abs(sum(count*weights[(qq,a)] for (qq,a),count in frequencies.items() if qq==q)/n-1)<1e-12
    assert min(weights.values())>=.25 and max(weights.values())<=4
    values=[{'question':q,'normalized_answer_frequency_key':a,'TRAIN_count':frequencies[(q,a)],
        'question_OPEN_rows':question_n[q],'question_unique_normalized_answers':question_k[q],
        'clipped_before_question_normalization':clipped[(q,a)],'weight':weights[(q,a)]} for q,a in sorted(weights)]
    unit=numeric_unit_gate()
    initial=BASE/'fracatlas_B_roco_scale_pilot_v3_initialization.pt'
    assert sha(initial)=='bfc799a5fb6db140ab833db460308c98cba76d871b9d415856175ed769abc09b'
    deps={str(p.resolve()):sha(p) for p in [Path(__file__),
        ROOT/'refactor_v2/experiments/sequence_answer_loss_candidate_v3_1.py',
        errors/'report.json',errors/'independent_artifact_audit.json',
        BASE/'fracatlas_B_roco_scale_pilot_v3.json',BASE/'data_seal.json']}
    folder.mkdir()
    def dump(name,value):(folder/name).write_text(json.dumps(value,indent=2)+'\n')
    dump('TRAIN_OPEN_weights.json',{'scope':'Weights from534 existing TRAIN OPEN targets ONLY; never runtime inputs',
        'formula':'raw=sqrt(n_question/(k_question*n_question_normalizedanswer)); clip raw [.5,2]; divide by TRAIN-row-weighted mean raw within question',
        'frequency_key_only':'frozen normalize_answer; ORIGINAL answer text/tokens NOT rewritten',
        'every_question_mean_weight':1,'actual_min':min(weights.values()),'actual_max':max(weights.values()),'weights':values})
    dump('numeric_loss_gate.json',unit)
    dump('protocol.json',{'scope':'Proposed two-arm capped B-loss refinement; NOT a training release',
        'motivation':'Verified TRAIN rare-OPEN errors62/72; descriptions80errors; does not prove causal imbalance',
        'arms':['sequence_uniform','sequence_rarity'],
        'raw_token_mean_reference':'existing Bseed42 LAST296, not best seed/epoch selection',
        'causal_control':'Uniform-sequence arm controls the change from token reduction to sequence reduction',
        'OPEN_loss':'mean over sequences of mean answer/EOS-token CE, optionally multiplied by fixed TRAIN-only row weights',
        'unchanged':'All original GT strings, CLS/fracture/type/contrastive losses, raw questions, original images, component schedule and all data roles',
        'new_modules':'none; no T/V/L/P or ROI; experimental routing adapter OFF',
        'proposed_budget_each':'seed42, same263ROCO PREtask tensors,1173TRAIN142modelval,4stage3epochs296updates,batch4accum4,CPU8float32',
        'initial_checkpoint_sha256':sha(initial),'actual_manifests':manifests,
        'weights_sha256':sha(folder/'TRAIN_OPEN_weights.json'),'numeric_gate_sha256':sha(folder/'numeric_loss_gate.json'),
        'dependency_sha256':deps,
        'required_before_any_pilot':['Actual Qwen native-token CE value/selected-gradient equivalence on fixed TRAIN gate',
            'Nonzero finite answer gradients/actual disposable updates for each candidate; discard smoke weights',
            'All263saved fresh initial tensors byteexact and optimizer/data/schedule/loss binding independently verified',
            'Validation/public inference unchanged; TRAIN weights never applied using modelval/calibration/test labels',
            'Separate per-run execution seal with bounded-only release; no quality run yet'],
        'eventual_quality_rule':'LAST296 primary, fixed142QA321development canonical metrics, paired component intervals and fresh-process replay; do not select thresholds or best epochs/templates',
        'new_clinical_labels':False,'model_instantiations':0,'training_updates':0,
        'actual_model_gates_passed':False,'bounded_pilot_authorized':False,'long_training_authorized':False,'step3_complete':False})
    print(json.dumps({'proposed_arms':2,'OPEN_weight_keys':len(weights),'minimum_weight':min(weights.values()),
        'maximum_weight':max(weights.values()),'numeric_gate':unit,'pilot_authorized':False},indent=2))


if __name__=='__main__':main()
