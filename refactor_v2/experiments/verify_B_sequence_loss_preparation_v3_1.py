"""Independent TRAIN-only weight arithmetic and pure numeric loss gates."""
import json
from collections import Counter,defaultdict
import math
from src.utils import normalize_answer
from src.protocol import records_of
from experiments.run_T_bounded_current_v3_1 import ROOT,data_and_schedule
from experiments.vl_bounded_proposal_contract_v3_1 import sha
from experiments.sequence_answer_loss_candidate_v3_1 import numeric_unit_gate


def main():
    folder=ROOT/'refactor_v2/reports/B_sequence_loss_candidate_v3_1'
    output=folder/'preparation_verification.json';assert not output.exists()
    protocol=json.loads((folder/'protocol.json').read_text())
    for path,digest in protocol['dependency_sha256'].items():assert sha(path)==digest
    assert sha(folder/'TRAIN_OPEN_weights.json')==protocol['weights_sha256']
    assert sha(folder/'numeric_loss_gate.json')==protocol['numeric_gate_sha256']
    assert numeric_unit_gate()==json.loads((folder/'numeric_loss_gate.json').read_text())
    _,datasets,manifests=data_and_schedule();assert manifests==protocol['actual_manifests']
    groups=defaultdict(list)
    for row in records_of(datasets['train']):
        if row['answer_type']=='OPEN':groups[row['question']].append(normalize_answer(row['answer']))
    saved=json.loads((folder/'TRAIN_OPEN_weights.json').read_text())
    table={(r['question'],r['normalized_answer_frequency_key']):r for r in saved['weights']}
    expected=set()
    for question,answers in groups.items():
        counts=Counter(answers);n=len(answers);k=len(counts)
        raw={a:max(.5,min(2.,math.sqrt(n/k/count))) for a,count in counts.items()}
        mean=sum(raw[a] for a in answers)/n
        for answer,count in counts.items():
            key=(question,answer);expected.add(key);item=table[key]
            assert item['TRAIN_count']==count and item['question_OPEN_rows']==n
            assert item['question_unique_normalized_answers']==k
            assert abs(item['weight']-raw[answer]/mean)<1e-12
            assert .25<=item['weight']<=4
        assert abs(sum(table[(question,a)]['weight'] for a in answers)/n-1)<1e-12
    assert expected==set(table) and sum(len(v) for v in groups.values())==534
    assert not protocol['actual_model_gates_passed'] and not protocol['bounded_pilot_authorized']
    report={'passed':True,'source_script_sha256':sha(__file__),'protocol_sha256':sha(folder/'protocol.json'),
        'all534existing_TRAIN_OPEN_weight_keys_counts_formula_and_perquestion_mean_exact':True,
        'numeric_CE_value_gradient_padding_and_uniform_control_gates_passed':True,
        'model_instantiations':0,'training_updates':0,'new_clinical_labels':False,
        'actual_Qwen_forward_gradient_update_and_initialization_gates_pending':True,
        'bounded_pilot_authorized':False,'long_training_authorized':False,'step3_complete':False}
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
