"""Read-only completed V contrast audit; no model inference or optimization."""
import json
from pathlib import Path
from src.utils import normalize_answer,normalize_yes_no,is_canonical_fracture_presence_question
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def correct(row):
    norm=normalize_yes_no if row['answer_type']=='CLOSED' else normalize_answer
    return norm(row['pred'])==norm(row['gt'])


def main():
    folder=Path.cwd()/'refactor_v2/reports/vl_bounded_quality_v3_1'
    output=folder/'V_matched_interpretation_audit.json'
    assert not output.exists()
    pair=json.loads((folder/'V_LAST296_matched_comparison.json').read_text())
    paths={arm:folder/arm/'independent_LAST296_predictions.json' for arm in ('V_real','V_location_placebo')}
    records={a:json.loads(p.read_text()) for a,p in paths.items()}
    changes=[];net={};fracture=[]
    for a,b in zip(records['V_real'],records['V_location_placebo']):
        assert all(a[k]==b[k] for k in ('image_id','question','gt','answer_type'))
        delta=int(correct(a))-int(correct(b))
        row=net.setdefault(a['image_id'],{'overall_correct_delta':0,'open_correct_delta':0})
        row['overall_correct_delta']+=delta
        if a['answer_type']!='CLOSED':row['open_correct_delta']+=delta
        if a['pred']!=b['pred']:
            changes.append({'image_id':a['image_id'],'question':a['question'],'answer_type':a['answer_type'],
                            'gt':a['gt'],'real_pred':a['pred'],'control_pred':b['pred'],
                            'real_correct':correct(a),'control_correct':correct(b)})
        if is_canonical_fracture_presence_question(a['question']):
            fracture.append({'image_id':a['image_id'],'gt':a['gt'],'real':a['p_yes'],'control':b['p_yes']})
    assert len(fracture)==32 and len(net)==32 and len(changes)==6
    strict_order_changes=sum((a['real']>b['real'])!=(a['control']>b['control']) for a in fracture for b in fracture)
    threshold_changes=[r for r in fracture if (r['real']>=.5)!=(r['control']>=.5)]
    contrast=pair['contrasts']['V_location_placebo']
    report={'scope':'same completed seed42 Vpair development interpretation; no new inference',
            'source_script_sha256':sha(__file__),'comparison_sha256':sha(folder/'V_LAST296_matched_comparison.json'),
            'prediction_sha256':{a:sha(p) for a,p in paths.items()},
            'QA_correct_counts':{a:sum(correct(r) for r in rows) for a,rows in records.items()},
            'OPEN_correct_counts':{a:sum(correct(r) for r in rows if r['answer_type']!='CLOSED') for a,rows in records.items()},
            'answer_changes':changes,'per_image_net_correct_deltas':net,
            'canonical_fracture_pairwise_strict_order_changes':strict_order_changes,
            'canonical_fracture_max_probability_difference':max(abs(r['real']-r['control']) for r in fracture),
            'raw_half_threshold_crossing_rows':threshold_changes,
            'same_ranking_is_not_same_probabilities':True,
            'overall_accuracy_descriptive_interval_contains_zero':contrast['paired_component_bootstrap_95_descriptive']['overall_acc']['interval'][0]<=0<=contrast['paired_component_bootstrap_95_descriptive']['overall_acc']['interval'][1],
            'conclusion':'small local OPEN signal; overall/clinical V benefit not established. No expansion or Final Solution claim from this single-seed reused development pair.',
            'clinical_positive_images':5,'calibration_or_test_used':False,'threshold_fitted':False}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
