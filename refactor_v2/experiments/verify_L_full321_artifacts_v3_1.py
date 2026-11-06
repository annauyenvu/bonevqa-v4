"""Independent report/record arithmetic verification; no model inference."""
import json
from pathlib import Path
from src.utils import compute_metrics,normalize_yes_no
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def main():
    root=Path.cwd();folder=root/'refactor_v2/reports/L_full321_development_v3_1'
    destination=folder/'independent_artifact_verification.json'
    assert not destination.exists()
    report=json.loads((folder/'report.json').read_text())
    progress=json.loads((folder/'progress.json').read_text())
    assert progress['status']=='complete_development_only' and sha(folder/'report.json')==progress['report_sha256']
    assert sha(folder/'evaluation_binding.json')==report['evaluation_binding_sha256']
    binding=json.loads((folder/'evaluation_binding.json').read_text())
    assert sha(root/'refactor_v2/experiments/evaluate_L_full321_development_v3_1.py')==binding['source_script_sha256']
    assert report['training_updates']==0 and not report['calibration_or_test_used']
    checks={};identities=None
    for arm,expected in report['results'].items():
        data=json.loads((folder/(arm+'.json')).read_text())
        rows=data['records']
        assert compute_metrics(rows,fracture_threshold=.5)==expected['metrics']==data['metrics']
        assert data['weights_unchanged'] and data['overlap32_max_probability_error']<=1e-4
        assert len(rows)==321 and len({r['image_id'] for r in rows})==321
        assert sum(normalize_yes_no(r['gt'])=='yes' for r in rows)==57
        identity=[(r['image_id'],r['gt'],r['question']) for r in rows]
        if identities is None:identities=identity
        else:assert identities==identity
        assert [r['image_id'] for r in rows]==binding['image_ids']
        assert all(r['answer_type']=='CLOSED' and r['question']=='Is there a fracture in this X-ray?' for r in rows)
        checkpoint=root/'refactor_v2/reports/vl_bounded_quality_v3_1'/arm/'last.pt'
        assert sha(checkpoint)==binding['checkpoint_sha256'][arm]
        counts={k:0 for k in ('TP','FN','TN','FP')}
        for row in rows:
            label=normalize_yes_no(row['gt'])=='yes';pred=row['p_yes']>=.5
            counts[('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')]+=1
        checks[arm]={'passed':True,'confusion_raw_half':counts,'records_sha256':sha(folder/(arm+'.json'))}
    result={'passed':True,'scope':'independent saved-record/checkpoint-hash arithmetic audit; not new holdout',
            'source_script_sha256':sha(__file__),'report_sha256':sha(folder/'report.json'),'checks':checks,
            'OPEN_metrics_not_applicable':'zero OPEN questions in this canonical binary endpoint; reported zeros do not mean OPEN failure',
            'calibration_or_test_used':False,'training_updates':0,'production_or_efficacy_release':False}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
