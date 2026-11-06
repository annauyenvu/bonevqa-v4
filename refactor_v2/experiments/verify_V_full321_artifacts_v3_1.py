"""Fresh-process V321 record arithmetic/source audit, not clinical confirmation."""
import json
from pathlib import Path
from src.utils import compute_metrics,normalize_yes_no
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def main():
    root=Path.cwd();folder=root/'refactor_v2/reports/V_full321_development_v3_1'
    output=folder/'independent_artifact_verification.json';assert not output.exists()
    progress=json.loads((folder/'progress.json').read_text());report=json.loads((folder/'report.json').read_text())
    assert progress['status']=='complete_development_only' and sha(folder/'report.json')==progress['report_sha256']
    assert sha(folder/'evaluation_binding.json')==report['evaluation_binding_sha256']
    binding=json.loads((folder/'evaluation_binding.json').read_text())
    assert binding['source_script_sha256']==sha(root/'refactor_v2/experiments/evaluate_V_full321_development_v3_1.py')
    assert binding['all321_canonical_dimensions_independently_checked']
    proposal=root/'refactor_v2/reports/V_full321_proposals_v3_1'
    assert sha(proposal/'boxes.json')==binding['proposal_sha256']
    assert sha((proposal/'boxes.json').with_suffix('.provenance.json'))==binding['proposal_sidecar_sha256']
    assert sha(proposal/'independent_contract_verification.json')==binding['independent_proposal_verification_sha256']
    checks={};identities=None
    for arm,expected in report['results'].items():
        data=json.loads((folder/(arm+'.json')).read_text());rows=data['records']
        assert compute_metrics(rows,fracture_threshold=.5)==data['metrics']==expected['metrics']
        assert data['weights_unchanged'] and data['overlap32_max_probability_error']<=1e-4
        assert len(rows)==len({r['image_id'] for r in rows})==321
        assert sum(normalize_yes_no(r['gt'])=='yes' for r in rows)==57
        assert all(r['question']=='Is there a fracture in this X-ray?' and r['answer_type']=='CLOSED' for r in rows)
        identity=[(r['image_id'],r['gt']) for r in rows]
        if identities is None:identities=identity
        else:assert identities==identity
        assert [r['image_id'] for r in rows]==binding['image_ids']
        assert sha(root/'refactor_v2/reports/vl_bounded_quality_v3_1'/arm/'last.pt')==binding['checkpoint_sha256'][arm]
        counts={k:0 for k in ('TP','FN','TN','FP')}
        for row in rows:
            label=normalize_yes_no(row['gt'])=='yes';pred=row['p_yes']>=.5
            counts[('TP' if pred else 'FN') if label else ('FP' if pred else 'TN')]+=1
        checks[arm]={'passed':True,'confusion_raw_half':counts,'records_sha256':sha(folder/(arm+'.json'))}
    assert report['training_updates']==0 and not report['calibration_or_test_used']
    result={'passed':True,'scope':'independent source/hash/saved-record arithmetic audit, NOT independent holdout',
            'source_script_sha256':sha(__file__),'report_sha256':sha(folder/'report.json'),'checks':checks,
            'OPEN_metrics_not_applicable':'No OPEN questions in canonical321 fracture endpoint',
            'training_updates':0,'calibration_or_test_used':False,'production_or_efficacy_release':False}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
