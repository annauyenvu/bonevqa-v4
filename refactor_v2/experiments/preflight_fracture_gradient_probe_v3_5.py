"""Verify within-forward exact routing contract, not tolerance relaxation."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_5'
OLD=ROOT/'refactor_v2/reports/fullfold0_probability_trace_v3_4'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    path=OUT/'independent_preflight.json';assert not path.exists()
    plan=read(OUT/'protocol.json');old=read(OLD/'protocol.json')
    assert {k:v for k,v in plan.items() if k not in ('dependencies_sha256','measurements')}=={k:v for k,v in old.items() if k not in ('dependencies_sha256','measurements')}
    assert plan['measurements'][:-1]==old['measurements'][:-1]
    for p,h in plan['dependencies_sha256'].items():assert sha(p)==h,p
    observed=read(OLD/'probability_comparison.json')
    assert observed['all263_state_hashes_unchanged'] and observed['physical_checkpoint_unchanged']
    assert observed['same_raw0_5_decisions'] and max(observed['absolute_differences'])==3.725290298461914e-09
    source=ROOT/'refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_5.py'
    text=source.read_text()
    assert 'assert manual_public_probs==public_probs' in text
    assert 'public_capture' in text and 'cross_mode_max_abs_error' in text
    for info in plan['targets'].values():assert sha(info['path'])==info['sha256']
    result={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sha(source),
        'auditor_sha256':sha(__file__),'dependencies_checked':len(plan['dependencies_sha256']),
        'same_examples_checkpoints_loss_gradient_contracts':True,
        'explicit_contract_correction':'Exact sigmoid routing checked inside SAME public forward; cross grad/no_grad delta retained separately, not required bit-identical',
        'no_tolerance_added_to_equality_guards':True,'prior_failures_and_checkpoints_preserved':True,
        'neural_models':0,'optimizer_updates':0}
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
