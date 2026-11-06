"""Metadata-only trace preflight; original guards and failed artifacts retained."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_state_trace_v3_3'
OLD=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_2'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_preflight.json';assert not destination.exists()
    plan=read(OUT/'protocol.json');old=read(OLD/'protocol.json')
    assert read(OLD/'progress.json')['status']=='failed_preserved_no_automatic_retry'
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    assert {k:v for k,v in plan.items() if k!='dependencies_sha256'}=={k:v for k,v in old.items() if k!='dependencies_sha256'}
    for info in plan['targets'].values():assert sha(info['path'])==info['sha256']
    base=ROOT/'refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_1.py'
    wrapper=ROOT/'refactor_v2/experiments/probe_fullfold0_fracture_state_trace_v3_3.py'
    assert "assert digest(model)==before and sha(info['path'])==info['sha256']" in base.read_text()
    text=wrapper.read_text()
    assert 'assert digest(model)' not in text
    assert 'after_private_forward' in text and 'after_autograd' in text and 'after_public_predict' in text
    result={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sha(wrapper),
        'auditor_sha256':sha(__file__),'same_four_examples_and_scientific_choices':True,
        'stage_hash_instrumentation_only_original_invariance_assertion_retained':True,
        'physical_checkpoints_unchanged':True,'dependencies_checked':len(plan['dependencies_sha256']),
        'neural_models':0,'optimizer_updates':0}
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
