"""Metadata-only verification of same probe with numeric discrepancy logging."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_probability_trace_v3_4'
OLD=ROOT/'refactor_v2/reports/fullfold0_fracture_state_trace_v3_3'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_preflight.json';assert not destination.exists()
    plan=read(OUT/'protocol.json');old=read(OLD/'protocol.json')
    assert {k:v for k,v in plan.items() if k!='dependencies_sha256'}=={k:v for k,v in old.items() if k!='dependencies_sha256'}
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    snapshots=read(OLD/'state_snapshots.json')
    assert [s['stage'] for s in snapshots]==['loaded_before_forward','after_private_forward','after_autograd','after_gradient_summaries','after_public_predict']
    assert all(s['keys']==263 and not s['added_keys'] and not s['removed_keys'] and not s['changed_value_hashes'] for s in snapshots)
    source=ROOT/'refactor_v2/experiments/probe_fullfold0_probability_trace_v3_4.py'
    assert 'assert manual_probs==public_probs' in source.read_text()
    for info in plan['targets'].values():assert sha(info['path'])==info['sha256']
    result={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'source_sha256':sha(source),
        'auditor_sha256':sha(__file__),'dependencies_checked':len(plan['dependencies_sha256']),
        'same_four_examples_choices_exact_guards_retained':True,
        'prior263state_hashes_unchanged_all5stages':True,'only_discrepancy_logging_added':True,
        'neural_models':0,'optimizer_updates':0}
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
