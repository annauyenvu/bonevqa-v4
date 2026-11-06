"""Separate source/selection verification for exact-CE-reduction repair."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_2'
OLD=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_preflight.json';assert not destination.exists()
    assert read(OLD/'progress.json')['status']=='failed_preserved_no_automatic_retry'
    assert read(OLD/'progress.json')['failure']=='AssertionError()'
    plan=read(OUT/'protocol.json');previous=read(OLD/'protocol.json')
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    assert {k:v for k,v in plan.items() if k!='dependencies_sha256'}=={k:v for k,v in previous.items() if k!='dependencies_sha256'}
    assert plan['selected_indices']==previous['selected_indices']
    rows=read(plan['TRAIN_manifest'])
    assert [rows[i]['answer'] for i in plan['selected_indices']]==['no','yes','no','yes']
    assert len({rows[i]['image_id'] for i in plan['selected_indices']})==4
    base=ROOT/'refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_1.py'
    wrapper=ROOT/'refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_2.py'
    assert sha(base)=='ae0a2501b340162dee430bdd8e03b21d9c021e07182efce86961f21591a395bd'
    # Independent AST validation of the one numerical change: the exact frozen
    # loss operation is CE(reduction=none).mean(), not a relaxed comparison.
    tree=ast.parse(wrapper.read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replacements' for t in n.targets))
    replacements=ast.literal_eval(assignment.value)
    assert len(replacements)==3
    key="ce=F.cross_entropy(captured['closed_logits'].float(),labels)"
    assert replacements[key]=="ce=F.cross_entropy(captured['closed_logits'].float(),labels,reduction='none').mean()"
    source=base.read_text()
    assert "assert float((ce.detach()-out['loss_cls']).abs())==0" in source
    losses=(ROOT/'refactor_v2/src/models/losses.py').read_text()
    assert 'reduction="none"' in losses and 'return per_sample.mean()' in losses
    for mode,info in plan['targets'].items():assert sha(info['path'])==info['sha256']
    result={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'auditor_sha256':sha(__file__),
        'base_failed_source_sha256':sha(base),'wrapper_sha256':sha(wrapper),
        'original_failed_progress_sha256':sha(OLD/'progress.json'),
        'same_four_examples_checkpoints_choices':True,
        'exact_frozen_CE_reduction_restored':True,'zero_equality_tolerance_unchanged':True,
        'no_scientific_loss_or_budget_change':True,'dependencies_checked':len(plan['dependencies_sha256']),
        'neural_models':0,'optimizer_updates':0}
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
