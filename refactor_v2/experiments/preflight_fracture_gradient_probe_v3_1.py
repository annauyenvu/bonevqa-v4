"""Metadata/source-only preflight for tiny zero-update gradient probe."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_1'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    destination=OUT/'independent_preflight.json';assert not destination.exists()
    plan=read(OUT/'protocol.json')
    for path,digest in plan['dependencies_sha256'].items():assert sha(path)==digest,path
    assert plan['optimizer_updates']==plan['threshold_searches']==0
    assert not plan['new_weights_saved'] and not plan['calibration_test_or_new_modelval_inference']
    source=ROOT/'refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_1.py'
    tree=ast.parse(source.read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('backward','step','train','save') for n in ast.walk(tree))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr=='predict_batch']
    assert len(calls)==1 and [ast.literal_eval(k) for k in calls[0].args[0].keys]==['images','questions']
    train=read(plan['TRAIN_manifest']);rows=[train[i] for i in plan['selected_indices']]
    assert [r['answer'] for r in rows]==['no','yes','no','yes']
    assert all(r['question']=='Is there a fracture in this X-ray?' for r in rows)
    assert [r['image_id'] for r in rows]==plan['image_ids'] and len(set(plan['image_ids']))==4
    parent=read(ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1/protocol.json')
    cfg=read(parent['configs']['token_mean']['path']);b=cfg['train']['component_batching']
    components=read(b['manifest'])['components']
    lookup={image:g for g,images in enumerate(components) for image in images}
    assert len({lookup[r['image_id']] for r in rows})==4
    chosen=[];used=set()
    for label in ('no','yes','no','yes'):
        i=next(i for i,r in enumerate(train) if r['question']=='Is there a fracture in this X-ray?'
               and r['answer']==label and lookup[r['image_id']] not in used)
        chosen.append(i);used.add(lookup[train[i]['image_id']])
    assert chosen==plan['selected_indices']
    for mode,info in plan['targets'].items():
        assert sha(info['path'])==info['sha256']
        assert sha(info['config']['path'])==info['config']['sha256']
    result={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),'auditor_sha256':sha(__file__),
        'source_sha256':sha(source),'dependencies_checked':len(plan['dependencies_sha256']),
        'deterministic4TRAIN_examples4components2yes2no':True,'no_prediction_selection':True,
        'AST_public_inputs_images_questions_only':True,'AST_no_backward_step_train_save':True,
        'mode':'eval_autograd_not_stochastic_train','neural_models':0,'optimizer_updates':0}
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
