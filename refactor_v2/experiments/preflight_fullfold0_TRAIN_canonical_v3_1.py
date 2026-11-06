"""Separate preflight; dataset construction only, no neural model."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
OUT=ROOT/'refactor_v2/reports/fullfold0_TRAIN_canonical_diagnostic_v3_1'
QUALITY=ROOT/'refactor_v2/reports/fullfold0_fixed_LAST_evaluation_v3_1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


if __name__=='__main__':
    destination=OUT/'independent_preflight.json'
    assert not destination.exists()
    plan=read(OUT/'protocol.json')
    for path,digest in plan['dependencies_sha256'].items():
        assert sha(path)==digest,path
    source=ROOT/'refactor_v2/experiments/diagnose_fullfold0_TRAIN_canonical_v3_1.py'
    tree=ast.parse(source.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and n.func.attr=='predict_batch']
    assert len(calls)==1 and isinstance(calls[0].args[0],ast.Dict)
    assert [ast.literal_eval(k) for k in calls[0].args[0].keys]==['images','questions']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and n.func.attr in ('backward','step','train','save') for n in ast.walk(tree))
    assert plan['batch_size']==4 and plan['raw_threshold']==.5
    assert plan['constructor_seeds']==[90210,90211] and plan['training_seed']==42
    assert not plan['calibration_test_inference'] and plan['optimizer_updates']==plan['threshold_searches']==0
    assert not plan['new_quality_training_released']
    quality=read(QUALITY/'independent_artifact_audit.json')
    assert quality['passed'] and quality['report_sha256']==sha(QUALITY/'report.json')
    from src.data import build_dataset
    from src.train import build_split
    from src.protocol import records_of
    from experiments.run_vl_bounded_pilot_v3_1 import check_environment
    check_environment()
    raw=read(plan['TRAIN_manifest'])
    expected=[i for i,r in enumerate(raw) if r['question']==plan['canonical_question']]
    assert expected==plan['selected_indices']
    selected=[raw[i] for i in expected]
    assert len(selected)==len({r['image_id'] for r in selected})==2568
    assert sum(r['answer']=='yes' for r in selected)==458
    assert sum(r['answer']=='no' for r in selected)==2110
    for mode,info in plan['targets'].items():
        assert sha(info['path'])==info['sha256']==read(QUALITY/(mode+'.json'))['checkpoint_sha256']
        cfg=read(info['config']['path'])
        assert sha(info['config']['path'])==info['config']['sha256']
        assert cfg['data']['loader_kwargs']['sealed_records_by_split']['train']==plan['TRAIN_manifest']
        assert not cfg['data']['loader_kwargs']['with_boxes'] and not cfg['data']['loader_kwargs']['with_masks']
        dataset=build_split(build_dataset,cfg['data']['dataset'],'train',None,True,cfg['data']['loader_kwargs'])
        assert records_of(dataset)==raw
        assert [records_of(dataset)[i]['image_id'] for i in expected]==plan['selected_image_ids']
    report={'passed':True,'protocol_sha256':sha(OUT/'protocol.json'),
        'auditor_sha256':sha(__file__),'worker_sha256':sha(source),
        'dependencies_checked':len(plan['dependencies_sha256']),
        'both_physical_LAST_hashes_match_verified_quality_pair':True,
        'dataset_original_order_records_exact':True,'all2568canonical_TRAIN_images':True,
        'positive_negative_counts':[458,2110],'AST_public_predict_keys':['images','questions'],
        'AST_no_backward_step_train_save_calls':True,'neural_model_instantiations':0,
        'optimizer_updates':0,'reserved_role_model_inference':False}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
