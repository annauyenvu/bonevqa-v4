"""Metadata-only five-fold LAST/role seal; does not authorize reserved inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
REPORTS=ROOT/'refactor_v2/reports'
DATA=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3'
OUT=REPORTS/'grouped_evaluation_bindings_v3_1'
CANONICAL='Is there a fracture in this X-ray?'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert not OUT.exists(),'Preserve prior bindings'
    manifest=read(DATA/'manifest.json')
    owner={}
    for index,members in enumerate(manifest['components']):
        for image in members:
            assert image not in owner
            owner[image]=f'component_{index:05d}'
    dependencies={};folds={};outer_owners={};outer_components={}
    def bind(p):
        p=Path(p);dependencies[str(p)]=sha(p);return str(p)
    bind(DATA/'manifest.json');bind(__file__)
    for name in ['GROUPED_EVALUATION_RULES_V3_1.json','GROUPED_CONFIRMATION_AMENDMENT_V3_2.json','grouped_calibration_technical_gate_v3_1.json','grouped_bootstrap_technical_gate_v3_1.json']:
        bind(REPORTS/name)
    for gate in ('grouped_calibration_technical_gate_v3_1.json','grouped_bootstrap_technical_gate_v3_1.json'):
        assert read(REPORTS/gate)['passed']
    original=read(REPORTS/'fullfold0_B_uniform_screen_v3_1/protocol.json')['configs']['token_mean']
    for fold in ('0','1','2','3','4'):
        if fold=='0':
            config_path=Path(original['path']);assert sha(config_path)==original['sha256']
            done_path=REPORTS/'fullfold0_B_uniform_screen_v3_1/token_mean_completion.json'
            cfg=read(config_path);checkpoint=ROOT/cfg['checkpoint_dir']/cfg['run_name']/'last.pt'
            assert read(done_path)['primary_is_LAST1438']
            assert read(REPORTS/'fullfold0_fixed_LAST_evaluation_v3_1/independent_artifact_audit.json')['passed']
            bind(REPORTS/'fullfold0_fixed_LAST_evaluation_v3_1/independent_artifact_audit.json')
        else:
            b=read(REPORTS/'grouped_native_training_v3_1/release.json')['bindings'][fold]
            config_path=Path(b['config']);cfg=read(config_path);checkpoint=Path(b['checkpoint_dir'])/'last.pt'
            done_path=REPORTS/f'grouped_native_training_v3_1/fold_{fold}_completion.json'
            audit=REPORTS/f'grouped_modelval_evaluation_v3_1/fold_{fold}_independent_metric_audit.json'
            assert read(audit)['passed'];bind(audit)
        assert sha(checkpoint)==read(done_path)['last_sha256']
        bind(config_path);bind(checkpoint);bind(done_path);bind(cfg['data']['sealed_answer_vocab'])
        roles={};sets={}
        for role in ('train','model_val','calibration','test'):
            path=DATA/f'fold_{fold}/vqa_{role}.json';rows=read(path);m=manifest['folds'][fold][role]
            assert sha(path)==m['sha256'] and len(rows)==m['qa']
            ids={r['image_id'] for r in rows};assert len(ids)==m['images'] and ids<=owner.keys()
            canonical=[r for r in rows if r['question']==CANONICAL]
            assert len(canonical)==len(ids)==len({r['image_id'] for r in canonical})
            assert all(r['answer'].strip().lower() in ('yes','no') for r in canonical)
            components={owner[i] for i in ids};sets[role]=components
            roles[role]={'path':bind(path),'sha256':sha(path),'QA':len(rows),'images':len(ids),'known_components':len(components)}
            if role=='test':
                for image in ids:assert image not in outer_owners;outer_owners[image]=fold
                for c in components:assert c not in outer_components;outer_components[c]=fold
        for role in sets:
            for other in sets:
                if role!=other:assert not sets[role]&sets[other],(fold,role,other)
        folds[fold]={'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'config':str(config_path),'roles':roles,'reuse_disclosure':'Historical fold0 trained and inspected before amendment' if fold=='0' else 'Prospectively bound fixed native two-epoch LAST'}
    assert set(outer_owners)==set(owner),'Outer roles must cover each manifest image exactly once'
    plan={'status':'METADATA_SEALED_PENDING_RUNNER_INTEGRATION_GATES','folds':folds,'image_to_known_component':owner,'dependencies':dependencies,
          'outer_images':len(outer_owners),'outer_known_components':len(outer_components),'public_input_keys':['images','questions'],'seed':42,'raw_threshold':.5,
          'calibration_test_execution_authorized':False,'scope':'Development-informed grouped evaluation; no untouched confirmation or verified patient grouping','production_released':False}
    OUT.mkdir()
    with (OUT/'bindings.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
    print(json.dumps({'passed_metadata_seal':True,'folds':list(folds),'outer_images':len(outer_owners),'outer_known_components':len(outer_components),'reserved_role_inference':False}))

if __name__=='__main__':main()
