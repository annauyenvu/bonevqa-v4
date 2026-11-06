"""Independent role/checkpoint metadata audit, without inference or calibration."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path('/Users/uyen/Downloads/BONEVQA')
REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'grouped_evaluation_bindings_v3_1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    output=OUT/'independent_audit.json';assert not output.exists()
    plan=read(OUT/'bindings.json')
    for p,h in plan['dependencies'].items():assert sha(p)==h,p
    manifest=read(ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json')
    expected_owner={image:f'component_{i:05d}' for i,component in enumerate(manifest['components']) for image in component}
    assert expected_owner==plan['image_to_known_component']
    all_outer=[];components_outer=[];summary={}
    for fold,b in plan['folds'].items():
        assert Path(b['checkpoint']).name=='last.pt' and sha(b['checkpoint'])==b['checkpoint_sha256']
        cfg=read(b['config'])
        assert not any(cfg['model'][k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        assert not cfg['data']['loader_kwargs']['with_boxes'] and not cfg['data']['loader_kwargs']['with_masks']
        roles={}
        for role,r in b['roles'].items():
            path=Path(r['path']);assert path.parent.name==f'fold_{fold}' and path.name==f'vqa_{role}.json'
            assert sha(path)==r['sha256']==manifest['folds'][fold][role]['sha256']
            rows=read(path);ids={row['image_id'] for row in rows}
            assert len(ids)==r['images'] and len(rows)==r['QA']
            roles[role]={expected_owner[i] for i in ids}
            if role=='test':all_outer.extend(ids);components_outer.extend(roles[role])
        assert all(not roles[a]&roles[b] for a,b in itertools.combinations(roles,2))
        summary[fold]={r:len(c) for r,c in roles.items()}
    assert len(all_outer)==len(set(all_outer))==4013 and set(all_outer)==set(expected_owner)
    assert len(components_outer)==len(set(components_outer))==3920
    result={'passed':True,'bindings_sha256':sha(OUT/'bindings.json'),'auditor_sha256':sha(__file__),'known_components_by_role':summary,'outer_images':4013,'outer_known_components':3920,'reserved_role_inference':False,'scope':'Metadata integrity and role partition audit, not clinical efficacy or patient-identity verification'}
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
