"""Reconcile frozen data/provenance artifacts without model calls or new splits.

Candidate-link arithmetic is rechecked, not an exhaustive similarity search or
proof of patient identity. Existing protected-role manifests are metadata only.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path.cwd()
REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'data_lineage_closure_v3_1.json'
DEPENDENCIES={}


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def read(path):
    path=Path(path);DEPENDENCIES[str(path.resolve())]=sha(path)
    return json.loads(path.read_text())


def main():
    assert not OUT.exists(),'Preserve historical audit'
    inventory=read(REPORTS/'data_inventory_exact_v3.json');assert inventory['status']=='complete'
    runtime=read(REPORTS/'STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json')
    alignment=read(REPORTS/'STEP1_DATA_SEAL_V3_ALIGNMENT.json')
    assert alignment['parent_seal_sha256']==sha(REPORTS/'STEP1_DATA_SEAL_V3_RUNTIME_VOCAB.json')
    for relative,digest in alignment['artifacts'].items():
        path=ROOT/relative;assert sha(path)==digest,relative
        DEPENDENCIES[str(path)]=digest
    manifest=read(ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3/manifest.json')
    components={image:index for index,group in enumerate(manifest['components']) for image in group}
    assert len(components)==4013 and len(manifest['components'])==3920
    assert not set(components)&set(manifest['excluded_component_images'])
    assert len(manifest['excluded_component_images'])==70
    outer_counts=Counter();folds={}
    roles=('train','model_val','calibration','test')
    for fold in range(5):
        folder=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3'/('fold_'+str(fold))
        detector=read(folder/'detector_split_v3.json')
        image_roles={};group_roles={};counts={}
        for role in roles:
            path=folder/('vqa_'+role+'.json');rows=read(path)
            info=manifest['folds'][str(fold)][role]
            assert sha(path)==info['sha256'] and len(rows)==info['qa']
            images={row['image_id'] for row in rows};assert len(images)==info['images']
            assert images==set(detector['split_images'][role])
            image_roles[role]=images;group_roles[role]={components[i] for i in images}
            counts[role]={'images':len(images),'QA':len(rows),'components':len(group_roles[role])}
        for i,left in enumerate(roles):
            for right in roles[i+1:]:
                assert not image_roles[left]&image_roles[right]
                assert not group_roles[left]&group_roles[right]
        assert set.union(*image_roles.values())==set(components)
        assert all(len({detector['component_by_image'][i] for i in group})==1 for group in manifest['components'])
        assert len({detector['component_by_image'][group[0]] for group in manifest['components']})==3920
        outer_counts.update(image_roles['test']);folds[str(fold)]=counts
    assert set(outer_counts)==set(components) and set(outer_counts.values())=={1}
    fingerprint_path=REPORTS/'perceptual_fingerprints_v3.jsonl'
    DEPENDENCIES[str(fingerprint_path)]=sha(fingerprint_path)
    fingerprints={}
    for line in fingerprint_path.open():
        row=json.loads(line)
        if 'phash' in row:fingerprints[row['path']]=int(row['phash'],16)
    neighbours={}
    for stem in ('cross_dataset','within_pmc','within_roco'):
        progress=read(REPORTS/(stem+'_neighbour_progress_v3.json'))
        assert progress['status']=='complete' and progress['processed']==progress['total']
        path=REPORTS/(stem+'_hamming_candidates_v3.jsonl');DEPENDENCIES[str(path)]=sha(path)
        candidates=[json.loads(line) for line in path.open()];keys=set()
        for pair in candidates:
            distance=(fingerprints[pair['pmc']]^fingerprints[pair['roco']]).bit_count()
            assert distance==pair['hamming'] and distance<=8
            keys.add((pair['pmc'],pair['roco'],distance))
        assert len(candidates)==progress['candidate_pairs']
        edges=read(REPORTS/(stem+'_pixel_edges_v3.json'))
        assert edges['status']=='complete' and edges['candidate_pairs']==len(candidates)
        for edge in edges['selected_edges']:
            assert (edge['pmc'],edge['roco'],edge['hamming']) in keys
            assert edges['correlation_min']<=edge['correlation']<=1.000001
        neighbours[stem]={'candidate_pairs':len(candidates),'selected_edges':len(edges['selected_edges']),
            'saved_Hamming_arithmetic_recomputed':True,'pixel_correlation_not_recomputed_this_audit':True}
    transitive=read(REPORTS/'independent_transitive_check_v3.json')
    assert transitive['status']=='pass' and transitive['shared_components']==0
    # This rebinds the prior independent union-find report; it does not pretend
    # to reconstruct unknown patients or rerun raw-image correlation here.
    upstream={}
    for name,report_name,checkpoint_dir in [
        ('ROCO','roco_alignment_execution_v3/independent_verification.json','revised_roco_alignment_v3'),
        ('PMC','pmc_alignment_execution_v3/independent_artifact_verification.json','revised_pmcvqa_alignment_v3')]:
        audit=read(REPORTS/report_name);assert audit['passed'] and all(audit['checks'].values())
        progress=read(REPORTS/report_name.split('/')[0]/'progress.json')
        target=ROOT/'refactor_v2/checkpoints'/checkpoint_dir
        for kind in ('best','last'):
            path=target/(kind+'.pt');digest=sha(path)
            assert digest==progress[kind+'_checkpoint_sha256']
            DEPENDENCIES[str(path)]=digest
        upstream[name]={'artifact_verification_passed':True,'best_sha256':sha(target/'best.pt'),
            'quality_gate_passed':audit.get('quality_gate_passed'),
            'interpretation':'Technical lineage only, not independent clinical benefit'}
    labels=read(REPORTS/'pmc_label_integrity_v3.json');assert labels['accepted']==28490 and len(labels['quarantined'])==3
    outcome={'passed':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'No-model artifact reconciliation, role/component parity and saved candidate-link arithmetic',
        'source_script_sha256':sha(Path(__file__).resolve()),'dependencies_sha256':DEPENDENCIES,
        'alignment_seal_artifacts_verified':len(alignment['artifacts']),
        'runtime_seal_artifacts':len(runtime['artifacts']),'folds':folds,
        'all5fold_detector_VQA_role_image_sets_exact':True,'all5fold_role_components_disjoint':True,
        'all4013_retained_images_exactly_once_outer_test':True,
        'known_neighbour_artifacts':neighbours,'prior_transitive_report_rebound_no_known_shared_components':True,
        'upstream_checkpoint_hashes_reverified':upstream,
        'historical_PMC_checkpoint_eligible':False,'PMC_train_lon_independent_evidence':False,
        'PMC_train_label_valid_QA':28490,'PMC_train_malformed_quarantined':3,
        'model_instantiations':0,'new_training_updates':0,'reserved_role_model_inference':False,
        'limitations':['No new exhaustive similarity search or raw-pixel correlation rerun',
            'Transitive separation report is rebound, not recomputed here',
            'Grouping is not verified patient identity; base-model pretraining independence unknown',
            'ROCO is technically verified, not isolated pretraining efficacy; PMC quality remains failed',
            'Existing FracAtlas evaluation is development-informed; generated folds are not executed nested confirmation'],
        'next_required':'Define evidence-backed step3 exit and honest confirmation protocol; no automatic long training',
        'long_training_authorized':False,'step3_complete':False,'production_released':False}
    OUT.write_text(json.dumps(outcome,indent=2)+'\n')
    print(json.dumps({k:v for k,v in outcome.items() if k not in ('dependencies_sha256','folds')},indent=2))


if __name__=='__main__':main()
