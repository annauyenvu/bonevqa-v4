"""Bind fresh byte inventory to recorded upstream configs/checkpoints/label filtering."""
import json
import time
from pathlib import Path
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT, sha, dump


def main():
    started = time.perf_counter()
    target = ROOT/'refactor_v2/reports/upstream_byte_label_provenance_v3_1.json'
    assert not target.exists()
    folder = ROOT/'refactor_v2/reports/fresh_exact_upstream_fracatlas_v3_1'
    inventory = json.loads((folder/'report.json').read_text())
    verification = json.loads((folder/'independent_verification.json').read_text())
    assert verification['passed'] and verification['report_sha256'] == sha(folder/'report.json')
    sealpath = ROOT/'refactor_v2/reports/STEP1_DATA_SEAL_V3_ALIGNMENT.json'
    seal = json.loads(sealpath.read_text())
    assert len(seal['artifacts']) == 36
    for name, digest in seal['artifacts'].items():
        assert sha(ROOT/name) == digest, name
    base = ROOT/'.evaluation_runtime/data/pretraining_joint_groups_v3'
    source = base/'pmc_vqa_train.json'; filtered = base/'pmc_vqa_train_label_filtered.json'
    labelpath = ROOT/'refactor_v2/reports/pmc_label_integrity_v3.json'
    label = json.loads(labelpath.read_text())
    assert sha(source) == label['source_sha256'] and sha(filtered) == label['output_sha256']
    before, after = json.loads(source.read_text()), json.loads(filtered.read_text())
    quarantine = label['quarantined']
    assert len(before) == 28493 and len(after) == label['accepted'] == 28490 and len(quarantine) == 3
    assert [row for row in before if row not in quarantine] == after
    assert all(sum(r == bad for r in before) == 1 for bad in quarantine)
    upstream = {}; configs = {}; hashes = {}
    for dataset, directory, auditname in [('roco','roco_alignment_execution_v3','independent_verification.json'),
                                         ('pmc_vqa','pmc_alignment_execution_v3','independent_artifact_verification.json')]:
        f = ROOT/'refactor_v2/reports'/directory
        configpath = f/'released_config.json'; progresspath = f/'progress.json'; auditpath = f/auditname
        cfg, progress, audit = [json.loads(p.read_text()) for p in (configpath, progresspath, auditpath)]
        assert audit['passed'] and sha(configpath) == progress['config_sha256']
        train = Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train'])
        val = Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['model_val'])
        assert sha(train) == seal['artifacts'][str(train.relative_to(ROOT))]
        assert sha(val) == seal['artifacts'][str(val.relative_to(ROOT))]
        key = 'clean_pretraining/'+train.name
        assert sha(train) == inventory['datasets'][key]['manifest_sha256']
        for which in ('best','last'):
            p = Path(progress['checkpoint_dir'])/(which+'.pt')
            hashes[dataset+'/'+which] = sha(p)
            assert hashes[dataset+'/'+which] == progress[which+'_checkpoint_sha256']
        assert hashes[dataset+'/best'] == audit['best_checkpoint_sha256']
        configs[dataset] = cfg
        upstream[dataset] = {'config_sha256':sha(configpath), 'driver_progress_sha256':sha(progresspath),
            'prior_independent_artifact_audit_sha256':sha(auditpath),
            'train_manifest_sha256':sha(train),'train_rows':len(json.loads(train.read_text())),
            'model_val_manifest_sha256':sha(val),'model_val_rows':len(json.loads(val.read_text()))}
    assert configs['roco']['init_from'] is None
    assert sha(ROOT/configs['pmc_vqa']['init_from']) == hashes['roco/best']
    result = {'passed':True,'script_sha256':sha(Path(__file__).resolve()),
        'fresh_inventory_sha256':sha(folder/'report.json'),
        'fresh_inventory_independent_verification_sha256':sha(folder/'independent_verification.json'),
        'alignment_seal_sha256':sha(sealpath),'alignment_seal_artifacts_rehashed':36,
        'upstream':upstream,'checkpoint_bytes_sha256':hashes,
        'initialization_chain': 'Recorded ROCO init_from=null, PMC init_from=verified new ROCO BEST; no historical PMC checkpoint introduced by these configs',
        'PMC_structural_label_filter':{'report_sha256':sha(labelpath),'before':28493,'after':28490,
            'quarantined':3,'retained_records_order_and_all_existing_fields_unchanged':True,
            'clinical_label_correctness_adjudicated':False},
        'historical_PMC_checkpoint_provenance': 'Not certified by this audit; retain ineligible status',
        'PMC_train_lon_test_shared_sha256_images':inventory['PMC_train_lon_test_shared_byte_hashes'],
        'PMC_quality_gate': 'Previously failed 124/1500 OPEN exact; byte/provenance pass does not release downstream training',
        'limitations':['Recorded configs/artifact lineage checked, no new model-state reload or training replay',
                      'Base-model pretraining exposure and patient identity remain uncertified',
                      'No independent holdout or crop/rotation invariance certification'],
        'elapsed_seconds':time.perf_counter()-started,'model_calls':0,'training_updates':0,
        'manifest_changes':0,'long_training_authorized':False,'step3_complete':False,
        'Final_Solution_selected':False,'production_released':False}
    dump(target,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
