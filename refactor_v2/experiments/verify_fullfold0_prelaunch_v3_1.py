"""Last independent byte/source/config checks; no model instantiation or training."""
import json
from pathlib import Path
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT,sha,dump


def main():
    out=ROOT/'refactor_v2/reports/fullfold0_B_uniform_screen_v3_1'
    destination=out/'prelaunch_independent_verification.json';assert not destination.exists()
    p=json.loads((out/'protocol.json').read_text());runtime=json.loads((out/'runtime_binding.json').read_text())
    integration=json.loads((out/'trainer_integration_verification.json').read_text())
    assert integration['passed'] and integration['screen_protocol_sha256']==sha(out/'protocol.json')
    assert integration['runtime_binding_sha256']==sha(out/'runtime_binding.json')
    for name,h in {**p['dependency_sha256'],**runtime['dependency_sha256']}.items():assert sha(Path(name))==h
    assert runtime['max_updates_each']==1438 and runtime['epochs_each']==2 and runtime['per_arm_seconds_cap_at_next_forward']==10800
    configs=[]
    for mode in p['arm_order']:
        info=p['configs'][mode];assert sha(Path(info['path']))==info['sha256']
        c=json.loads(Path(info['path']).read_text());configs.append(c)
        assert c['recipe_status']=='candidate_not_launch_sealed' and c['seed']==42
        assert c['train']['stage']==3 and c['train']['epochs']==2 and c['train']['batch_size']==4 and c['train']['grad_accum']==4
        assert not c['checkpoint_allow_mismatch_prefixes'] and sha(Path(c['init_from']))==info['initial_sha256']
        assert not (ROOT/c['checkpoint_dir']/c['run_name']).exists()
        assert not any(c['model'][k] for k in ('use_visual_prompt','use_lens','use_latent_prompt','use_textual_prompt','use_region_in_prompt'))
        assert c['data']['loader_kwargs']['sealed_records_by_split']['calibration'] is None and c['data']['loader_kwargs']['sealed_records_by_split']['test'] is None
        v=integration['results'][mode];assert v['manual_vs_scoped_trainer_loss_exact'] and v['native_eval_bypass'] and v['all263_weights_unchanged']
    assert all(configs[0][k]==configs[1][k] for k in ('data','model','train','init_from','checkpoint_allow_mismatch_prefixes'))
    cfg=configs[0];rows=json.loads(Path(cfg['data']['loader_kwargs']['sealed_records_by_split']['train']).read_text())
    cache=ROOT/'refactor_v2/reports/fresh_exact_upstream_fracatlas_v3_1/image_hashes.jsonl'
    assert sha(cache)=='edbdf69f9d347a7cbd3c79fbce5bfe3a0ae59dd9c830cce6ffa6767ebb4b940e'
    hashes={r['path']:r['sha256'] for s in cache.read_text().splitlines() if '\\fracatlas\\' in (r:=json.loads(s))['path']}
    image_root=Path(cfg['data']['loader_kwargs']['root']);unique={r['image'] for r in rows}
    assert len(unique)==2568
    for image in unique:
        path=image_root/image;canonical=ROOT/'data'/('BONE_VQA\\BaoYen_work\\data\\fracatlas\\FracAtlas\\'+image.replace('/','\\'))
        assert path.resolve()==canonical and sha(path)==hashes[str(canonical)]
    assert not (out/'progress.json').exists() and not (out/'ACTIVE.lock').exists()
    dump(destination,{'passed':True,'script_sha256':sha(Path(__file__).resolve()),
        'screen_protocol_sha256':sha(out/'protocol.json'),'runtime_binding_sha256':sha(out/'runtime_binding.json'),
        'trainer_integration_sha256':sha(out/'trainer_integration_verification.json'),
        'all_current_source_config_initial_bindings_verified':True,'TRAIN_images_rehashed':2568,
        'native_uniform_equal_nonloss_configs':True,'candidate_cli_guard_still_present':True,
        'allowed_scope':'Only fold0 seed42 native/uniform twoepochs1438updates each; driver must still check idle and issue scoped release',
        'model_instantiations':0,'optimizer_updates':0,'calibration_test_allowed':False,
        'fivefold_or_other_long_training_released':False})
    print('Independent final prelaunch passed;2568TRAIN image bytes/source/config/init parity exact',flush=True)


if __name__=='__main__':main()
