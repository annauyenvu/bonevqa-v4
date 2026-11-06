"""Fresh updates-zero initialization after actual scheduled AdamW gate."""
import json
from pathlib import Path
import platform
import shutil
import torch
import transformers
from experiments.favp_pilot_route_v1 import PilotRoute,ROOT,INPUTS
from experiments.lapa_integrated_gate_v1 import digest
from experiments.prepare_lapa_pilot_v1 import sha

OUT=ROOT/'reports/favp_bounded_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(8);torch.manual_seed(42)
    gate=ROOT/'reports/favp_pilot_optimizer_gate_v1'
    update=json.loads((gate/'update.json').read_text());replay=json.loads((gate/'replay.json').read_text())
    assert update['updates']==2 and all(update['changed_groups'].values())
    assert update['frozen_versions_unchanged'] and replay['exact_logits'] and replay['exact_loss']
    model=PilotRoute()
    initial={g:digest(ps) for g,ps in model.groups().items()}
    # Read gate metadata only to compare fresh initialization, never load updates.
    prior=torch.load(gate/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert initial==prior['initial_group_digests']
    del prior
    names={n for n,p in model.named_parameters(remove_duplicate=False) if p.requires_grad}
    state=model.state_dict();mutable={n:t for n,t in state.items() if n in names}
    required=sum(t.numel()*t.element_size() for t in mutable.values())
    assert shutil.disk_usage(OUT).free-2*required>8*1024**3
    fixed=digest((n,t) for n,t in state.items() if n not in names)
    torch.save({'mutable':mutable,'initial_group_digests':initial,'mutable_digest':digest(mutable.items()),
        'fixed_digest':fixed,'updates':0,'seed':42},OUT/'shared_PREupdate.pt')
    config={'seed':42,'epochs':2,'updates':64,'batch':1,'train':32,'model_val':8,
        'lr':3e-5,'min_lr':1e-5,'warmup_lr':1e-6,'warmup_steps':6,'weight_decay':.05,
        'betas':[.9,.999],'eps':1e-8,'clip':1.,'checkpoint':'LAST2','max_new_tokens':20,
        'initialization':str(OUT/'shared_PREupdate.pt'),'selection':str(INPUTS/'selection_and_schedules.json'),
        'wall_budget_seconds':14400,'quality_released':False}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    (OUT/'trainability.json').write_text(json.dumps({'groups':{g:sum(p.numel() for n,p in ps) for g,ps in model.groups().items()},
        'initial_group_digests':initial,'mutable_bytes':required,'fixed_digest':fixed,'audit':model.audit},indent=2))
    deps=json.loads((ROOT/'reports/lapa_native_quality_v1/execution_release.json').read_text())['dependencies']
    for p,h in deps.items():assert sha(Path(p))==h,p
    inventory_path=ROOT/'reports/native_asset_acquisition_closure_v1/inventory.json'
    inventory=json.loads(inventory_path.read_text())
    for package in inventory['packages']:
        if package['package'].startswith('FAVP_'):
            for f in package['files']:
                p=Path(f['path']);assert sha(p)==f['sha256'];deps[str(p)]=f['sha256']
    source=json.loads((ROOT/'reports/vl_author_sources_v1/source_manifest.json').read_text())['sources']['FAVP']['sha256']
    for p,h in source.items():assert sha(Path(p))==h;deps[p]=h
    files=[Path(__file__),OUT/'shared_PREupdate.pt',OUT/'config.json',OUT/'trainability.json',inventory_path,
        ROOT/'reports/FAVP_BOUNDED_ADAPTED_PILOT_PROTOCOL_V2.md',ROOT/'reports/step4_localization_v3_1/component_lookup.json',
        ROOT/'reports/favp_train_proposal_census_v1/receipt.json',ROOT/'reports/favp_train_proposal_census_v1/frozen_census.json',
        gate/'update.json',gate/'replay.json',ROOT/'reports/favp_public_generation_v2/receipt.json']
    files+=list(INPUTS.glob('*.json'))
    files+=list((ROOT/'reports/favp_train_proposal_census_v1').glob('IMG*.json'))
    for n in ['favp_pilot_route_v1.py','favp_trainable_route_v3.py','favp_visual_bridge_v2.py','favp_vicuna_loader_v3.py',
              'check_favp_cpu_compute_v1.py','check_favp_projector_update_v1.py','check_favp_public_generation_v2.py',
              'favp_circle_cache_v1.py','favp_generation_compat_v1.py','favp_bounded_quality_worker_v1.py',
              'operate_favp_bounded_quality_v1.py','analyze_favp_bounded_pilot_v1.py','release_favp_bounded_pilot_v1.py']:
        files.append(ROOT/'experiments'/n)
    deps.update({str(p):sha(p) for p in files})
    (OUT/'preparation_seal.json').write_text(json.dumps({'dependencies':deps,'quality_released':False,
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'threads':8}},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
