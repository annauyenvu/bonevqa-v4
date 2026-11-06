"""Release only once both actual scheduled gates/replays and dependencies pass."""
import importlib
import json
from pathlib import Path
import shutil
import sys
from experiments.lens_bounded_route_v1 import ROOT,PREP,QUALITY,sha


def main():
    QUALITY.mkdir(exist_ok=False)
    prep=json.loads((PREP/'preparation_seal.json').read_text());assert prep['complete']
    dependencies=dict(prep['dependencies'])
    for arm in ['ce_only','lens_losses']:
        folder=ROOT/'reports/lens_bounded_optimizer_gates_v1'/arm
        update=json.loads((folder/'update.json').read_text());replay=json.loads((folder/'replay.json').read_text())
        assert update['updates']==2 and update['microbatches']==8 and update['frozen_invariance'] and replay['exact']
        assert update['initial_full_state_digest']==json.loads((PREP/'trainability.json').read_text())['full_state_digest']
        assert update['full_state_digest']==replay['full_state_digest']
        assert all(sha(Path(p))==h for p,h in update['dependencies'].items())
        for name in ['started.json','update.json','replay.json']:dependencies[str(folder/name)]=sha(folder/name)
    for file in ['lens_bounded_quality_worker_v1.py','release_lens_bounded_quality_v1.py','operate_lens_bounded_quality_v1.py']:
        p=ROOT/'experiments'/file;dependencies[str(p)]=sha(p)
    dependencies[str(PREP/'preparation_seal.json')]=sha(PREP/'preparation_seal.json')
    lookup=ROOT/'reports/step4_localization_v3_1/component_lookup.json';dependencies[str(lookup)]=sha(lookup)
    # Bind actual installed implementation modules used by this native route, not merely version labels.
    for module in ['transformers.models.idefics3.modeling_idefics3','transformers.models.idefics3.processing_idefics3',
        'transformers.models.idefics3.image_processing_idefics3','peft.tuners.lora.layer','torch.optim.adamw']:
        p=Path(importlib.import_module(module).__file__);dependencies[str(p)]=sha(p)
    assert all(sha(Path(p))==h for p,h in dependencies.items())
    free=shutil.disk_usage(ROOT).free;estimated_output=256*1024**2
    assert free-estimated_output>=8*1024**3
    (QUALITY/'execution_release.json').write_text(json.dumps({'quality_released':True,'dependencies':dependencies,
        'arms':['ce_only','lens_losses'],'fresh_initialization':'shared_PREupdate updates0, never gate weights',
        'wall_cap_seconds':14400,'rss_cap_bytes':20*1024**3,'disk_reserve_bytes':8*1024**3,
        'estimated_output_bytes':estimated_output,'free_bytes_at_release':free,'python':sys.executable,
        'scope':'bounded adapted native-asset Lens engineering pilot; loss contrast conditional on same three-view training, not exact original reproduction'},indent=2))


if __name__=='__main__':main()
