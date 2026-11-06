"""Verify frozen preparation before the one bounded adapted quality pilot."""
import json
from pathlib import Path
import shutil
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.prepare_lapa_pilot_v1 import sha

OUT=ROOT/'reports/favp_bounded_quality_v1'


def main():
    OUT.mkdir(exist_ok=False)
    prep=ROOT/'reports/favp_bounded_preparation_v1'
    seal=json.loads((prep/'preparation_seal.json').read_text());deps=seal['dependencies']
    for p,h in deps.items():assert sha(Path(p))==h,p
    gate=json.loads((ROOT/'reports/favp_pilot_optimizer_gate_v1/replay.json').read_text())
    assert gate['exact_logits'] and gate['exact_loss']
    required=json.loads((prep/'trainability.json').read_text())['mutable_bytes']+10*1024**2
    free=shutil.disk_usage(OUT).free;assert free-required>8*1024**3
    deps[str(prep/'preparation_seal.json')]=sha(prep/'preparation_seal.json')
    (OUT/'execution_release.json').write_text(json.dumps({'quality_released':True,'scope':'one resource-bounded adapted native-asset pilot, not original FAVP reproduction',
        'dependencies':deps,'expected_updates':64,'expected_validation_records':8,'wall_budget_seconds':14400,
        'disk_free_bytes':free,'estimated_remaining_output_bytes':required,'reserve_bytes':8*1024**3},indent=2))


if __name__=='__main__':main()
