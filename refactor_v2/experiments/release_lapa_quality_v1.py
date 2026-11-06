"""Verify bounded pilot prerequisites and seal execution dependencies."""
import json
from pathlib import Path
from experiments.prepare_lapa_pilot_v1 import sha

ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'refactor_v2/reports'
OUT=REPORTS/'lapa_shared_quality_v1'


def main():
    assert not OUT.exists()
    prep=REPORTS/'lapa_pilot_preparation_v1'
    seal=json.loads((prep/'preparation_seal.json').read_text())
    for p,h in seal['hashes'].items():assert sha(Path(p))==h,p
    gate=REPORTS/'lapa_pilot_gates_v1/gate_summary.json'
    gates=json.loads(gate.read_text());assert gates['all_five_gates_passed']
    for arm,h in gates['receipt_hashes'].items():assert sha(gate.parent/arm/'receipt.json')==h
    data=REPORTS/'step4_localization_v3_1/data_seal.json'
    for p,h in json.loads(data.read_text())['artifacts'].items():assert sha(ROOT/p)==h
    evaluator=REPORTS/'lapa_evaluator_check_v1_retry2/receipt.json'
    receipt=json.loads(evaluator.read_text());assert receipt['records']==1432 and receipt['canonical_images']==321 and receipt['finite_logits']
    assert sha(evaluator.parent/'outputs.json')==receipt['outputs_sha256']
    schedule=REPORTS/'step4_localization_v3_1/schedules.json'
    schedules=json.loads(schedule.read_text())
    assert len(schedules)==4 and all(sorted(i for batch in epoch for i in batch)==list(range(1164)) for epoch in schedules)
    paths=set(Path(p) for p in seal['hashes'])
    paths.update(ROOT/p for p in json.loads(data.read_text())['artifacts'])
    paths.update((data,gate,evaluator,evaluator.parent/'outputs.json',prep/'preparation_seal.json',REPORTS/'step4_localization_v3_1/B.json'))
    paths.update((ROOT/'refactor_v2/src').rglob('*.py'))
    paths.update((ROOT/'refactor_v2/experiments').glob('*lapa*.py'))
    paths.update((ROOT/'refactor_v2/references/lapa_author_677faf90').rglob('*.py'))
    paths.add(REPORTS/'lapa_author_source_v1/source_and_fixture_audit.json')
    binding=REPORTS/'lapa_port_backbone_binding_v1/cache_binding.json';paths.add(binding)
    for model in json.loads(binding.read_text())['models']:
        for f in model['files']:
            assert sha(Path(f['resolved_path']))==f['sha256'];paths.add(Path(f['resolved_path']))
    for arm in gates['receipt_hashes']:paths.add(gate.parent/arm/'receipt.json')
    deps={str(p.resolve()):sha(p) for p in sorted(paths)}
    OUT.mkdir()
    with (OUT/'execution_release.json').open('x') as f:json.dump({'quality_released':True,'scope':'Five fixed shared-backbone classifier arms; seed42, LAST4,292updates/arm. Not native LaPA/PF.',
        'validation_check_seconds':receipt['seconds'],'dependencies':deps},f,indent=2)
    print(json.dumps({'released':True,'dependencies':len(deps)}))


if __name__=='__main__':main()
