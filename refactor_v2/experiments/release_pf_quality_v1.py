"""Release bounded four-arm execution only after all frozen checks pass."""
import ast
import json
import shutil
from pathlib import Path
from experiments.lapa_integrated_gate_v1 import REPORTS
from experiments.prepare_lapa_pilot_v1 import sha
from experiments.pf_comparison_v1 import ARMS

OUT=REPORTS/'pf_comparison_quality_v1'
HERE=Path(__file__).parent


def main():
    prep=REPORTS/'pf_comparison_preparation_v1'
    dependencies=json.loads((prep/'preparation_seal.json').read_text())['dependencies']
    for path,h in dependencies.items():assert sha(Path(path))==h,path
    evaluator=REPORTS/'pf_evaluator_check_v1/receipt.json'
    receipt=json.loads(evaluator.read_text())
    assert receipt['metrics']['records']==1432 and receipt['metrics']['fracture_records']==321
    for path,h in receipt['hashes'].items():assert sha(Path(path))==h
    # The exact full evaluator checked above is retained by the new worker.
    def funcs(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('evaluate','metrics')}
    assert funcs(HERE/'pf_quality_worker_v1.py')==funcs(HERE/'lapa_quality_worker_v1.py')
    size=max((REPORTS/'pf_comparison_gates_v2'/a/'trainable_state.pt').stat().st_size for a in ARMS)
    reserve=8*1024**3
    estimate=len(ARMS)*size+512*1024**2
    free=shutil.disk_usage(REPORTS).free
    assert free-estimate>reserve,(free,estimate,reserve)
    extras=[HERE/n for n in ['pf_quality_worker_v1.py','operate_pf_quality_v1.py','release_pf_quality_v1.py','check_pf_evaluator_v1.py']]
    extras += [prep/'preparation_seal.json',evaluator,REPORTS/'pf_evaluator_check_v1/rows.json']
    dependencies.update({str(p):sha(p) for p in extras})
    OUT.mkdir(exist_ok=False)
    with (OUT/'execution_release.json').open('x') as f:json.dump({'quality_released':True,'arms':list(ARMS),
        'epochs':4,'updates_per_arm':292,'seed':42,'dependencies':dependencies,
        'disk_free_before':free,'estimated_new_output_bytes':estimate,'disk_reserve':reserve,
        'scope':'Reconstructed PF variants/shared backbone; not native LaPA'},f,indent=2)
    print(json.dumps({'quality_released':True,'dependency_count':len(dependencies),'estimated_new_bytes':estimate,'free_bytes':free}))


if __name__=='__main__':main()
