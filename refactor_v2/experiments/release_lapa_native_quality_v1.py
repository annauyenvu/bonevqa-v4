"""Verify native pilot's prerequisites and bounded output budget before release."""
import ast
import json
import shutil
from pathlib import Path
from experiments.lapa_integrated_gate_v1 import REPORTS
from experiments.lapa_native_pilot_v2 import ARMS
from experiments.prepare_lapa_pilot_v1 import sha

OUT=REPORTS/'lapa_native_quality_v1'


def main():
    prep=REPORTS/'lapa_native_preparation_v1'
    dependencies=json.loads((prep/'preparation_seal.json').read_text())['dependencies']
    for p,h in dependencies.items():assert sha(Path(p))==h,p
    path=REPORTS/'lapa_native_evaluator_v1/receipt.json';evaluator=json.loads(path.read_text())
    assert evaluator['metrics']['records']==1432 and evaluator['metrics']['fracture_records']==321
    for p,h in evaluator['hashes'].items():assert sha(Path(p))==h
    def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('evaluate','metrics')}
    scripts=Path(__file__).parent
    assert functions(scripts/'lapa_native_quality_worker_v1.py')==functions(scripts/'pf_quality_worker_v1.py')
    summary=json.loads((prep/'gate_summary.json').read_text())
    expected=sum(r['checkpoint_bytes'] for r in summary['gates'].values())+512*1024**2
    free=shutil.disk_usage(REPORTS).free;reserve=8*1024**3
    assert free-expected>reserve,(free,expected,reserve)
    extra=[prep/'preparation_seal.json',path,REPORTS/'lapa_native_evaluator_v1/rows.json']
    extra += [scripts/n for n in ['lapa_native_quality_worker_v1.py','check_lapa_native_evaluator_v1.py','release_lapa_native_quality_v1.py','operate_lapa_native_quality_v1.py']]
    dependencies.update({str(p):sha(p) for p in extra})
    OUT.mkdir(exist_ok=False)
    (OUT/'execution_release.json').write_text(json.dumps({'quality_released':True,'arms':list(ARMS),
            'epochs':4,'updates_per_arm':292,'dependencies':dependencies,'free_bytes_before':free,
            'estimated_new_output_bytes':expected,'reserve_bytes':reserve,
            'scope':'native-pretrained frozen-encoder BoneVQA pilot with reconstructed graph409'},indent=2))
    print(json.dumps({'quality_released':True,'dependencies':len(dependencies),'estimated_bytes':expected,'free':free}))


if __name__=='__main__':main()
