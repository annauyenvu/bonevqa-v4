"""Bind passed gates and fresh pre-update mutable state; never optimize."""
import json
import shutil
from pathlib import Path
import torch
from experiments.gate_lapa_native_scheduled_v2 import initialize,ROOT as GATES
from experiments.lapa_native_pilot_v2 import ARMS
from experiments.lapa_integrated_gate_v1 import REPORTS,digest
from experiments.prepare_lapa_pilot_v1 import sha

OUT=REPORTS/'lapa_native_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(8)
    model=initialize('pf_author');state=model.state_dict();initial=digest(state.items())
    gates={a:json.loads((GATES/a/'receipt.json').read_text()) for a in ARMS}
    assert all(r['initial_digest']==initial and r['updates']==2 and r['fixed_state_unchanged'] and
               all(r['changed_groups'].values()) and all(r['replay'].values()) for r in gates.values())
    assert all(r['microbatches']==gates[ARMS[0]]['microbatches'] for r in gates.values())
    fixed=lambda n:n.startswith(('backbone.text.','backbone.visual.'))
    mutable={n:t for n,t in state.items() if not fixed(n)}
    required=sum(t.numel()*t.element_size() for t in mutable.values())
    assert shutil.disk_usage(OUT).free-required>8*1024**3
    torch.save({'mutable':mutable,'initial_digest':initial,
                'encoder_digest':digest((n,t) for n,t in state.items() if fixed(n)),
                'seed':42,'updates':0},OUT/'shared_PREupdate.pt')
    schedule=REPORTS/'step4_localization_v3_1/schedules.json';epochs=json.loads(schedule.read_text())
    assert len(epochs)==4 and all(sum(map(len,e))==1164 for e in epochs)
    assert sum((len(e)+3)//4 for e in epochs)==292
    configs={a:{'arm':a,'seed':42,'epochs':4,'updates':292,'batch':4,'accumulation':4,
                'base_lr':5e-6,'MM_multiplier':5,'head_multiplier':50,'warmup_updates':29,'eta':0. if a=='carrier' else .1,
                'clip':1.,'checkpoint':'LAST4','initialization':str(OUT/'shared_PREupdate.pt'),
                'schedule':str(schedule),'quality_released':False} for a in ARMS}
    (OUT/'configs.json').write_text(json.dumps(configs,indent=2))
    (OUT/'gate_summary.json').write_text(json.dumps({'all_five_passed':True,'initial_digest':initial,'gates':gates},indent=2))
    # Reverify every historical source/data binding used by the shared pipeline.
    dependencies=json.loads((REPORTS/'pf_comparison_quality_v1/execution_release.json').read_text())['dependencies']
    for p,h in dependencies.items():assert sha(Path(p))==h,p
    root=REPORTS.parent
    files=[Path(__file__),OUT/'shared_PREupdate.pt',OUT/'configs.json',OUT/'gate_summary.json',schedule,
           REPORTS/'LAPA_NATIVE_BACKBONE_PILOT_PROTOCOL_V1.md',root/'references/native_assets_lapa_v1/m3ae.ckpt']
    source=root/'experiments'
    files += [source/n for n in ['lapa_native_backbone_v1.py','lapa_native_pilot_v1.py','lapa_native_pilot_v2.py','gate_lapa_native_scheduled_v2.py']]
    files += [p for p in (root/'references/native_assets_lapa_v1/roberta-base').iterdir() if p.is_file()]
    for a in ARMS:files += [GATES/a/'receipt.json',GATES/a/'replay.json']
    dependencies.update({str(p):sha(p) for p in files})
    (OUT/'preparation_seal.json').write_text(json.dumps({'dependencies':dependencies,'quality_released':False},indent=2))
    print(json.dumps({'all_five_gates_passed':True,'initial_digest':initial,'dependencies':len(dependencies)}))


if __name__=='__main__':main()
