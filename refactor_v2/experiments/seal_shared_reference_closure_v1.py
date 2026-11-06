import json
from pathlib import Path
from experiments.prepare_lapa_pilot_v1 import sha

OUT=Path(__file__).resolve().parents[1]/'reports/lapa_shared_quality_v1'
release=json.loads((OUT/'execution_release.json').read_text())
for p,h in release['dependencies'].items():assert sha(Path(p))==h
assert json.loads((OUT/'execution_completed.json').read_text())['full_replays']==5
paths=[OUT/n for n in ['execution_release.json','execution_completed.json','paired_component_bootstrap.json','REFERENCE_PILOT_RESULTS.md','FINAL_INTERPRETATION.md','learning_curves.png']]
for arm in ['carrier','train_cs0','train_cs1','random_cs0','random_cs1']:
    assert json.loads((OUT/arm/'replay.json').read_text())['exact']
    assert json.loads((OUT/arm/'independent_metric_audit.json').read_text())['saved_logits_recount_passed']
    paths.extend(OUT/arm/n for n in ['completed.json','replay.json','independent_metric_audit.json','epoch1.json','epoch2.json','epoch3.json','epoch4.json','LAST.pt'])
with (OUT/'closure_seal.json').open('x') as f:json.dump({'bounded_pilot_complete':True,'figure_visually_reviewed':True,'native_reproduction_complete':False,'production_ready':False,'hashes':{str(p):sha(p) for p in paths}},f,indent=2)
print('Verified and sealed',len(paths),'artifacts')
