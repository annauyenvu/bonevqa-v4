"""Preserved probe instrumentation: stage hashes, original invariance retained."""
from pathlib import Path
import hashlib

SOURCE_BASE=Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_1.py')
REPAIR_SOURCE=SOURCE_BASE.with_name('probe_fullfold0_fracture_gradients_v3_2.py')
assert hashlib.sha256(SOURCE_BASE.read_bytes()).hexdigest()=='ae0a2501b340162dee430bdd8e03b21d9c021e07182efce86961f21591a395bd'
assert hashlib.sha256(REPAIR_SOURCE.read_bytes()).hexdigest()=='86114d8e629cf09aa5dd4fc99578c01f6e79e3d88aaedd1156a7be5ea7005b6e'
text=SOURCE_BASE.read_text()
replacements={
    "OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_1'":
        "OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_state_trace_v3_3'",
    "ce=F.cross_entropy(captured['closed_logits'].float(),labels)":
        "ce=F.cross_entropy(captured['closed_logits'].float(),labels,reduction='none').mean()",
    "    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])":
        "    deps[str(SOURCE_BASE)]=sha(SOURCE_BASE)\n    deps[str(REPAIR_SOURCE)]=sha(REPAIR_SOURCE)\n    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])",
    "    results={}\n    for mode,info in plan['targets'].items():":
        "    results={};snapshots=[]\n    for mode,info in plan['targets'].items():",
    "        assert len(before)==263 and model.cfg['fracture_positive_weight']==2110/458":
        "        def snapshot(stage):\n            current=digest(model)\n            snapshots.append({'mode':mode,'stage':stage,'keys':len(current),'added_keys':sorted(set(current)-set(before)),'removed_keys':sorted(set(before)-set(current)),'changed_value_hashes':{k:{'before':before[k],'after':current[k]} for k in current if k in before and current[k]!=before[k]},'checkpoint_sha256':sha(info['path'])})\n            dump(OUT/'state_snapshots.json',snapshots)\n        snapshot('loaded_before_forward')\n        assert len(before)==263 and model.cfg['fracture_positive_weight']==2110/458",
    "        out=model(private,stage=3)":
        "        out=model(private,stage=3)\n        snapshot('after_private_forward')",
    "        errors={n:float((gradients['BCE'][i+2]-gradients['total'][i+2]).abs().max()) for i,n in enumerate(head_names)}":
        "        snapshot('after_autograd')\n        errors={n:float((gradients['BCE'][i+2]-gradients['total'][i+2]).abs().max()) for i,n in enumerate(head_names)}",
    "        with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})":
        "        snapshot('after_gradient_summaries')\n        with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})\n        snapshot('after_public_predict')"}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(SOURCE_BASE),'exec'),globals())
