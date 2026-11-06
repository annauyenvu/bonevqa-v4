"""Measure grad-enabled/public-no_grad discrepancy; retain exact guard."""
from pathlib import Path
import importlib.util
import hashlib

SOURCE_TRACE=Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/experiments/probe_fullfold0_fracture_state_trace_v3_3.py')
assert hashlib.sha256(SOURCE_TRACE.read_bytes()).hexdigest()=='581fde0f8b2628da36257e504270c93535804232cc86d2d2c83f2608d2859e82'
spec=importlib.util.spec_from_file_location('preserved_trace_v3_3',SOURCE_TRACE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SOURCE_BASE=module.SOURCE_BASE;REPAIR_SOURCE=module.REPAIR_SOURCE
text=module.text
replacements={
    "OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_state_trace_v3_3'":
        "OUT=ROOT/'refactor_v2/reports/fullfold0_probability_trace_v3_4'",
    "    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])":
        "    deps[str(SOURCE_TRACE)]=sha(SOURCE_TRACE)\n    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])",
    "        assert manual_probs==public_probs":
        "        dump(OUT/'probability_comparison.json',{'mode':mode,'manual_grad_enabled':manual_probs,'public_no_grad':public_probs,'absolute_differences':[abs(a-b) for a,b in zip(manual_probs,public_probs)],'same_raw0_5_decisions':[a>=.5 for a in manual_probs]==[b>=.5 for b in public_probs],'all263_state_hashes_unchanged':digest(model)==before,'physical_checkpoint_unchanged':sha(info['path'])==info['sha256'],'model_training':model.training})\n        assert manual_probs==public_probs"}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(SOURCE_BASE),'exec'),globals())
