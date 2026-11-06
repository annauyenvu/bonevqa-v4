"""Exact within-public-forward route check; cross-mode delta is descriptive.

Preserve v3_1..4 failures. No tolerance added to any equality assertion.
"""
from pathlib import Path
import importlib.util
import hashlib

SOURCE_PROB=Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/experiments/probe_fullfold0_probability_trace_v3_4.py')
assert hashlib.sha256(SOURCE_PROB.read_bytes()).hexdigest()=='ddddc4e3a4cfe60cba2e732e19a78cc9bb4745a15e165470e279447df3fb3097'
spec=importlib.util.spec_from_file_location('preserved_probability_trace',SOURCE_PROB)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SOURCE_BASE=module.SOURCE_BASE;REPAIR_SOURCE=module.REPAIR_SOURCE;SOURCE_TRACE=module.SOURCE_TRACE
text=module.text
replacements={
    "OUT=ROOT/'refactor_v2/reports/fullfold0_probability_trace_v3_4'":
        "OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_5'",
    "    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])":
        "    deps[str(SOURCE_PROB)]=sha(SOURCE_PROB)\n    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])",
    "'Public p_yes equals manual fracture-head sigmoid, all263weights unchanged'":
        "'Within-public-no_grad-forward p_yes equals hook-captured fracture sigmoid exactly; cross grad/no_grad delta recorded separately; all263weights unchanged'",
    "        with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})":
        "        public_capture={}\n        def public_hook(module,args,output):public_capture['logit']=output.detach()\n        public_handle=model.fracture_head.register_forward_hook(public_hook)\n        with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})\n        public_handle.remove()\n        manual_public_probs=torch.sigmoid(public_capture['logit']).tolist()",
    "        assert manual_probs==public_probs":
        "        assert manual_public_probs==public_probs",
    "            'public_probabilities':public_probs,'public_sigmoid_exact':True,'all263weights_unchanged':True,":
        "            'public_probabilities':public_probs,'public_sigmoid_exact':True,'all263weights_unchanged':True,\n            'grad_enabled_probabilities':manual_probs,'cross_mode_max_abs_error':max(abs(a-b) for a,b in zip(manual_probs,public_probs)),'same_raw0_5_cross_mode_decisions':[a>=.5 for a in manual_probs]==[b>=.5 for b in public_probs],"}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(SOURCE_BASE),'exec'),globals())
