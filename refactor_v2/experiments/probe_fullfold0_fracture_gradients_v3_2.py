"""Preserved-v3_1 repair: exact frozen per-example CE reduction then mean.

No loss/protocol tolerance relaxation. Original failed run/source untouched.
"""
from pathlib import Path
import hashlib

SOURCE_BASE=Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/experiments/probe_fullfold0_fracture_gradients_v3_1.py')
assert hashlib.sha256(SOURCE_BASE.read_bytes()).hexdigest()=='ae0a2501b340162dee430bdd8e03b21d9c021e07182efce86961f21591a395bd'
text=SOURCE_BASE.read_text()
replacements={
    "OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_1'":
        "OUT=ROOT/'refactor_v2/reports/fullfold0_fracture_gradient_probe_v3_2'",
    "ce=F.cross_entropy(captured['closed_logits'].float(),labels)":
        "ce=F.cross_entropy(captured['closed_logits'].float(),labels,reduction='none').mean()",
    "    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])":
        "    deps[str(SOURCE_BASE)]=sha(SOURCE_BASE)\n    deps.update(read(PARENT/'runtime_binding.json')['dependency_sha256'])"}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(SOURCE_BASE),'exec'),globals())
