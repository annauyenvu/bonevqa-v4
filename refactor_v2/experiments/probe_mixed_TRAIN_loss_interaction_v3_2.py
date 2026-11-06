"""Preserved v3_1: only detached gradient cosine reductions use CPU float64."""
from pathlib import Path
import hashlib

SOURCE_BASE=Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/experiments/probe_mixed_TRAIN_loss_interaction_v3_1.py')
text=SOURCE_BASE.read_text()
replacements={
    "OUT=ROOT/'refactor_v2/reports/mixed_TRAIN_loss_interaction_v3_1'":
        "OUT=ROOT/'refactor_v2/reports/mixed_TRAIN_loss_interaction_v3_2'",
    "    OUT.mkdir();dump(OUT/'protocol.json',":
        "    deps[str(SOURCE_BASE)]=sha(SOURCE_BASE)\n    OUT.mkdir();dump(OUT/'protocol.json',",
    "        denominator=float((a.norm()*v.norm()).cpu())\n        return float((a.flatten()*v.flatten()).sum().cpu())/denominator if denominator else None":
        "        a64=a.detach().cpu().double().flatten();v64=v.detach().cpu().double().flatten()\n        denominator=float(a64.norm()*v64.norm())\n        return float((a64*v64).sum())/denominator if denominator else None",
    "'mode':'eval_autograd_dropoutoff;manual loss algebra same as TRAIN adapter, not stochastic training-mode audit',":
        "'mode':'eval_autograd_dropoutoff;manual loss algebra same as TRAIN adapter, not stochastic training-mode audit','cosine_reduction':'DetachedCPUfloat64;no model/loss/gradientprecisionchange;rawvaluesnotclamped',"}
for old,new in replacements.items():
    assert text.count(old)==1,old
    text=text.replace(old,new)
exec(compile(text,str(SOURCE_BASE),'exec'),globals())
