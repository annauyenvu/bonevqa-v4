"""Compare complete resolved model config using frozen constructor merge rules."""
import copy
from pathlib import Path
from src.models.bonevqa import DEFAULT_MODEL_CFG
from experiments.audit_fresh_exact_upstream_fracatlas_v3_1 import ROOT


if __name__=='__main__':
    source=ROOT/'refactor_v2/experiments/audit_perfold_PREtask_roundtrip_recovery_v3_2.py'
    text=source.read_text()
    replacements={
        "perfold_PREtask_roundtrip_recovery_v3_2.json":"perfold_PREtask_roundtrip_recovery_v3_3.json",
        "        assert all(payload['config'][k]==v for k,v in modelcfg.items())":"        resolved=copy.deepcopy(DEFAULT_MODEL_CFG)\n        for k,v in modelcfg.items():\n            if isinstance(v,dict) and isinstance(resolved.get(k),dict):resolved[k].update(v)\n            else:resolved[k]=v\n        assert payload['config']==resolved"}
    for old,new in replacements.items():
        assert text.count(old)==1,old
        text=text.replace(old,new)
    namespace={'__name__':'resolved_checkpoint_artifact_audit','__file__':str(Path(__file__).resolve()),
        'copy':copy,'DEFAULT_MODEL_CFG':DEFAULT_MODEL_CFG}
    exec(compile(text,str(source),'exec'),namespace)
    namespace['main']()
