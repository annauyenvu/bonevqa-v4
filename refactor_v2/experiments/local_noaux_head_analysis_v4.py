"""Role-exact batch contract: mixed model-val/test, canonical calibration."""
from pathlib import Path
base=Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/experiments/local_noaux_head_analysis_v3.py')
source=base.read_text()
changes={
"if role=='train':return _extract(model,fold,arm,role)":
"""if role=='train':return _extract(model,fold,arm,role)
    if role=='calibration':
        x,rows=_extract(model,fold,arm,role)
        CACHE.clear();CACHE.extend((x[i:i+4],list(range(len(x[i:i+4])))) for i in range(0,len(x),4))
        return x,rows""",
"rows=_predict(model,fold,arm,role,canonical=False,**kwargs)":
"rows=_predict(model,fold,arm,role,canonical=(role=='calibration'),**kwargs)",
"fp=dest/(role+'_features.pt');assert not fp.exists();torch.save({'features':x,'original_rows':original,'batch_contract':'mixed_QA_batch4'},fp)":
"""fp=dest/(role+'_features.pt')
        if fp.exists():
            saved=torch.load(fp,map_location='cpu',weights_only=False)
            assert torch.equal(saved['features'],x) and saved['original_rows']==original
        else:torch.save({'features':x,'original_rows':original,'batch_contract':'role_exact_batch4'},fp)""",
"h.dump(dest/(role+'_predictions.json'),{'records':full},True);counts[role]=len(full)":
"""predpath=dest/(role+'_predictions.json')
        if predpath.exists():assert h.read(predpath)=={'records':full}
        else:h.dump(predpath,{'records':full},True)
        counts[role]=len(full)""",
"dest=O/'postprocess_v3'":"dest=O/'postprocess_v4'",
"postprocess_v3_failure.json":"postprocess_v4_failure.json",
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(base),'exec'),globals())
