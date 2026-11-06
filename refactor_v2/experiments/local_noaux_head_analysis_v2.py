"""Batch-consistent head prediction repair before any head fitting starts.

Original scientific configuration, 10 head epochs and strict exact replay remain.
The v1 source and release are preserved byte-for-byte.
"""
import hashlib,json
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA')
O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
amendment=json.loads((O/'head_batch_consistency_v2/release.json').read_text())
for file,expected in amendment['dependencies'].items():
    with open(file,'rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    assert actual==expected,file
original=ROOT/'refactor_v2/experiments/local_noaux_head_analysis_v1.py'
source=original.read_text()
changes={
    "with torch.no_grad():prob=train_head(x).sigmoid().flatten().tolist()":
    "with torch.no_grad():prob=torch.cat([train_head(x[i:i+4]).sigmoid().flatten() for i in range(0,len(x),4)]).tolist()",
    "dest=O/'postprocess';assert not dest.exists();dest.mkdir();dump(dest/'started.json'":
    "dest=O/'postprocess_v2';assert not dest.exists();dest.mkdir();dump(dest/'started.json'",
    "analyze();dump(dest/'completed.json',{'analysis_generated':True,'main_visual_review_final_closure_pending':True},True)":
    "analyze();dump(dest/'completed.json',{'analysis_generated':True,'main_visual_review_final_closure_pending':True,'batch_consistency_version':2},True);dump(O/'postprocess/completed.json',{'analysis_generated':True,'main_visual_review_final_closure_pending':True,'batch_consistency_version':2,'supervisor_receipt':str(dest/'completed.json')},True)",
}
for old,new in changes.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(original),'exec'),globals())
