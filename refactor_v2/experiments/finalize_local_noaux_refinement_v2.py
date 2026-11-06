"""Add amendment audit to the unchanged v1 finalization and closure contract."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path('/Users/uyen/Downloads/BONEVQA');O=ROOT/'refactor_v2/reports/local_noaux_refinement_v1'
if sys.argv[1]=='seal':
    amend=json.loads((O/'head_batch_consistency_v2/release.json').read_text())
    for file,expected in amend['dependencies'].items():
        with open(file,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected,file
    assert json.loads((O/'postprocess_v2/completed.json').read_text())['batch_consistency_version']==2
    assert json.loads((O/'postprocess/completed.json').read_text())['batch_consistency_version']==2
original=ROOT/'refactor_v2/experiments/finalize_local_noaux_refinement_v1.py'
exec(compile(original.read_text(),str(original),'exec'),globals())
