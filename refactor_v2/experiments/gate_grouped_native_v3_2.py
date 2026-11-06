"""Preserve v3_1; normalize diagnostic NumPy arrays before JSON/replay comparison."""
from pathlib import Path
p=Path(__file__).with_name('gate_grouped_native_v3_1.py')
s=p.read_text().replace("grouped_native_model_gate_v3_1'","grouped_native_model_gate_v3_2'")
assert s.count('return results')==1
s=s.replace('return results',"return json.loads(json.dumps(results,allow_nan=False,default=lambda value:value.tolist()))")
exec(compile(s,str(p),'exec'))
