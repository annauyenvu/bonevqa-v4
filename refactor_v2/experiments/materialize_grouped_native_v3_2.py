"""Correct preparation assertion to existing donor-disjoint sampler contract.
v3_1 incorrectly demanded every batch component be unique; preserved unchanged.
"""
from pathlib import Path
from src.context_controls import component_disjoint_donors
p=Path(__file__).with_name('materialize_grouped_native_v3_1.py')
source=p.read_text().replace("grouped_native_preparation_v3_1'","grouped_native_preparation_v3_2'")
old="assert all(len({lookup[records[i]['image_id']] for i in batch})==len(batch) for batch in batches)"
new="assert all(len(component_disjoint_donors([lookup[records[i]['image_id']] for i in batch]))==len(batch) for batch in batches)"
assert source.count(old)==1
exec(compile(source.replace(old,new),str(p),'exec'))
