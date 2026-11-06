"""Resolve frozen trainer's TRAIN/seed-derived model fields before sealing.

Preserves v3_1 rejected plan/source. No scientific recipe change: exact same
dynamic resolution already performed by unchanged src.train.
"""
import argparse
import json
from pathlib import Path
from src.endpoints import is_canonical_fracture_presence_question
from src.protocol import records_of
from src.utils import normalize_answer
from experiments import run_B_sequence_loss_pilots_v3_1 as core
from experiments.run_T_bounded_current_v3_1 import data_and_schedule

IMPLEMENTATION=Path(core.__file__).resolve()
ORIGINAL_RESOLVE=core.resolved_base_model
core.__file__=str(Path(__file__).resolve())
core.OUT=core.ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_2'


def resolved_base_model(base):
    result=ORIGINAL_RESOLVE(base)
    result['module_init_seed']=int(base['seed'])+int(result.get('module_init_seed_offset',1729))
    _,datasets,_=data_and_schedule();counts={'yes':0,'no':0}
    for row in records_of(datasets['train']):
        answer=normalize_answer(row['answer'])
        if row['answer_type']=='CLOSED' and is_canonical_fracture_presence_question(row['question']) and answer in counts:
            counts[answer]+=1
    assert counts=={'yes':133,'no':128}
    result['fracture_positive_weight']=counts['no']/counts['yes']
    return result


core.resolved_base_model=resolved_base_model


def prepare():
    core.prepare()
    path=core.OUT/'plan.json';plan=json.loads(path.read_text())
    failure=core.ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_1/integration_precheck_failure.json'
    for dependency in [IMPLEMENTATION,failure]:
        plan['dependencies_sha256'][str(dependency)]=core.sha(dependency)
    plan['runtime_config_resolution']={'module_init_seed':1771,'fracture_positive_weight':128/133,
        'source':'Unchanged src.train seed+offset and canonical TRAIN yes/no counts, not held-out labels',
        'previous_unexecuted_plan_preserved':str(failure.parent),'scientific_recipe_changed':False}
    core.dump(path,plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','verify','run','worker']);parser.add_argument('mode',nargs='?',choices=core.MODES)
    args=parser.parse_args()
    if args.action=='worker':
        assert args.mode;core.worker(args.mode)
    else:{'prepare':prepare,'verify':core.verify,'run':core.run}[args.action]()
