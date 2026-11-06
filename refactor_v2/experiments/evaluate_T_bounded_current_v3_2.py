"""Scoped reporting-contract repair; preserve failed v3_1 unchanged.

Same predeclared checkpoints/data/metrics/inference and replay. Optional p_yes
is recorded only if actually returned; canonical fracture requires the score.
"""
import argparse
import json
from pathlib import Path

import torch
from experiments import evaluate_T_bounded_current_v3_1 as core

IMPLEMENTATION = Path(core.__file__).resolve()
WRAPPER = Path(__file__).resolve()
core.__file__ = str(WRAPPER)  # Fresh subprocess replay must use this repair too.
core.OUT = core.ROOT/'refactor_v2/reports/T_independent_quality_current_v3_2'


def prediction_record(batch, j, prediction):
    row = {'image_id':batch['image_ids'][j], 'question':batch['questions'][j],
           'gt':batch['answers'][j], 'answer_type':batch['answer_types'][j],
           'pred':prediction['answer']}
    if 'p_yes' in prediction:
        row['p_yes'] = prediction['p_yes']
    if row['question']=='Is there a fracture in this X-ray?':
        assert 'p_yes' in row, 'Canonical fracture score must exist, never fabricate it'
    return row


@torch.no_grad()
def predict(model, ds, indices):
    before = core.hashes(model.trainable_state_dict())
    rows = []
    for offset in range(0,len(indices),4):
        batch = core.collate_fn([ds[i] for i in indices[offset:offset+4]])
        predictions = model.predict_batch({'images':batch['images'],'questions':batch['questions']})
        assert len(predictions)==len(batch['questions'])
        rows.extend(prediction_record(batch,j,p) for j,p in enumerate(predictions))
    assert core.hashes(model.trainable_state_dict())==before
    return {'records':rows,'metrics':core.compute_metrics(rows,fracture_threshold=.5),'weights_unchanged':True}


core.predict = predict


def prepare():
    # Confirm the original failed run remains preserved before creating a new
    # plan. Neither the original script nor any scientific criterion is changed.
    failed = core.ROOT/'refactor_v2/reports/T_independent_quality_current_v3_1/progress.json'
    prior = json.loads(failed.read_text())
    assert prior['status']=='failed_preserved_no_automatic_retry'
    assert prior['failure']=="KeyError('p_yes')"
    core.prepare()
    plan_path = core.OUT/'evaluation_plan.json'
    plan = json.loads(plan_path.read_text())
    plan['dependencies_sha256'][str(IMPLEMENTATION)] = core.sha(IMPLEMENTATION)
    plan['dependencies_sha256'][str(failed)] = core.sha(failed)
    plan['repair'] = {'previous_failed_folder':str(failed.parent),
        'reason':'Non-fracture public predictions do not necessarily contain p_yes',
        'change':'Record p_yes only when returned, exactly as frozen src.train evaluate_split; require it for canonical fracture',
        'fabricated_probabilities':False,'scientific_protocol_changed':False,
        'original_failed_script_modified':False,'original_failed_artifacts_overwritten':False}
    core.dump(plan_path,plan)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','run','verify'])
    args = parser.parse_args()
    {'prepare':prepare,'run':core.run,'verify':core.replay}[args.mode]()
