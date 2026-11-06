"""Supplementary TRAIN-majority constant baseline, no model execution/tuning."""
from collections import Counter
import json
from pathlib import Path
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.prepare_lapa_pilot_v1 import sha
from src.utils import normalize_answer

OUT=ROOT/'reports/favp_constant_baseline_v1'


def main():
    OUT.mkdir(exist_ok=False)
    selection_path=ROOT/'reports/favp_pilot_inputs_v1/selection_and_schedules.json'
    selection=json.loads(selection_path.read_text())
    train_path=ROOT/'reports/step4_localization_v3_1/train.json'
    records=json.loads(train_path.read_text());counts=Counter()
    for chosen in selection['subset']['train']:
        r=records[chosen['record_index']]
        assert r['image_id']==chosen['image_id'] and r['question']==chosen['question']
        answer=normalize_answer(r['answer']);assert answer in ('yes','no');counts[answer]+=1
    # Deterministic lexical tie-break; no model-validation labels inspected.
    prediction=sorted(counts,key=lambda a:(-counts[a],a))[0]
    quality=ROOT/'reports/favp_bounded_quality_v1'
    assert not (quality/'epoch1.json').exists() and not (quality/'epoch2.json').exists(), 'Define before post-training aggregate results'
    definition={'prediction':prediction,'training_answer_counts':dict(counts),'training_examples':32,
        'inputs_ignored':['image','question'],'selection_sha256':sha(selection_path),'train_manifest_sha256':sha(train_path),
        'timing_disclosure':'supplementary exploratory baseline added during running pilot, after PRE but before completed epoch1/epoch2 aggregates; not original primary protocol',
        'no_validation_labels_used_to_choose_prediction':True,'no_training_or_generation_run':True,
        'purpose':'interpret strict QA accuracy against a constant TRAIN-majority answer; not a new tuned model or proof of image conditioning',
        'script_sha256':sha(Path(__file__))}
    (OUT/'definition.json').write_text(json.dumps(definition,indent=2))


if __name__=='__main__':main()
