"""After final execution, compare frozen TRAIN-only constant without tuning."""
import json
from pathlib import Path
from experiments.favp_visual_bridge_v2 import ROOT
from experiments.prepare_lapa_pilot_v1 import sha
from src.utils import normalize_answer

OUT=ROOT/'reports/favp_constant_baseline_v1'


def main():
    assert not (OUT/'comparison.json').exists()
    quality=ROOT/'reports/favp_bounded_quality_v1'
    assert (quality/'execution_completed.json').exists()
    definition=json.loads((OUT/'definition.json').read_text())
    selection_path=ROOT/'reports/favp_pilot_inputs_v1/selection_and_schedules.json'
    train_path=ROOT/'reports/step4_localization_v3_1/train.json'
    assert sha(selection_path)==definition['selection_sha256'] and sha(train_path)==definition['train_manifest_sha256']
    selection=json.loads(selection_path.read_text());records=json.loads((ROOT/'reports/step4_localization_v3_1/model_val.json').read_text())
    targets=[]
    for chosen in selection['subset']['model_val']:
        r=records[chosen['record_index']];assert r['image_id']==chosen['image_id'];targets.append(normalize_answer(r['answer']))
    accuracy=sum(a==definition['prediction'] for a in targets)/len(targets)
    result={'constant_prediction':definition['prediction'],'constant_accuracy':accuracy,'records':len(targets),
        'timing_disclosure':definition['timing_disclosure'],'sources':{str(OUT/'definition.json'):sha(OUT/'definition.json')},'stages':{}}
    for stage in ['PRE','epoch1','epoch2']:
        p=quality/f'{stage}.json';data=json.loads(p.read_text())
        assert [r['target'] for r in data['rows']]==targets
        result['stages'][stage]={'model_accuracy':data['metrics']['accuracy'],
            'model_minus_constant':data['metrics']['accuracy']-accuracy,
            'same_answer_as_constant':sum(r['normalized_answer']==definition['prediction'] for r in data['rows'])}
        result['sources'][str(p)]=sha(p)
    result['interpretation_limit']='Matching/beating this exploratory constant is not proof of image use; no clinical efficacy or patient-independent inference from eight examples.'
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
