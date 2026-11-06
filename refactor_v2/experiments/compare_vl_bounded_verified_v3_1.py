"""Predeclared semantic contrasts on verified LAST checkpoints only.

Same inspected development data; component bootstrap is descriptive, not
independent confirmation or clinical efficacy evidence.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.context_controls import load_component_lookup
from src.utils import compute_metrics,normalize_answer,normalize_yes_no
from experiments.vl_bounded_proposal_contract_v3_1 import sha


def correct(row):
    norm=normalize_yes_no if row['answer_type']=='CLOSED' else normalize_answer
    return norm(row['pred'])==norm(row['gt'])


def main(group):
    root=Path.cwd();folder=root/'refactor_v2/reports/vl_bounded_quality_v3_1'
    arms=['V_real','V_location_placebo'] if group=='V' else ['L_real','L_original_teacher','L_location_placebo']
    destination=folder/(group+'_LAST296_matched_comparison.json')
    assert not destination.exists(), 'Never overwrite a completed comparison'
    base=json.loads((root/'refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json').read_text())
    binding=base['train']['component_batching']
    lookup,_=load_component_lookup(binding['manifest'],binding['seal'],root)
    records={};reports={}
    for arm in arms:
        report=json.loads((folder/arm/'independent_LAST296_verification.json').read_text())
        assert report['passed'] and report['global_step']==296 and not report['calibration_or_test_used']
        assert sha(folder/arm/'last.pt')==report['checkpoint_sha256']
        reports[arm]=report
        records[arm]=json.loads((folder/arm/'independent_LAST296_predictions.json').read_text())
        assert len(records[arm])==142 and compute_metrics(records[arm])==report['metrics']
    reference=records[arms[0]]
    identities=[{k:r[k] for k in ('image_id','question','gt','answer_type')} for r in reference]
    for arm in arms:
        assert [{k:r[k] for k in ('image_id','question','gt','answer_type')} for r in records[arm]]==identities
    components=sorted({lookup[r['image_id']] for r in reference})
    assert len(components)==32
    indices={c:[i for i,r in enumerate(reference) if lookup[r['image_id']]==c] for c in components}
    keys=['overall_acc','open_acc','fracture_auroc','fracture_auprc','fracture_sensitivity','fracture_specificity']
    contrasts={}
    for control in arms[1:]:
        draws={k:[] for k in keys};rng=np.random.default_rng(20261020)
        for _ in range(500):
            sample=[i for c in rng.choice(components,size=len(components),replace=True) for i in indices[c]]
            left=compute_metrics([reference[i] for i in sample])
            right=compute_metrics([records[control][i] for i in sample])
            for k in keys:
                # Undefined one-class clinical resamples are omitted, not
                # assigned a fabricated zero sensitivity/AUROC/AP.
                if k.startswith('fracture_') and ('fracture_auroc' not in left or 'fracture_auroc' not in right):
                    continue
                if k in left and k in right:draws[k].append(left[k]-right[k])
        contrasts[control]={
            'point_difference_real_minus_control':{k:reports[arms[0]]['metrics'][k]-reports[control]['metrics'][k] for k in keys},
            'paired_component_bootstrap_95_descriptive':{k:{'interval':np.quantile(v,[.025,.975]).tolist(),'valid_resamples':len(v)} for k,v in draws.items()},
            'real_correct_control_wrong':sum(correct(a) and not correct(b) for a,b in zip(reference,records[control])),
            'real_wrong_control_correct':sum(not correct(a) and correct(b) for a,b in zip(reference,records[control])),
            'verbatim_answer_differences':sum(a['pred']!=b['pred'] for a,b in zip(reference,records[control]))}
    result={'scope':'same142QA32component inspected development LAST296 matched semantic controls ONLY',
            'group':group,'seed':42,'arms':arms,'components':32,'QA_rows':142,'fracture_positive_images':5,
            'reports_sha256':{a:sha(folder/a/'independent_LAST296_verification.json') for a in arms},
            'metrics':{a:reports[a]['metrics'] for a in arms},'contrasts':contrasts,
            'bootstrap_resamples':500,'bootstrap_seed':20261020,'clinical_threshold':.5,
            'actual_train_wall_seconds':{a:reports[a]['train_wall_seconds'] for a in arms},
            'matched':'within V or L controls share full initial tensor values, QA schedule, phase/optimizer budgets, parameter and encoded view counts',
            'limitations':['single seed/reused development; no independent confirmation',
                           'Only32images5positive, not full321clinical cohort',
                           'Location-placebo preserves group geometry/shape priors; not a perfect semantic null',
                           'B comparison not capacity/phase/detector-GT-exposure matched',
                           'No automatic module contribution, production release, or Final Solution selection'],
            'calibration_or_test_used':False,'long_training_released':False}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['V','L'],required=True)
    main(parser.parse_args().group)
