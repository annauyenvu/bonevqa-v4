"""Public-only inference and per-fold calibration/outer evaluation primitives.

No command-line execution or role authorization. A separately gated driver must
check the frozen plan before constructing datasets or calling these functions.
"""
import numpy as np
from experiments.grouped_calibration_v3_1 import fit,apply

CANONICAL='Is there a fracture in this X-ray?'

def predict_records(model,dataset,expected,canonical_only=False,progress=None):
    import torch
    from src.data import collate_fn
    if len(dataset)!=len(expected):raise ValueError('Dataset length differs from sealed records')
    indices=[i for i,r in enumerate(expected) if not canonical_only or r['question']==CANONICAL]
    rows=[]
    for offset in range(0,len(indices),4):
        selected=indices[offset:offset+4]
        batch=collate_fn([dataset[i] for i in selected])
        # Construct the public input anew: no answer/type/region/annotation key is forwarded.
        with torch.no_grad():predictions=model.predict_batch({'images':batch['images'],'questions':batch['questions']})
        if len(predictions)!=len(selected):raise ValueError('Prediction length mismatch')
        for j,(index,pred) in enumerate(zip(selected,predictions)):
            e=expected[index]
            identity=(batch['image_ids'][j],batch['questions'][j],batch['answers'][j],batch['answer_types'][j])
            if identity!=(e['image_id'],e['question'],e['answer'],e['answer_type']):raise ValueError('Sealed record/order mismatch')
            row={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':pred['answer']}
            if 'p_yes' in pred:
                p=float(pred['p_yes'])
                if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid probability')
                row['p_yes']=p
            if row['question']==CANONICAL:
                if 'p_yes' not in row:raise ValueError('Missing canonical fracture probability')
                if row['pred']!=('yes' if row['p_yes']>=.5 else 'no'):raise ValueError('Raw decision violates frozen threshold')
            rows.append(row)
        if progress:progress(len(rows),len(indices))
    return rows

def fit_from_canonical(rows):
    if not rows or any(r['question']!=CANONICAL for r in rows):raise ValueError('Canonical calibration rows required')
    if len(rows)!=len({r['image_id'] for r in rows}):raise ValueError('Duplicate calibration image')
    if any(r['gt'].strip().lower() not in ('yes','no') for r in rows):raise ValueError('Existing binary labels required')
    return fit([r['p_yes'] for r in rows],[int(r['gt'].strip().lower()=='yes') for r in rows])

def calibrated_records(raw,calibration):
    # Failure leaves the saved raw output usable; there is no fallback refit.
    if not calibration['passed']:return None
    result=[dict(r) for r in raw]
    selected=[i for i,r in enumerate(raw) if r['question']==CANONICAL]
    probabilities=apply([raw[i]['p_yes'] for i in selected],calibration)
    for index,p in zip(selected,probabilities):
        result[index]['p_yes']=float(p)
        result[index]['pred']='yes' if p>=.5 else 'no'
    return result
