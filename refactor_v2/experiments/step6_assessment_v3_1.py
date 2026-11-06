"""Frozen calibration/assessment and exact fresh-process replay for fixed seeds."""
import sys,time
import torch
from PIL import Image
from src.models.bonevqa import BoneVQAModel
from src.preprocessing import canonicalize_raster
from src.utils import compute_metrics
from experiments.grouped_public_evaluation_core_v3_1 import fit_from_canonical,calibrated_records,CANONICAL
from experiments.prepare_step4_localization_v3_1 import ROOT,R,read,dump
from experiments.vl_bounded_proposal_contract_v3_1 import sha
OUT=R/'step6_confirmation_v3_1'

def predict(model,rows):
    output=[]
    with torch.no_grad():
        for i in range(0,len(rows),4):
            selected=rows[i:i+4];images=[]
            for r in selected:
                with Image.open(ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'/r['image']) as im:images.append(canonicalize_raster(im).copy())
            predictions=model.predict_batch({'images':images,'questions':[r['question'] for r in selected]})
            assert len(predictions)==len(selected)
            for e,p in zip(selected,predictions):
                r={'image_id':e['image_id'],'question':e['question'],'gt':e['answer'],'answer_type':e['answer_type'],'pred':p['answer']}
                if 'p_yes' in p:r['p_yes']=float(p['p_yes'])
                output.append(r)
    return output

def measure(seed,arm):
    torch.set_num_threads(8);release=read(OUT/'assessment_release.json')
    for p,h in release['dependencies'].items():assert sha(p)==h,p
    checkpoint=release['checkpoints'][str(seed)][arm];assert sha(checkpoint['path'])==checkpoint['sha256']
    assert not arm.startswith('V'),'V assessment requires separately bound complete proposals'
    model=BoneVQAModel.load(checkpoint['path'],device='cpu').eval()
    binding=read(OUT/'role_binding.json')
    for p,h in binding['dependencies'].items():assert sha(p)==h,p
    calibration=[r for r in read(binding['reserved_paths']['calibration']) if r['question']==CANONICAL]
    assessment=read(binding['reserved_paths']['test'])
    cal=predict(model,calibration);fit=fit_from_canonical(cal)
    if fit['passed'] and fit['slope']<=0:fit={**fit,'passed':False,'failure':'nonpositive_slope'}
    raw=predict(model,assessment);adjusted=calibrated_records(raw,fit)
    assert sha(checkpoint['path'])==checkpoint['sha256']
    return {'seed':seed,'arm':arm,'checkpoint_sha256':checkpoint['sha256'],'calibration_records':cal,'calibration':fit,'raw_records':raw,'calibrated_records':adjusted,'raw_metrics':compute_metrics(raw,fracture_threshold=.5),'calibrated_metrics':compute_metrics(adjusted,fracture_threshold=.5) if adjusted is not None else None,'scope':'Development-informed assessment, not independent clinical confirmation','optimizer_updates':0}

if __name__=='__main__':
    mode,seed,arm=sys.argv[1],int(sys.argv[2]),sys.argv[3];path=OUT/f'assessment_seed{seed}_{arm}.json';started=time.perf_counter();actual=measure(seed,arm)
    if mode=='measure':dump(path,actual)
    else:
        assert actual==read(path)
        dump(OUT/f'assessment_seed{seed}_{arm}_replay.json',{'passed':True,'all_records_calibration_metrics_exact':True,'source_sha256':sha(path),'elapsed_seconds':time.perf_counter()-started})
