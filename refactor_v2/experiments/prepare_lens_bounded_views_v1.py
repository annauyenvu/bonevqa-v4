"""Freeze reused development cohort then prepare native automatic TRAIN views."""
import csv
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from transformers import SamModel, SamImageProcessor, pipeline
from experiments.lens_native_common_v1 import ROOT,SOURCE,sha,verify_source,verify_package,fixed_train_image

OUT=ROOT/'reports/lens_bounded_inputs_v1'


def main():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(8)
    selection=ROOT/'reports/favp_pilot_inputs_v1/selection_and_schedules.json'
    prior=json.loads((selection.parent/'receipt.json').read_text())
    assert prior['complete'] and sha(selection)==prior['selection_sha256']
    original_selection=json.loads(selection.read_text())
    for p,h in original_selection['dependencies'].items():assert sha(Path(p))==h
    subset=original_selection['subset']
    seal=json.loads((ROOT/'reports/step4_localization_v3_1/data_seal.json').read_text())
    for role in ['train','model_val']:
        p=ROOT/f'reports/step4_localization_v3_1/{role}.json'
        assert sha(p)==seal['artifacts'][f'refactor_v2/reports/step4_localization_v3_1/{role}.json']
    rows=json.loads((ROOT/'reports/step4_localization_v3_1/train.json').read_text())
    csv_path=ROOT.parent/'.evaluation_runtime/data/fracatlas/FracAtlas/dataset.csv'
    with csv_path.open(encoding='utf-8-sig') as f: metadata={r['image_id']:r for r in csv.DictReader(f)}
    dependencies={str(p):sha(p) for p in [Path(__file__),ROOT/'reports/LENS_BOUNDED_NATIVE_ASSET_PILOT_PROTOCOL_V1.md',selection,
        ROOT/'reports/step4_localization_v3_1/train.json',ROOT/'reports/step4_localization_v3_1/model_val.json',csv_path,
        ROOT/'reports/lens_fixed_native_views_v1/receipt.json',ROOT/'reports/lens_sam_native_recovery_v2/receipt.json']}
    for rel in ['src/localization_lens/sam_augment.py','src/localization_lens/views.py','src/localization_lens/io.py']:
        p=verify_source(rel);dependencies[str(p)]=sha(p)
    (OUT/'frozen_selection.json').write_text(json.dumps({'candidate_subset':subset,'dependencies':dependencies,
        'seed':42,'quality_released':False,'source':'reuse immutable sorted canonical FAVP cohort; not fresh validation'},indent=2))
    sys.path.insert(0,str(SOURCE/'src'))
    from localization_lens.sam_augment import generate_masks
    from localization_lens.views import render_views
    accepted=json.loads((ROOT/'reports/lens_fixed_native_views_v1/receipt.json').read_text())['cases']
    reused={c['image']['image_id']:c for c in accepted if c['eligible']}
    args=SimpleNamespace(crops_n_layers=0,points_per_batch=8,points_per_crop=32,pred_iou_thresh=.8,stability_score_thresh=.95)
    generator=None;rng=np.random.default_rng(42);results=[]
    for candidate in subset['train']:
        assert time.perf_counter()-start<5400,'90-minute preparation cap'
        assert shutil.disk_usage(ROOT).free>=8*1024**3,'8-GiB disk reserve'
        index=candidate['record_index'];row=rows[index]
        assert row['image_id']==candidate['image_id'] and row['question']==candidate['question']
        tick=time.perf_counter();image,binding=fixed_train_image(index)
        meta=metadata[row['image_id']+'.jpg'];regions=[k for k in ['hand','leg','hip','shoulder'] if int(meta[k])==1]
        region=regions[0] if len(regions)==1 else (' and '.join(regions) if len(regions)==2 else ', '.join(regions[:-1])+' and '+regions[-1] if regions else None)
        result={'candidate':candidate,'image':binding,'region':region,'metadata_row':meta,'eligible':False}
        if region is None or region!=row['region']:
            result['reason']='No verified existing region label; no replacement'
        elif row['image_id'] in reused:
            existing=reused[row['image_id']]
            assert binding['file_sha256']==existing['image']['file_sha256']
            artifacts=existing.get('artifacts') or json.loads((ROOT/'reports/lens_sam_native_recovery_v2/receipt.json').read_text())['view_artifacts']
            for p,h in artifacts.items():assert sha(Path(p))==h
            result.update(eligible=True,reused=True,view_directory=existing['view_directory'],artifacts=artifacts,
                          alpha=existing['alpha'],retained_masks=existing['retained_masks'])
        else:
            if generator is None:
                folder=verify_package('sam-vit-huge')
                model,loading=SamModel.from_pretrained(folder,local_files_only=True,use_safetensors=True,dtype=torch.float32,output_loading_info=True)
                assert not loading['missing_keys'] and not loading['unexpected_keys'] and not loading.get('mismatched_keys',[])
                generator=pipeline('mask-generation',model=model.eval().requires_grad_(False),
                    image_processor=SamImageProcessor.from_pretrained(folder,local_files_only=True),device=-1)
            masks=generate_masks(generator,image,args);alpha=float(np.clip(rng.normal(.5,.1),0.,1.))
            directory=OUT/row['image_id']
            try: artifacts=render_views(image,masks,directory,'sam',alpha)
            except ValueError as exc:
                result.update(reason=str(exc),raw_selected_masks=len(masks),alpha=alpha)
            else:
                result.update(eligible=True,reused=False,view_directory=str(directory),raw_selected_masks=len(masks),alpha=alpha,
                    artifacts={str(p):sha(Path(p)) for p in artifacts.values()},
                    retained_masks=int(np.load(directory/'sam_masks.npz')['masks'].shape[0]))
        result['seconds']=time.perf_counter()-tick;results.append(result)
        (OUT/(row['image_id']+'.json')).write_text(json.dumps(result,indent=2))
        (OUT/'progress.json').write_text(json.dumps({'completed':len(results),'candidates':32,
            'eligible':sum(r['eligible'] for r in results),'seconds':time.perf_counter()-start,'last_image':row['image_id']}))
    eligible=[r for r in results if r['eligible']]
    count=len(eligible)//2*2
    assert count>=16,'Insufficient eligible candidates; no extra images automatically added'
    assert all(sha(Path(p))==h for p,h in dependencies.items())
    (OUT/'receipt.json').write_text(json.dumps({'complete':True,'cases':results,'selected_training_ids':[r['image']['image_id'] for r in eligible[:count]],
        'eligible_count':len(eligible),'training_count':count,'unused_final_unmatched_ids':[r['image']['image_id'] for r in eligible[count:]],
        'model_validation':subset['model_val'],'dependencies':dependencies,'dependencies_unchanged':True,
        'seconds':time.perf_counter()-start,'quality_released':False,'arguments':vars(args)},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            with (OUT/'failure.json').open('x') as f:json.dump({'error':repr(exc)},f)
        raise
