"""New bounded pilot preparation; preserves all historical releases."""
import copy, json, math, os, random, time
from pathlib import Path
import torch
from PIL import Image
from src.utils import normalize_answer, set_seed
from src.context_controls import ComponentBatchSampler
from src.preprocessing import canonicalize_raster
from experiments.vl_bounded_proposal_contract_v3_1 import sha, PilotPreparationDetector640
from experiments.audit_vl_matched_controls_v3_1 import rounded, group_translate, union_area

ROOT=Path.cwd(); R=ROOT/'refactor_v2/reports'; OUT=R/'step4_localization_v3_1'
def read(p): return json.loads(Path(p).read_text())
def dump(p,v):
    with Path(p).open('x') as f: json.dump(v,f,indent=2); f.write('\n')

def main():
    assert not OUT.exists()
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    torch.set_num_threads(8); started=time.perf_counter()
    parent=ROOT/'.evaluation_runtime/data/fracatlas_component_cv_v3'
    manifest=read(parent/'manifest.json'); train=read(parent/'fold_0/vqa_train.json'); val=read(parent/'fold_0/vqa_model_val.json')
    owner={im:i for i,c in enumerate(manifest['components']) for im in c}
    canonical={r['image_id']:r for r in train if r['question']=='Is there a fracture in this X-ray?'}
    rng=random.Random(20261022); selected=set()
    # Whole components, stratified by presence of positive TRAIN images; no model scores.
    for positive,target in [(True,46),(False,210)]:
        groups=[c for c in manifest['components'] if set(c)<=set(canonical) and any(canonical[i]['fractured'] for i in c)==positive]
        rng.shuffle(groups); count=0
        for c in groups:
            if count>=target: break
            selected.update(c); count+=len(c)
    tr=[r for r in train if r['image_id'] in selected]
    assert not {owner[i] for i in selected}&{owner[r['image_id']] for r in val}
    OUT.mkdir(); dump(OUT/'train.json',tr); dump(OUT/'model_val.json',val)
    vocab=sorted({normalize_answer(r['answer']) for r in tr}); dump(OUT/'vocab.json',vocab)
    seal={'status':'sealed_development_data','artifacts':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'train.json',OUT/'model_val.json',OUT/'vocab.json',parent/'manifest.json']}}
    dump(OUT/'data_seal.json',seal)
    base=read(R/'fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json')
    detector=read(R/'vl_pilot_proposals640_v3_1/active_roles_manifest.json')
    detector['source_QA_seal']={'path':str(OUT/'data_seal.json'),'sha256':sha(OUT/'data_seal.json')}
    detector['roles']={role:{'path':str(OUT/(role+'.json')),'sha256':sha(OUT/(role+'.json')),'QA_rows':len(rows),'image_ids':sorted({r['image_id'] for r in rows})} for role,rows in [('train',tr),('model_val',val)]}
    dump(OUT/'active_roles_manifest.json',detector)
    sampler=ComponentBatchSampler([owner[r['image_id']] for r in tr],4,42); schedules=[]
    for epoch in range(4):
        sampler.set_epoch(epoch); batches=list(sampler)
        assert sorted(i for batch in batches for i in batch)==list(range(len(tr)))
        schedules.append(batches)
    dump(OUT/'schedules.json',schedules)
    for arm in ('B','V_real','V_placebo'):
        cfg=copy.deepcopy(base); cfg['run_name']='step4_'+arm+'_v3_1'
        cfg['data']['sealed_answer_vocab']=str(OUT/'vocab.json')
        kw=cfg['data']['loader_kwargs']; kw.update(seal_path=str(OUT/'data_seal.json'),boxes_from=None,boxes_as_masks=False)
        kw['sealed_records_by_split']={'train':str(OUT/'train.json'),'model_val':str(OUT/'model_val.json'),'calibration':None,'test':None}
        cfg['train']['component_batching']['seal']=str(OUT/'data_seal.json')
        cfg['train']['epochs']=4; cfg['model'].update(use_visual_prompt=arm!='B',prompt_variant='B' if arm=='B' else 'V')
        positives=sum(canonical[i]['fractured'] for i in selected)
        cfg['model']['fracture_positive_weight']=(len(selected)-positives)/positives
        cfg['init_from']=str(OUT/(arm+'_PREtask.pt'))
        cfg['pilot_contract']={'primary':'LAST4','proposal_injection':'explicit automatic proposal collate, not annotation loader','proposal_path':str(OUT/('placebo.json' if arm=='V_placebo' else 'boxes.json')),'updates':sum(math.ceil(len(b)/4) for b in schedules),'training_released':False}
        dump(OUT/(arm+'.json'),cfg)
    dump(OUT/'manifest.json',{'source_manifest_sha256':sha(parent/'manifest.json'),'selection_seed':20261022,'train_images':len(selected),'train_positive':positives,'train_QA':len(tr),'model_val_images':len({r['image_id'] for r in val}),'model_val_QA':len(val),'microbatches':[len(b) for b in schedules],'optimizer_updates':sum(math.ceil(len(b)/4) for b in schedules),'scope':'development-informed bounded pilot','training_released':False})
    runtime=PilotPreparationDetector640(OUT/'active_roles_manifest.json')
    paths={r['image_id']:Path(base['data']['loader_kwargs']['root'])/r['image'] for r in tr+val} if 'root' in base['data']['loader_kwargs'] else {r['image_id']:ROOT/'.evaluation_runtime/data/fracatlas/FracAtlas'/r['image'] for r in tr+val}
    boxes={}; placebo={}; images={}
    for i,(iid,path) in enumerate(sorted(paths.items())):
        with Image.open(path) as im:
            im.load(); w,h=canonicalize_raster(im).size
            with torch.no_grad(): items=runtime.predict(im)
        real=rounded(items,w,h); control=group_translate(real,w,h)
        assert len(real)==len(control) and union_area(real)==union_area(control)
        boxes[iid]=real; placebo[iid]=control
        images[iid]={'sha256':sha(path),'width':w,'height':h,'role':'train' if iid in selected else 'model_val'}
        if i%50==0: print(json.dumps({'exported':i+1,'total':len(paths),'elapsed':time.perf_counter()-started}),flush=True)
    dump(OUT/'boxes.json',boxes); dump(OUT/'placebo.json',placebo)
    dump(OUT/'proposal_provenance.json',{'detector':runtime.provenance,'images':images,'real_sha256':sha(OUT/'boxes.json'),'placebo_sha256':sha(OUT/'placebo.json'),'source_sha256':sha(__file__),'annotation_inputs':False})
    dump(OUT/'preparation.json',{'passed':True,'elapsed_seconds':time.perf_counter()-started,'coverage':len(boxes),'nonempty':sum(bool(v) for v in boxes.values()),'remaining':['initialization','gradient/update/routing/reload','timing'],'training_released':False})
if __name__=='__main__': main()
