"""Location-placebo geometry and forward-view/initial-state parity;no updates."""
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from src.checkpointing import compatible_state
from src.data import build_dataset,collate_fn
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.endpoints import is_canonical_fracture_presence_question
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.train import build_split
from src.utils import set_seed
from experiments.vl_bounded_proposal_contract_v3_1 import sha,validate_prepared_proposals


def union_area(boxes):
    xs=sorted({b[i] for b in boxes for i in (0,2)})
    area=0
    for x1,x2 in zip(xs,xs[1:]):
        spans=sorted((b[1],b[3]) for b in boxes if b[0]<x2 and b[2]>x1)
        length=0;end=-1
        for y1,y2 in spans:
            length+=max(0,y2-max(y1,end));end=max(end,y2)
        area+=(x2-x1)*length
    return area


def rounded(boxes,width,height):
    return [[max(0,min(width,int(round(b[0])))),max(0,min(height,int(round(b[1])))),max(0,min(width,int(round(b[2])))),max(0,min(height,int(round(b[3]))))] for b in boxes]


def group_translate(boxes,width,height):
    if not boxes:
        return []
    xmin,ymin=min(b[0] for b in boxes),min(b[1] for b in boxes)
    xmax,ymax=max(b[2] for b in boxes),max(b[3] for b in boxes)
    dxs={-xmin,width-xmax,0};dys={-ymin,height-ymax,0}
    candidates=[]
    for dx in dxs:
        for dy in dys:
            shifted=[[b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy] for b in boxes]
            overlap=sum(max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])) for a in boxes for b in shifted)
            candidates.append((overlap,-dx*dx-dy*dy,dx,dy,shifted))
    return min(candidates,key=lambda c:c[:4])[-1]


@torch.no_grad()
def main():
    started=time.perf_counter()
    root=Path.cwd()
    folder=root/"refactor_v2/reports/vl_matched_control_audit_v3_1"
    assert not folder.exists()
    source=root/"refactor_v2/reports/vl_pilot_proposals640_v3_1/boxes.json"
    validate_prepared_proposals(source)
    verified=json.loads((source.parent/"independent_contract_verification.json").read_text())
    assert verified["passed"] and sha(source)==verified["proposal_sha256"]
    boxes=json.loads(source.read_text())
    metadata=json.loads(source.with_suffix(".provenance.json").read_text())["images"]
    translated={};geometry=[]
    for iid,items in boxes.items():
        w,h=metadata[iid]["width"],metadata[iid]["height"]
        real=rounded(items,w,h)
        assert all(b[2]>b[0] and b[3]>b[1] for b in real)
        placebo=group_translate(real,w,h)
        assert len(real)==len(placebo) and union_area(real)==union_area(placebo)
        assert [(b[2]-b[0],b[3]-b[1]) for b in real]==[(b[2]-b[0],b[3]-b[1]) for b in placebo]
        independent=FracAtlasVQADataset._matched_placebo_boxes(items,(w,h)) or []
        translated[iid]=placebo
        geometry.append({"image_id":iid,"role":metadata[iid]["role"],"boxes":len(real),"translated":real!=placebo,"real_union_pixels":union_area(real),"group_placebo_union_pixels":union_area(placebo),"existing_independent_matched_union_difference":union_area(independent)-union_area(real)})
    folder.mkdir()
    (folder/"group_translation_location_placebo.json").write_text(json.dumps(translated,indent=2)+"\n")
    state={"status":"checking_forward_views","scope":"label-independent detector-derivedcontrols,no optimizer updates"}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        base=json.loads((root/"refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json").read_text())
        ds=build_split(build_dataset,base["data"]["dataset"],"train",None,True,{**base["data"]["loader_kwargs"],"with_masks":False,"with_boxes":False,"boxes_from":str(source),"boxes_as_masks":True,"roi_control":"real"})
        rows=records_of(ds)
        selected=[]
        for nonempty in (True,False):
            iid=next(i for i in sorted({r["image_id"] for r in rows}) if bool(boxes[i])==nonempty)
            selected.extend((next(i for i,r in enumerate(rows) if r["image_id"]==iid and is_canonical_fracture_presence_question(r["question"])),next(i for i,r in enumerate(rows) if r["image_id"]==iid and r.get("answer_type","").upper()=="OPEN")))
        real_batch=collate_fn([ds[i] for i in selected])
        placebo_batch=dict(real_batch)
        placebo_batch["boxes"]=[translated[i] or None for i in real_batch["image_ids"]]
        placebo_batch["masks"]=[FracAtlasVQADataset._boxes_to_masks(translated[i],im.size) for i,im in zip(real_batch["image_ids"],real_batch["images"])]
        init_path=root/"refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3_initialization.pt"
        assert sha(init_path)=="bfc799a5fb6db140ab833db460308c98cba76d871b9d415856175ed769abc09b"
        initial=torch.load(init_path,map_location="cpu",weights_only=False)
        variants=(("V_real","V",False,real_batch),("V_location_placebo","V",False,placebo_batch),("L_real","L",False,real_batch),("L_original_teacher","L",True,real_batch),("L_location_placebo","L",False,placebo_batch))
        states={};checks={}
        for name,kind,zero,batch in variants:
            cfg=copy.deepcopy(initial["config"])
            cfg.update(prompt_variant=kind,use_visual_prompt=kind=="V",use_lens=kind=="L",use_latent_prompt=False,use_region_in_prompt=False,use_textual_prompt=False,lens_force_zero=zero)
            set_seed(42)
            model=BoneVQAModel(cfg,initial["answer_vocab"],device="cpu")
            model.load_state_dict(compatible_state(model.trainable_state_dict(),initial["state"],base["checkpoint_allow_mismatch_prefixes"]),strict=False)
            current=model.trainable_state_dict()
            assert all(torch.equal(current[k],v) for k,v in initial["state"].items())
            hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in current.items()}
            if kind in states:
                assert hashes==states[kind]
            else:
                states[kind]=hashes
            counts=[]
            preprocess=model.visual.preprocess
            def record_views(images,device):
                counts.append(len(images))
                return preprocess(images,device)
            model.visual.preprocess=record_views
            phases=(3,) if kind=="V" else (1,3)
            observed={}
            for phase in phases:
                set_seed(42)
                model.train()
                out=model(batch,stage=phase)
                assert torch.isfinite(out["loss"])
                observed[str(phase)]={"visual_encoder_input_views":counts[-1],"loss_finite":True,"loss_is_training_metric_not_efficacy":True}
                del out
            assert {k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in model.trainable_state_dict().items()}==hashes
            checks[name]={"shared_B_tensors_exact":263,"all_initial_tensors":len(hashes),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"phases":observed,"lens_original_image_only_inference":cfg["lens_training_only"] if kind=="L" else None,"optimizer_updates":0}
            del model,current,preprocess
            gc.collect()
            state.update(last_variant=name);save()
        assert checks["V_real"]["phases"]==checks["V_location_placebo"]["phases"]
        assert checks["L_real"]["phases"]==checks["L_original_teacher"]["phases"]==checks["L_location_placebo"]["phases"]
        result={"passed":True,"scope":state["scope"],"proposal_source_sha256":sha(source),"location_placebo_sha256":sha(folder/"group_translation_location_placebo.json"),"geometry":geometry,"all293_perbox_geometry_and_union_pixels_exact":True,"nonempty_images":sum(bool(v) for v in boxes.values()),"nonempty_images_relocated":sum(r["translated"] for r in geometry),"existing_independent_matched_union_changes":sum(r["existing_independent_matched_union_difference"]!=0 for r in geometry),"selected_TRAIN_QA_indices":selected,"checks":checks,"initial_tensor_hashes":states,"elapsed_seconds":time.perf_counter()-started,"calibration_or_test_used":False,"quality_training_released":False,"limitations":["grouptranslationdestroyslocationnotallshape/lesionpriors;notperfectsemanticnull","actualforwardviews/parameterparitychecked;notproofequalwalltimeorallFLOPs","fourTRAINQAforwardsonly,nooptimizationorefficacy","BvsV/Ldifferentcapacity,detectorlocalizationGTandcompute;discloseextrasupervision"]}
        (folder/"report.json").write_text(json.dumps(result,indent=2)+"\n")
        state.update(status="passed_controls_preparation_only",report_sha256=sha(folder/"report.json"));save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error));save();raise


if __name__=="__main__":
    main()
