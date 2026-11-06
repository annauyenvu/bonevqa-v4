"""Two-epoch detector development pilot; NEVER run automatic test evaluation."""
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from scripts.train_detector_v2 import BoPhatHienDataset, PositiveAnchoredBatchSampler, collate, tao_mo_hinh, danh_gia
from src.detector_protocol import load_detector_manifest, verify_detector_annotation_parity
from src.preprocessing import RASTER_PREPROCESSING_VERSION
from src.utils import set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector_fold0_bounded_quality_v3"
    checkpoints=root/"refactor_v2/detector_fold0_bounded_quality_v3"
    if folder.exists() or checkpoints.exists():
        raise RuntimeError("Existing pilot;preserve and inspect,never overwrite")
    commands=subprocess.check_output(["ps","-axo","command"],text=True)
    if any(" -m src.train " in row or "scripts/train_detector_v2.py" in row for row in commands.splitlines()):
        raise RuntimeError("Active training;do not duplicate")
    assert os.environ["HF_HUB_OFFLINE"]==os.environ["TRANSFORMERS_OFFLINE"]=="1" and os.environ["OMP_NUM_THREADS"]=="8"
    subprocess.run([sys.executable,"refactor_v2/scripts/seal_technical_reference_v3.py","--verify"],check=True)
    manifest=root/".evaluation_runtime/data/fracatlas_component_cv_v3/fold_0/detector_split_v3.json"
    payload,ids,manifest_hash=load_detector_manifest(manifest)
    assert payload["fold"]==0 and payload["scope"]=="development_informed_component_fold"
    # Full manifest hash/partition metadata are audited; only these two datasets
    # are instantiated/evaluated. No test/calibration image loop is possible.
    train=BoPhatHienDataset("train",canh_dai=320,tang_cuong=True,allowed_ids=set(ids["train"]))
    val=BoPhatHienDataset("val",canh_dai=320,allowed_ids=set(ids["val"]))
    assert len(train)==2568 and len(val)==321
    for role,ds in (("train",train),("val",val)):
        verify_detector_annotation_parity(ds,payload["source_vqa"][role]["path"])
    set_seed(42)
    torch.set_num_threads(8)
    sampler=PositiveAnchoredBatchSampler(train,4,seed=42)
    loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate,num_workers=0)
    validation=DataLoader(val,batch_size=4,shuffle=False,collate_fn=collate,num_workers=0)
    sampler_audit=[]
    for epoch in range(2):
        sampler.epoch=epoch
        batches=list(iter(sampler))
        covered=set(i for batch in batches for i in batch)
        assert covered==set(range(len(train))) and all(any(train.muc[i][1][1] for i in batch) for batch in batches)
        sampler_audit.append({"epoch":epoch+1,"updates":len(batches),"unique_images":len(covered),"image_exposures":sum(map(len,batches)),"positive_exposures":sum(bool(train.muc[i][1][1]) for batch in batches for i in batch),"batch_order_sha256":hashlib.sha256(json.dumps(batches).encode()).hexdigest()})
    sampler.epoch=0
    folder.mkdir()
    checkpoints.mkdir()
    state={"status":"preparing","pid":os.getpid(),"started_utc":now(),"scope":"two-epoch fold0 TRAIN/modelvalidation detector quality pilot only;no VLP or final proposal release"}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        model=tao_mo_hinh("mobilenet_320").to("cpu")
        provenance=model.initial_weight_provenance
        assert sha(provenance["path"])==provenance["sha256"]
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.SGD(parameters,lr=0.005,momentum=0.9,weight_decay=0.0005)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
        release={"passed":True,"scope":state["scope"],"fold":0,"manifest_path":str(manifest),"manifest_sha256":manifest_hash,"TRAIN_images":len(train),"modelvalidation_images":len(val),"TRAIN_positive":len(sampler.positive),"TRAIN_negative":len(sampler.negative),"sampler":sampler_audit,"architecture":"FasterRCNN_MobileNetV3_320_FPN","initial_weight_provenance":provenance,"optimizer":"SGD lr0.005 momentum0.9 wd0.0005 clip5 cosine2epochs","seed":42,"epochs_cap":2,"batch_size":4,"device":"cpu","torch_threads":8,"workers":0,"evaluation_threshold":0.05,"positive_IoU_metric":"bestIoU across first5predictions vs any existingGTbox;not mAP or clinical diagnosis","checkpoint_primary":"LAST second epoch;best by modelvalidation mean bestIoU secondary","test_or_calibration_evaluated":False,"source_script_sha256":sha(__file__),"annotation_parity_verified":True,"runtime_offline":True,"long_training_or_VLP_released":False,"additional_exposure_disclosure":"Detector uses all2568foldTRAINimages,not only261images in B QA pilot;future V/L comparisons must disclose extra localization supervision/compute and use matched proposal controls"}
        (folder/"execution_release.json").write_text(json.dumps(release,indent=2)+"\n")
        history=[]
        best=-1.0
        step=0
        started=time.perf_counter()
        selected="roi_heads.box_predictor.cls_score.weight"
        named=dict(model.named_parameters())
        initial_head=named[selected].detach().clone()
        for epoch in range(2):
            epoch_start=time.perf_counter()
            model.train()
            losses=[]
            for images,targets,_ in loader:
                terms=model(list(images),list(targets))
                loss=sum(terms.values())
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if step==0:
                    gradient=float(named[selected].grad.abs().max())
                    assert gradient>0
                    state["first_step_head_gradient_max"]=gradient
                torch.nn.utils.clip_grad_norm_(parameters,5.0)
                optimizer.step()
                step+=1
                losses.append(float(loss.detach()))
                if step==1:
                    update=float((named[selected].detach()-initial_head).abs().max())
                    assert update>0
                    state["first_step_head_update_max"]=update
                if step==1 or step%25==0:
                    state.update(status="training",epoch=epoch+1,optimizer_step=step,maximum_optimizer_steps=sum(r["updates"] for r in sampler_audit),latest_total_loss=losses[-1],elapsed_seconds=time.perf_counter()-started)
                    save()
            scheduler.step()
            scores=danh_gia(model,validation,torch.device("cpu"),nguong=0.05)
            scores.update(epoch=epoch+1,mean_training_loss=sum(losses)/len(losses),epoch_seconds=time.perf_counter()-epoch_start,optimizer_steps=step)
            history.append(scores)
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            checkpoint={"state":model.state_dict(),"canh_dai":320,"val":scores,"architecture":"mobilenet_320","includes_negative_images":True,"training_split_sha256":manifest_hash,"training_scope":payload["scope"],"training_image_ids":sorted(ids["train"]),"initial_weight_provenance":provenance,"preprocessing_version":RASTER_PREPROCESSING_VERSION,"pilot_scope":state["scope"],"epoch":epoch+1,"optimizer_steps":step}
            torch.save(checkpoint,checkpoints/"last.pt")
            torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"optimizer_steps":step},checkpoints/"optimizer.pt")
            if scores["iou_tot_nhat_tb"]>best:
                best=scores["iou_tot_nhat_tb"]
                torch.save(checkpoint,checkpoints/"best.pt")
            state.update(status="epoch_complete",completed_epochs=epoch+1,history=history)
            save()
        report={"complete":True,"scope":state["scope"],"history":history,"wall_seconds":time.perf_counter()-started,"optimizer_steps":step,"last_sha256":sha(checkpoints/"last.pt"),"best_sha256":sha(checkpoints/"best.pt"),"optimizer_sha256":sha(checkpoints/"optimizer.pt"),"manifest_sha256":manifest_hash,"test_or_calibration_evaluated":False,"quality_gate_passed":False,"pending":"Independent provenance/reload/held-development-box quality audit before any V/L release"}
        (folder/"completion.json").write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="two_epochs_complete_pending_independent_verification",finished_utc=now(),completion_sha256=sha(folder/"completion.json"))
        save()
        print(json.dumps(report,indent=2))
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error),finished_utc=now())
        save()
        raise


if __name__=="__main__":
    main()
