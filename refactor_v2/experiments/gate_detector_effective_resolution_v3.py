"""Eight TRAIN updates per fresh resolution arm;technical timing,not efficacy."""
import copy
import gc
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scripts.train_detector_v2 import BoPhatHienDataset,PositiveAnchoredBatchSampler,tao_mo_hinh
from src.detector_protocol import load_detector_manifest,verify_detector_annotation_parity
from src.utils import set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/detector_effective_resolution_gate_v3"
    if folder.exists():
        raise RuntimeError("Existing gate,preserve and inspect")
    commands=subprocess.check_output(["ps","-axo","command"],text=True)
    if any(" -m src.train " in row or "run_detector_fold0" in row for row in commands.splitlines()):
        raise RuntimeError("Existing training active")
    assert os.environ["OMP_NUM_THREADS"]=="8" and os.environ["HF_HUB_OFFLINE"]==os.environ["TRANSFORMERS_OFFLINE"]=="1"
    manifest=root/".evaluation_runtime/data/fracatlas_component_cv_v3/fold_0/detector_split_v3.json"
    payload,ids,digest=load_detector_manifest(manifest)
    previous=json.loads((root/"refactor_v2/reports/detector_fold0_bounded_quality_v3/execution_release.json").read_text())
    assert digest==previous["manifest_sha256"]
    folder.mkdir()
    state={"status":"running","pid":os.getpid(),"scope":"8TRAINupdates per fresh320/640arm;not detector quality or VLP release"}
    def save():
        (folder/"progress.json").write_text(json.dumps(state,indent=2)+"\n")
    save()
    started=time.perf_counter()
    results={}
    initial_hashes=None
    order=None
    try:
        torch.set_num_threads(8)
        for size in (320,640):
            ds=BoPhatHienDataset("train",canh_dai=size,tang_cuong=True,allowed_ids=set(ids["train"]))
            verify_detector_annotation_parity(ds,payload["source_vqa"]["train"]["path"])
            sampler=PositiveAnchoredBatchSampler(ds,4,seed=42)
            batches=list(iter(sampler))[:8]
            images_order=[[ds.muc[i][0] for i in batch] for batch in batches]
            if order is None:
                order=images_order
            else:
                assert order==images_order
            set_seed(42)
            model=tao_mo_hinh("mobilenet_320").to("cpu").train()
            provenance=model.initial_weight_provenance
            assert provenance["sha256"]==previous["initial_weight_provenance"]["sha256"]==sha(provenance["path"])
            hashes={k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
            if initial_hashes is None:
                initial_hashes=hashes
            else:
                assert initial_hashes==hashes
            original_transform={"min_size":list(model.transform.min_size),"max_size":model.transform.max_size}
            if size==640:
                model.transform.min_size=(640,)
                model.transform.max_size=640
            assert tuple(model.transform.min_size)==(size,)
            optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=0.005,momentum=0.9,weight_decay=0.0005)
            selected=model.roi_heads.box_predictor.cls_score.weight
            before=selected.detach().clone()
            timings,losses=[],[]
            geometry=None
            for step,indices in enumerate(batches):
                t0=time.perf_counter()
                # Paired independent augmentations;identical horizontal flips.
                np.random.seed(42+step)
                samples=[ds[i] for i in indices]
                images=[sample[0] for sample in samples]
                targets=[sample[1] for sample in samples]
                if step==0:
                    transformed,new_targets=model.transform(images,copy.deepcopy(targets))
                    assert all(tuple(s)==(size,size) for s in transformed.image_sizes)
                    error=max(float((a["boxes"]-b["boxes"]).abs().max()) if a["boxes"].numel() else 0 for a,b in zip(targets,new_targets))
                    assert error==0
                    geometry={"tensor_canvas_shape":list(transformed.tensors.shape),"actual_image_sizes":[list(s) for s in transformed.image_sizes],"targetbox_transform_max_error_at_intended_resolution":error,"positive_box_widths_pixels":[float(b[2]-b[0]) for target in new_targets for b in target["boxes"]],"positive_box_heights_pixels":[float(b[3]-b[1]) for target in new_targets for b in target["boxes"]]}
                    del transformed,new_targets
                terms=model(images,targets)
                loss=sum(terms.values())
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad=float(selected.grad.abs().max())
                assert np.isfinite(grad) and grad>0
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()
                timings.append(time.perf_counter()-t0)
                losses.append(float(loss.detach()))
                state.update(resolution=size,completed_updates=step+1,latest_step_seconds=timings[-1])
                save()
            update=float((selected.detach()-before).abs().max())
            assert update>0 and all(torch.isfinite(v).all() for v in model.state_dict().values())
            results[str(size)]={"original_constructor_transform":original_transform,"effective_transform":{"min_size":list(model.transform.min_size),"max_size":model.transform.max_size},"first_batch_geometry":geometry,"TRAIN_image_order":images_order,"update_count":8,"step_seconds":timings,"median_step_seconds_excluding_first":float(np.median(timings[1:])),"last_loss":losses[-1],"classifier_head_final_gradient_max":grad,"classifier_head_actual_update_max":update,"initial_weight_provenance":provenance,"process_high_water_RSS_bytes_macos":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"finite_weights":True}
            del model,optimizer,selected,before,terms,loss,images,targets,samples
            gc.collect()
        report={"passed":True,"scope":state["scope"],"fold_manifest_sha256":digest,"same_initial_state_tensor_hashes":initial_hashes,"all_initial_state_tensors_identical":len(initial_hashes),"results":results,"paired_640_to320_median_step_time_ratio":results["640"]["median_step_seconds_excluding_first"]/results["320"]["median_step_seconds_excluding_first"],"elapsed_seconds":time.perf_counter()-started,"calibration_or_modelvalidation_or_test_evaluated":False,"higher_resolution_quality_training_released":False,"VLP_or_production_released":False,"limitations":["8batches per arm,not throughput guarantee or efficacy evidence","RSS is process cumulative high-water mark,not isolated perarm allocation","timing includes TRAIN raster loading and firststep geometry audit excluded frommedian","all8updatedweights discarded;future comparison must reinitialize fromsamefreshCOCOstate,not this gate"]}
        (folder/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="passed_technical_timing_only",report_sha256=sha(folder/"report.json"))
        save()
        print(json.dumps({k:v for k,v in report.items() if k not in ("same_initial_state_tensor_hashes","results")},indent=2))
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=repr(error))
        save()
        raise


if __name__=="__main__":
    main()
