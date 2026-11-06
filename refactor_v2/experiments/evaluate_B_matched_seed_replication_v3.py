"""Evaluate frozen B downstream repeats; never pool seeds as independent patients."""
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.context_controls import load_component_lookup
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_split
from src.utils import compute_metrics, normalize_answer, set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    root=Path.cwd()
    folder=root/"refactor_v2/reports/B_matched_seed_replication_v3"
    parent=root/"refactor_v2/reports/fracatlas_scale_pilot_v3"
    assert json.loads((folder/"progress.json").read_text())["status"]=="four_runs_complete_pending_independent_evaluation"
    if any(" -m src.train " in row for row in subprocess.check_output(["ps","-axo","command"],text=True).splitlines()):
        raise RuntimeError("Training alive: do not compete")
    progress=folder/"independent_evaluation_progress.json"
    output=folder/"independent_evaluation.json"
    if progress.exists() or output.exists():
        raise RuntimeError("Existing evaluation must not be overwritten or duplicated")
    state={"status":"verifying","pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),"scope":"Matched DOWNSTREAM seed B initialization-chain stability on inspected development only"}
    def save():
        progress.write_text(json.dumps(state,indent=2)+"\n")
    save()
    started=time.perf_counter()
    try:
        gate=json.loads((folder/"execution_release.json").read_text())
        assert gate["passed"]
        seed42_path=parent/"independent_evaluation.json"
        assert sha(seed42_path)==gate["previous_seed42_evaluation_sha256"]
        seed42=json.loads(seed42_path.read_text())
        specs=[(seed,kind,f"B_{kind}_seed{seed}_bounded_v3") for seed in (43,44) for kind in ("fresh","roco")]
        cfg=json.loads((folder/(specs[0][2]+".json")).read_text())
        train=build_split(build_dataset,cfg["data"]["dataset"],"train",None,True,cfg["data"]["loader_kwargs"])
        pilot=build_split(build_dataset,cfg["data"]["dataset"],"model_val",None,True,cfg["data"]["loader_kwargs"])
        assert manifest(train)==gate["actual_manifests"]["train"] and manifest(pilot)==gate["actual_manifests"]["model_val"]
        base=json.loads((root/"refactor_v2/configs/revised_sealed_candidates_v3/b.json").read_text())
        full=build_split(build_dataset,base["data"]["dataset"],"model_val",None,True,base["data"]["loader_kwargs"])
        question="Is there a fracture in this X-ray?"
        endpoint_indices=[i for i,row in enumerate(records_of(full)) if row["question"]==question]
        assert len(endpoint_indices)==321 and len(pilot)==142
        bcfg=cfg["train"]["component_batching"]
        lookup,_=load_component_lookup(bcfg["manifest"],bcfg["seal"],root)
        first={}
        for i,row in enumerate(records_of(pilot)):
            first.setdefault(row["image_id"],i)
        ids=sorted(first)
        donors=dict(zip(ids,ids[1:]+ids[:1]))
        assert all(lookup[i]!=lookup[d] for i,d in donors.items())
        pixels={iid:pilot[index]["image"] for iid,index in first.items()}
        reference_q=json.loads((root/"refactor_v2/reports/fracatlas_baseline_pilot_v3/independent_paired_evaluation.json").read_text())["nonconstant_questions_defined_from_TRAIN"]
        def metrics(rows):
            result=compute_metrics(rows,fracture_threshold=0.5)
            subset=[row for row in rows if row["question"] in reference_q]
            result["shared_original_TRAIN_nonconstant_question_metrics"]=compute_metrics(subset,fracture_threshold=0.5)
            return result
        checks,results={},{}
        aggregate={"fresh":{"42":{}},"roco":{"42":{}}}
        for kind in aggregate:
            name=f"fracatlas_B_{kind}_scale_pilot_v3"
            aggregate[kind]["42"]={"fracture":seed42["full321_fracture"][name]["metrics"],"QA":seed42["pilot142_QA"][name+"/correct_image"]["metrics"]}
        for seed,kind,name in specs:
            run=root/"refactor_v2/checkpoints"/name
            done=json.loads((folder/(name+"_completion.json")).read_text())
            assert sha(folder/(name+".json"))==gate["config_sha256"][name]
            assert sha(run/"last.pt")==done["last_sha256"] and sha(run/"best.pt")==done["best_sha256"]
            saved_manifest=json.loads((run/"data_manifest.json").read_text())
            assert saved_manifest["train"]==gate["actual_manifests"]["train"] and saved_manifest["val"]==gate["actual_manifests"]["model_val"]
            last=torch.load(run/"last.pt",map_location="cpu",weights_only=False)
            optimizer=torch.load(run/"optimizer.pt",map_location="cpu",weights_only=False)
            assert finite_tree(last["state"]) and finite_tree(optimizer)
            assert last["extra"]["global_step"]==optimizer["global_step"]==296 and last["extra"]["epoch"]==4
            assert last["extra"]["base_seed"]==optimizer["base_seed"]==seed
            initial=torch.load(folder/(name+"_initialization.pt"),map_location="cpu",weights_only=False)["state"]
            assert {key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key,value in initial.items()}==gate["seed_gates"][str(seed)]["initial_tensor_hashes"][kind]
            updates={key:float((last["state"][key]-value).abs().max()) for key,value in initial.items() if key.startswith(("closed_head.","fracture_head."))}
            assert any(value>0 for key,value in updates.items() if key.startswith("closed_head.")) and any(value>0 for key,value in updates.items() if key.startswith("fracture_head."))
            checks[name]={"passed":True,"actual_seed":seed,"last_steps":296,"weights_optimizer_finite":True,"task_head_max_updates":updates}
            del initial,last,optimizer
            gc.collect()
            set_seed(seed)
            model=BoneVQAModel.load(str(run/"last.pt"),device="cpu").eval()
            assert not any(model.cfg.get(key) for key in ("use_visual_prompt","use_lens","use_latent_prompt","use_textual_prompt","use_region_in_prompt"))
            assert model.cfg["fracture_yes_threshold"]==0.5 and model.cfg.get("fracture_platt_slope") is None
            with torch.enable_grad():
                params=dict(model.named_parameters())
                selected=["qformer.query","prefix_proj.0.weight"]
                selected += [key for key in params if key.startswith("closed_head.") and key.endswith("weight")][-1:]
                selected += [key for key in params if key.startswith("fracture_head.") and key.endswith("weight")][-1:]
                out=model(collate_fn([train[i] for i in range(4)]),stage=3)
                grads=torch.autograd.grad(out["loss_answer"],[params[key] for key in selected],allow_unused=True)
                grad_qc={key:0.0 if value is None else float(value.abs().max()) for key,value in zip(selected,grads)}
                assert all(value>0 for value in grad_qc.values())
                checks[name]["isolated_TRAIN_answer_gradient_max"]=grad_qc
                del out,grads
            conditions={}
            for condition in ("correct_image","component_disjoint_wrong_image","full321_fracture"):
                ds=full if condition=="full321_fracture" else pilot
                indices=endpoint_indices if condition=="full321_fracture" else list(range(142))
                rows=[]
                for offset in range(0,len(indices),4):
                    batch=collate_fn([ds[i] for i in indices[offset:offset+4]])
                    images=batch["images"] if condition!="component_disjoint_wrong_image" else [pixels[donors[iid]] for iid in batch["image_ids"]]
                    pred=model.predict_batch({"images":images,"questions":batch["questions"]},tra_chi_tiet=True)
                    for i,item in enumerate(pred):
                        row={"image_id":batch["image_ids"][i],"question":batch["questions"][i],"gt":batch["answers"][i],"answer_type":batch["answer_types"][i],"pred":item["answer"]}
                        if "p_yes" in item:
                            row["p_yes"]=item["p_yes"]
                        rows.append(row)
                conditions[condition]={"metrics":metrics(rows),"records":rows}
                state.update(status="running",run_name=name,completed_condition=condition)
                save()
            results[name]=conditions
            aggregate[kind][str(seed)]={"fracture":conditions["full321_fracture"]["metrics"],"QA":conditions["correct_image"]["metrics"]}
            ref=seed42["full321_fracture"]["fracatlas_B_fresh_scale_pilot_v3"]["records"]
            assert [(r["image_id"],r["gt"]) for r in conditions["full321_fracture"]["records"]]==[(r["image_id"],r["gt"]) for r in ref]
            a={r["image_id"]:r for r in conditions["correct_image"]["records"] if r["question"]==question}
            b={r["image_id"]:r for r in conditions["component_disjoint_wrong_image"]["records"] if r["question"]==question}
            error=max(abs(row["p_yes"]-a[donors[iid]]["p_yes"]) for iid,row in b.items())
            assert error<=1e-6
            checks[name]["actual_donor_score_max_error"]=error
            del model
            gc.collect()
        summary={}
        for kind,byseed in aggregate.items():
            summary[kind]={}
            for cohort,keys in (("fracture",("fracture_auroc","fracture_auprc","fracture_sensitivity","fracture_specificity","fracture_balanced_accuracy","fracture_brier")),("QA",("overall_acc","open_acc","open_recall","bleu1"))):
                summary[kind][cohort]={}
                for key in keys:
                    values=[byseed[str(seed)][cohort][key] for seed in (42,43,44)]
                    summary[kind][cohort][key]={"per_seed":dict(zip((42,43,44),values)),"mean":statistics.mean(values),"sample_SD":statistics.stdev(values),"min":min(values),"max":max(values)}
        paired={key:{str(seed):aggregate["roco"][str(seed)]["fracture"][key]-aggregate["fresh"][str(seed)]["fracture"][key] for seed in (42,43,44)} for key in ("fracture_auroc","fracture_auprc","fracture_balanced_accuracy")}
        report={"scope":state["scope"],"new_results":results,"artifact_and_gradient_checks":checks,"all_seed_metrics":aggregate,"descriptive_seed_summary":summary,"paired_downstream_seed_differences":paired,
            "input_keys_only":["images","questions"],"donor_bijection":donors,"raw_threshold":0.5,"elapsed_seconds":time.perf_counter()-started,
            "limitations":["3downstream seeds, not3independent cohorts","ROCO upstream seed42fixed and routing type_head inherited, not isolated pretraining causality","same inspected development data; no independent clinical confirmation","no seed cherry-picking, pooled patient count or threshold fitting"],
            "calibration_or_test_used":False,"long_VLP_training_released":False}
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        state.update(status="complete_pending_interpretation",finished_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(output))
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=str(error),finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        raise


if __name__=="__main__":
    main()
