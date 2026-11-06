"""Two bounded B repeat seeds, matched treatment heads/budgets; never chains VLP."""
import copy
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
from src.context_controls import ComponentBatchSampler, load_component_lookup
from src.data import build_dataset
from src.models.bonevqa import BoneVQAModel
from src.protocol import manifest, records_of
from src.train import build_answer_vocab, build_split, collect_answers, load_init_weights
from src.utils import set_seed
from experiments.evaluate_pmc_objective_pilot_v3_2 import finite_tree


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    root = Path.cwd()
    parent = root / "refactor_v2/reports/fracatlas_scale_pilot_v3"
    folder = root / "refactor_v2/reports/B_matched_seed_replication_v3"
    if folder.exists():
        raise RuntimeError("Existing replication requires inspection, never duplicate/overwrite")
    commands = subprocess.check_output(["ps", "-axo", "command"], text=True)
    if any(" -m src.train " in line for line in commands.splitlines()):
        raise RuntimeError("Existing training alive")
    if any(os.environ.get(k) != v for k,v in {"HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "OMP_NUM_THREADS":"8",
        "HF_HOME":str(root/".evaluation_runtime/hf_cache"), "BONEVQA_DATA":str(root/".evaluation_runtime/data")}.items()):
        raise RuntimeError("Pinned offline CPU8 environment changed")
    subprocess.run([sys.executable, "refactor_v2/scripts/seal_technical_reference_v3.py", "--verify"], check=True)
    evaluated = json.loads((parent / "independent_evaluation_progress.json").read_text())
    assert evaluated["status"] == "complete_pending_interpretation"
    assert sha(parent / "independent_evaluation.json") == evaluated["report_sha256"]
    evidence = json.loads((parent / "independent_evaluation.json").read_text())
    assert all(value["passed"] for value in evidence["artifact_and_gradient_verification"].values())
    # This gate qualifies only B for bounded stability checks, not clinical use or VLP.
    assert all(value["metrics"]["fracture_auroc"] > 0.5 for value in evidence["full321_fracture"].values())
    base = json.loads((parent / "fracatlas_B_fresh_scale_pilot_v3.json").read_text())
    loader = base["data"]["loader_kwargs"]
    assert loader["sealed_records_by_split"]["test"] is None and loader["sealed_records_by_split"]["calibration"] is None
    dataset = build_split(build_dataset, base["data"]["dataset"], "train", None, True, loader)
    expected = json.loads((parent / "execution_release.json").read_text())
    assert manifest(dataset) == expected["actual_manifests"]["train"] and len(dataset)==1173
    vocab = build_answer_vocab(collect_answers(dataset), base["data"]["answer_vocab_size"])
    assert vocab == json.loads((parent / "TRAIN_answer_vocab.json").read_text())
    upstream = json.loads((root/"refactor_v2/reports/roco_alignment_execution_v3/independent_verification.json").read_text())
    assert upstream["passed"] and sha(upstream["best_checkpoint"]) == upstream["best_checkpoint_sha256"]
    batch_cfg = base["train"]["component_batching"]
    lookup, _ = load_component_lookup(batch_cfg["manifest"], batch_cfg["seal"], root)
    folder.mkdir()
    progress = folder/"progress.json"
    state = {"status":"preparing", "driver_pid":os.getpid(), "started_utc":now(),
             "scope":"B-only bounded downstream-seed stability check,1173TRAIN/142developmentQA, not independent clinical evidence"}
    def save():
        progress.write_text(json.dumps(state,indent=2)+"\n")
    save()
    try:
        configs, gates = [], {}
        for seed in (43,44):
            sampler = ComponentBatchSampler([lookup[row["image_id"]] for row in records_of(dataset)],4,seed)
            for epoch in range(4):
                sampler.set_epoch(epoch)
                assert len(sampler)==294 and sorted(i for batch in sampler for i in batch)==list(range(1173))
            heads, hashes = None, {}
            for kind in ("fresh","roco"):
                name=f"B_{kind}_seed{seed}_bounded_v3"
                if (root/"refactor_v2/checkpoints"/name).exists():
                    raise RuntimeError("Checkpoint target already exists")
                set_seed(seed)
                model_cfg=copy.deepcopy(base["model"])
                model_cfg.update(module_init_seed=seed+1729,fracture_positive_weight=expected["TRAIN_positive_weight"])
                model=BoneVQAModel(model_cfg,vocab,device="cpu")
                if heads is None:
                    heads={key:value.clone() for key,value in model.trainable_state_dict().items() if key.startswith(("closed_head.","fracture_head."))}
                if kind=="roco":
                    load_init_weights(model,upstream["best_checkpoint"],base["checkpoint_allow_mismatch_prefixes"],False)
                model.load_state_dict(heads,strict=False)
                initial=model.trainable_state_dict()
                assert finite_tree(initial)
                hashes[kind]={key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key,value in initial.items()}
                init_path=folder/(name+"_initialization.pt")
                model.save(str(init_path),{"scope":"bounded_B_seed_replication_initialization", "downstream_seed":seed,"fixed_ROCO_upstream_seed":42})
                cfg=copy.deepcopy(base)
                cfg.update(seed=seed,run_name=name,init_from=str(init_path),recipe_status="released_B_only_bounded_seed_replication_v3",figure_dir=str(folder/name/"figures"))
                cfg["model"].update(fracture_yes_threshold=0.5,fracture_platt_slope=None,fracture_platt_intercept=None)
                cfg["model"].pop("module_init_seed",None)
                path=folder/(name+".json")
                path.write_text(json.dumps(cfg,indent=2)+"\n")
                configs.append((cfg,path))
                del model,initial
                gc.collect()
            assert len(heads)==8 and all(hashes["fresh"][key]==hashes["roco"][key] for key in heads)
            gates[str(seed)]={"new_head_tensors_byte_identical":8,"initial_tensor_hashes":hashes,"sampler_all4epochs_no_row_loss":True}
        release={"passed":True,"scope":state["scope"],"seeds":[42,43,44],"new_runs_seeds":[43,44],"previous_seed42_evaluation_sha256":evaluated["report_sha256"],
            "seed_gates":gates,"queue":[cfg["run_name"] for cfg,_ in configs],"config_sha256":{cfg["run_name"]:sha(path) for cfg,path in configs},
            "data_seal_sha256":sha(loader["seal_path"]),"TRAIN_vocab_sha256":sha(parent/"TRAIN_answer_vocab.json"),"actual_manifests":expected["actual_manifests"],
            "fixed_ROCO_upstream_sha256":upstream["best_checkpoint_sha256"],"fixed_ROCO_upstream_seed":42,
            "budget_each":"4epochs296steps1173TRAIN142modelvalQA,CPU8float32", "primary":"LAST296step checkpoints; full321development fracture endpoints plus same142QA controls",
            "limitations":["matched DOWNSTREAM seeds only; ROCO upstream seed remains42","pretraining compute not matched","development cohorts already inspected; three seeds do not create an independent test","no threshold fitting or clinical quality claim"],
            "calibration_reserved":True,"test_used":False,"long_VLP_or_detector_training_released":False}
        (folder/"execution_release.json").write_text(json.dumps(release,indent=2)+"\n")
        for cfg,path in configs:
            started=time.perf_counter()
            with (folder/(cfg["run_name"]+".log")).open("w") as log:
                child=subprocess.Popen([sys.executable,"-m","src.train","--config",str(path),"--device","cpu"],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                guard=subprocess.Popen(["caffeinate","-i","-w",str(child.pid)])
                state.update(status="running",run_name=cfg["run_name"],training_pid=child.pid,guard_pid=guard.pid,run_started_utc=now())
                save()
                code=child.wait()
            if code:
                raise RuntimeError(f"{cfg['run_name']} exit{code}; preserve log, no retry")
            run=root/cfg["checkpoint_dir"]/cfg["run_name"]
            last=torch.load(run/"last.pt",map_location="cpu",weights_only=False)
            optimizer=torch.load(run/"optimizer.pt",map_location="cpu",weights_only=False)
            history=json.loads((run/"history.json").read_text())
            assert len(history)==4 and last["extra"]["global_step"]==optimizer["global_step"]==296
            assert finite_tree(last["state"]) and finite_tree(optimizer)
            assert all(row["val_n"]==142 and row["skipped_no_supervision_batches"]==0 for row in history)
            done={"complete":True,"wall_seconds":time.perf_counter()-started,"history":history,"optimizer_steps":296,"last_sha256":sha(run/"last.pt"),"best_sha256":sha(run/"best.pt"),"quality_gate_passed":False}
            (folder/(cfg["run_name"]+"_completion.json")).write_text(json.dumps(done,indent=2)+"\n")
            del last,optimizer
            gc.collect()
            state.setdefault("completed_runs",[]).append(cfg["run_name"])
            save()
        state.update(status="four_runs_complete_pending_independent_evaluation",finished_utc=now(),long_VLP_training_released=False)
        save()
    except Exception as error:
        state.update(status="failed_preserved_no_automatic_retry",failure=str(error),finished_utc=now())
        save()
        raise


if __name__=="__main__":
    main()
