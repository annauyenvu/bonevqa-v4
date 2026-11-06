"""Fresh-process reload and TRAIN-only context/bank/control parity checks."""
import copy
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from src.checkpointing import compatible_state
from src.data import build_dataset
from src.models.bonevqa import BoneVQAModel
from src.protocol import records_of
from src.train import build_split
from src.utils import set_seed

ROOT = Path.cwd()
FOLDER = ROOT / "refactor_v2/reports/P_target_scale_gate_v3"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hashes(state):
    return {k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in state.items()}


@torch.no_grad()
def worker(label):
    path = FOLDER / (label + ".pt")
    assert not path.exists()
    report = json.loads((FOLDER/"report.json").read_text())
    checkpoint = FOLDER/"technical_only.pt"
    assert report["passed"] and sha(checkpoint) == report["checkpoint_sha256"]
    cfg = json.loads((ROOT/"refactor_v2/reports/fracatlas_scale_pilot_v3/fracatlas_B_roco_scale_pilot_v3.json").read_text())
    ds = build_split(build_dataset,cfg["data"]["dataset"],"train",None,True,cfg["data"]["loader_kwargs"])
    chosen, seen = [], set()
    for i,row in enumerate(records_of(ds)):
        if row["image_id"] not in seen:
            chosen.append(i)
            seen.add(row["image_id"])
            if len(chosen)==4:
                break
    samples = [ds[i] for i in chosen]
    images = [s["image"] for s in samples]
    questions = [s["question"] for s in samples]
    set_seed(42)
    model = BoneVQAModel.load(str(checkpoint),device="cpu").eval()
    before = tensor_hashes(model.trainable_state_dict())
    def core(order):
        result=model._encode_core([images[i] for i in order],[questions[i] for i in order],[None]*len(order),[None]*len(order),model.cfg["prompt_kind"],False)
        return {"closed":result["logits_closed"].cpu(),"fracture":result["fracture_logit"].cpu(),"type":result["type_logit"].cpu(),"GM":result["x_lp"].cpu(),"fused":result["fus"]["x_f"].cpu(),"prefix":model._llm_prefix(result).cpu()}
    batch = core([0,1,2,3])
    single = core([0])
    reordered = core([3,2,1,0])
    pred_batch=model.predict_batch({"images":images,"questions":questions},tra_chi_tiet=True)
    pred_single=model.predict_batch({"images":[images[0]],"questions":[questions[0]]},tra_chi_tiet=True)
    pred_reverse=model.predict_batch({"images":images[::-1],"questions":questions[::-1]},tra_chi_tiet=True)
    single_error={k:float((v[:1]-single[k]).abs().max()) for k,v in batch.items()}
    reverse_error={k:float((v-reordered[k].flip(0)).abs().max()) for k,v in batch.items()}
    assert tensor_hashes(model.trainable_state_dict())==before
    assert max(single_error.values())<=1e-4 and max(reverse_error.values())<=1e-4
    answers=[p["answer"] for p in pred_batch]
    assert answers[0]==pred_single[0]["answer"] and answers==[p["answer"] for p in pred_reverse][::-1]
    torch.save({"batch":batch,"bank":model.latent.answer_bank.cpu(),"weights":before,"indices":chosen,"answers":answers,"single_max_errors":single_error,"reverse_max_errors":reverse_error},path)


@torch.no_grad()
def main():
    started=time.perf_counter()
    output=FOLDER/"independent_verification.json"
    progress=FOLDER/"verification_progress.json"
    assert not output.exists() and not progress.exists()
    state={"status":"running","scope":"TRAIN-only technical;no efficacy or scientific training release"}
    progress.write_text(json.dumps(state,indent=2)+"\n")
    try:
        for label in ("reload_a","reload_b"):
            subprocess.run([sys.executable,__file__,"--worker",label],check=True)
        a,b=[torch.load(FOLDER/(label+".pt"),weights_only=False) for label in ("reload_a","reload_b")]
        reload_errors={k:float((v-b["batch"][k]).abs().max()) for k,v in a["batch"].items()}
        assert all(v==0 for v in reload_errors.values()) and torch.equal(a["bank"],b["bank"]) and a["weights"]==b["weights"] and a["answers"]==b["answers"]
        reference=ROOT/"refactor_v2/reports/fracatlas_scale_pilot_v3"
        init=torch.load(reference/"fracatlas_B_roco_scale_pilot_v3_initialization.pt",map_location="cpu",weights_only=False)
        config=json.loads((reference/"fracatlas_B_roco_scale_pilot_v3.json").read_text())
        checks, shared, full, banks = {},None,None,{}
        for name,mode,context in (("P","train","conditioned"),("P_random_bank","random","conditioned"),("P_unconditioned","train","unconditioned")):
            cfg=copy.deepcopy(init["config"])
            cfg.update(prompt_variant="P",use_latent_prompt=True,prior_labels=[],latent_answer_bank_mode=mode,latent_context_control=context,latent_context_control_eval=context)
            set_seed(42)
            model=BoneVQAModel(cfg,init["answer_vocab"],device="cpu").eval()
            compatible=compatible_state(model.trainable_state_dict(),init["state"],config["checkpoint_allow_mismatch_prefixes"])
            model.load_state_dict(compatible,strict=False)
            tensors=model.trainable_state_dict()
            assert all(torch.equal(tensors[k],v) for k,v in init["state"].items())
            hashes=tensor_hashes(tensors)
            if full is None:
                full=hashes
            else:
                assert full==hashes
            model.refresh_answer_bank()
            banks[name]=model.latent.answer_bank.cpu().clone()
            checks[name]={"all_shared_B_tensors_exact":len(init["state"]),"all_P_control_initial_tensors_identical":len(hashes),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"training_context":context,"evaluation_context":context,"bank_shape":list(banks[name].shape),"bank_sha256":hashlib.sha256(banks[name].numpy().tobytes()).hexdigest()}
            del model,tensors,compatible
            gc.collect()
        assert torch.equal(banks["P"],banks["P_unconditioned"])
        assert banks["P"].shape==banks["P_random_bank"].shape==(68,768)
        norm_error=float((banks["P"].norm(dim=-1)-banks["P_random_bank"].norm(dim=-1)).abs().max())
        assert norm_error<=1e-5 and not torch.equal(banks["P"],banks["P_random_bank"])
        assert torch.equal(a["bank"],banks["P"])
        report={"passed":True,"scope":state["scope"],"reload_endpoint_max_errors":reload_errors,"reload_bank_exact":True,"trainable_weights_unchanged_by_first_inference":True,"batch_single_max_errors":a["single_max_errors"],"batch_reverse_max_errors":a["reverse_max_errors"],"generated_answers_batch_invariant":True,"TRAIN_indices":a["indices"],"controls_initialization_checks":checks,"random_bank_row_norm_max_error":norm_error,"same_TRAIN_vocab_bank_reconstructed_after_reload":True,"reload_checkpoint_sha256":sha(FOLDER/"technical_only.pt"),"PREtask_initialization_sha256":sha(reference/"fracatlas_B_roco_scale_pilot_v3_initialization.pt"),"input_keys_only":["images","questions"],"elapsed_seconds":time.perf_counter()-started,"calibration_validation_test_used":False,"training_released":False,"pending":["parameter/compute budget disclosure and frozen bounded semantic-control protocol","matched-seed efficacy and honest later confirmation"],"limitations":["four TRAIN images only;not an exhaustive batch-invariance proof","random bank preserves norms and shape,not a perfect semantic-null guarantee","technical smoke checkpoint is not downstream comparison initialization"]}
        output.write_text(json.dumps(report,indent=2)+"\n")
        state.update(status="passed_technical_only",report_sha256=sha(output))
    except Exception as error:
        state.update(status="failed_preserved_no_auto_retry",failure=repr(error))
        raise
    finally:
        progress.write_text(json.dumps(state,indent=2)+"\n")


if __name__=="__main__":
    if len(sys.argv)>1:
        worker(sys.argv[2])
    else:
        main()
