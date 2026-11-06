"""Two-update TRAIN-only GM+LF target-dimension wiring gate, never efficacy."""
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from src.checkpointing import compatible_state
from src.data import build_dataset, collate_fn
from src.endpoints import is_canonical_fracture_presence_question
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import manifest, records_of
from src.train import build_split, seed_training_epoch
from src.utils import normalize_answer, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.perf_counter()
    root = Path.cwd()
    folder = root / "refactor_v2/reports/P_target_scale_gate_v3"
    if folder.exists():
        raise RuntimeError("Prior artifacts exist; preserve and inspect, never overwrite")
    if any(" -m src.train " in row for row in subprocess.check_output(["ps", "-axo", "command"], text=True).splitlines()):
        raise RuntimeError("Training active; do not duplicate or compete")
    reference = root / "refactor_v2/reports/fracatlas_scale_pilot_v3"
    interpretation = root / "refactor_v2/reports/B_matched_seed_replication_v3/interpretation.json"
    decision = json.loads(interpretation.read_text())
    evaluation = interpretation.parent / "independent_evaluation.json"
    assert sha(evaluation) == decision["evaluation_sha256"]
    config = json.loads((reference / "fracatlas_B_roco_scale_pilot_v3.json").read_text())
    initial_path = reference / "fracatlas_B_roco_scale_pilot_v3_initialization.pt"
    release = json.loads((reference / "execution_release.json").read_text())
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    actual_hashes = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in initial["state"].items()}
    assert actual_hashes == release["initial_tensor_hashes"]["fracatlas_B_roco_scale_pilot_v3"]
    data = build_split(build_dataset, config["data"]["dataset"], "train", None, True, config["data"]["loader_kwargs"])
    rows = records_of(data)
    chosen = [next(i for i,r in enumerate(rows) if is_canonical_fracture_presence_question(r["question"]) and normalize_answer(r["answer"]) == answer) for answer in ("yes", "no")]
    for i,row in enumerate(rows):
        if row.get("answer_type", "").upper() == "OPEN" and i not in chosen:
            chosen.append(i)
            if len(chosen) == 4:
                break
    assert len(chosen) == 4
    batch = collate_fn([data[i] for i in chosen])
    cfg = copy.deepcopy(initial["config"])
    cfg.update(prompt_variant="P", use_latent_prompt=True, prior_labels=[], latent_answer_bank_mode="train", latent_context_control="conditioned", latent_context_control_eval="conditioned", grad_checkpoint=False, llm_grad_checkpoint=False)
    set_seed(42)
    model = BoneVQAModel(cfg, initial["answer_vocab"], device="cpu").to("cpu")
    transferred = compatible_state(model.trainable_state_dict(), initial["state"], config["checkpoint_allow_mismatch_prefixes"])
    model.load_state_dict(transferred, strict=False)
    state = model.trainable_state_dict()
    assert all(torch.equal(state[k],v) for k,v in initial["state"].items())
    assert cfg["prior_labels"] == [] and len(initial["answer_vocab"]) == 68
    folder.mkdir(parents=True)
    progress = {"status":"two_step_TRAIN_gate_running", "pid":os.getpid(), "scope":"GM+LF only, no PF;technical not efficacy", "initial_checkpoint_sha256":sha(initial_path)}
    (folder / "progress.json").write_text(json.dumps(progress, indent=2)+"\n")
    names = ("fusion.gamma_latent", "latent.latent", "fusion.lp_ca_l.norm_q.weight", "fusion.lp_ca_i.norm_q.weight", "fusion.lp_ca_mm.norm_q.weight")
    params = dict(model.named_parameters())
    before = {k:params[k].detach().clone() for k in names}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, betas=(0.9,0.98), weight_decay=0.01)
    steps = []
    for step in range(2):
        seed_training_epoch(42, step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch, stage=3)
        gradients = torch.autograd.grad(out["loss_answer"], [params[k] for k in names], retain_graph=True, allow_unused=True)
        grad = {k:0.0 if g is None else float(g.abs().max()) for k,g in zip(names,gradients)}
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps.append({"step":step+1,"answer_only_gradient_max":grad,"loss":float(out["loss"].detach())})
    updates = {k:float((params[k].detach()-before[k]).abs().max()) for k in names}
    nb = normalize_batch(batch)
    @torch.no_grad()
    def endpoints():
        model.eval()
        core = model._encode_core(nb["images"],nb["questions"],nb["regions"],[None]*4,model.cfg["prompt_kind"],False)
        prefix = model._llm_prefix(core)
        embeds,attention,labels = model._build_llm_inputs(prefix[2:],core["q_texts"][2:],nb["answers"][2:])
        logits = model.llm(inputs_embeds=embeds,attention_mask=attention).logits
        return {"closed":core["logits_closed"].cpu(),"fracture":core["fracture_logit"].cpu(),"open_answer_logits":logits[:,:-1][labels[:,1:]!=-100].float().cpu()}
    normal = endpoints()
    model.cfg["latent_force_zero"] = True
    neutral = endpoints()
    model.cfg["latent_force_zero"] = False
    differences = {k:float((normal[k]-neutral[k]).abs().max()) for k in normal}
    model.save(str(folder / "technical_only.pt"), {"scope":"two TRAIN-only updates;never production or ablation initialization"})
    report = {"scope":progress["scope"],"shared_initial_tensors_identical":len(initial["state"]),"TRAIN_manifest":manifest(data),"selected_TRAIN_indices":chosen,"vocab_size":len(initial["answer_vocab"]),"steps":steps,"updates":updates,"gate_zero_endpoint_max_differences":differences,"initial_checkpoint_sha256":sha(initial_path),"checkpoint_sha256":sha(folder/"technical_only.pt"),"elapsed_seconds":time.perf_counter()-started,"calibration_or_validation_or_test_used":False,"semantic_control_training_released":False,"pending":["independent double-load and batch-context-invariance","random-bank/unconditioned shared-initialization parity and semantic controls","scientific protocol release"],"passed":all(v>0 for v in updates.values()) and all(v>0 for v in steps[-1]["answer_only_gradient_max"].values()) and all(v>0 for v in differences.values())}
    (folder / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    progress.update(status="passed_technical_only" if report["passed"] else "failed_preserved",report_sha256=sha(folder/"report.json"))
    (folder / "progress.json").write_text(json.dumps(progress,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
