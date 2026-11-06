"""Two-step CPU wiring audit of GM+LF on sealed TRAIN rows, not efficacy."""
import copy
import json
import time
from pathlib import Path

import torch

from src.checkpointing import compatible_state
from src.data import build_dataset, collate_fn
from src.endpoints import is_canonical_fracture_presence_question
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import records_of
from src.train import build_split, seed_training_epoch
from src.utils import load_config, normalize_answer, set_seed


def main():
    start = time.perf_counter()
    config = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    folder = Path("refactor_v2/checkpoints/gate_p_answer_route_real_v3")
    if folder.exists():
        raise RuntimeError("Audit directory already exists; inspect prior artifacts before rerunning")
    data = build_split(build_dataset, config["data"]["dataset"], "train", None, True,
                       config["data"]["loader_kwargs"])
    rows = records_of(data)
    chosen = []
    for answer in ("yes", "no"):
        chosen.append(next(i for i, row in enumerate(rows)
                           if is_canonical_fracture_presence_question(row["question"])
                           and normalize_answer(row["answer"]) == answer))
    for i, row in enumerate(rows):
        if row.get("answer_type", "").upper() == "OPEN" and i not in chosen:
            chosen.append(i)
            if len(chosen) == 4:
                break
    if len(chosen) != 4:
        raise RuntimeError("Need positive/negative fracture and two real OPEN training rows")
    batch = collate_fn([data[i] for i in chosen])
    initial = torch.load("refactor_v2/checkpoints/common_initialization_seed42_fold0_v3/initial.pt",
                         map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(initial["config"])
    cfg.update(prompt_variant="P", use_latent_prompt=True, prior_labels=[],
               grad_checkpoint=False, llm_grad_checkpoint=False)
    set_seed(42)
    model = BoneVQAModel(cfg, initial["answer_vocab"], device="cpu").to("cpu")
    transferred = compatible_state(model.trainable_state_dict(), initial["state"],
                                   config["checkpoint_allow_mismatch_prefixes"])
    model.load_state_dict(transferred, strict=False)
    names = ("fusion.gamma_latent", "latent.latent", "fusion.lp_ca_l.norm_q.weight",
             "fusion.lp_ca_i.norm_q.weight", "fusion.lp_ca_mm.norm_q.weight")
    parameters = dict(model.named_parameters())
    before = {key: parameters[key].detach().clone() for key in names}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01)
    steps = []
    for step in range(2):
        seed_training_epoch(42, step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(batch, stage=3)
        # Auxiliary GM consistency must not be mistaken for answer routing.
        gradients = torch.autograd.grad(out["loss_answer"], [parameters[key] for key in names],
                                        retain_graph=True, allow_unused=True)
        answer_grad = {key: 0.0 if gradient is None else float(gradient.abs().max())
                       for key, gradient in zip(names, gradients)}
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        steps.append({"step": step + 1, "loss": float(out["loss"].detach()),
                      "answer_loss": float(out["loss_answer"].detach()),
                      "answer_only_gradient_max": answer_grad})
    updates = {key: float((parameters[key].detach() - before[key]).abs().max()) for key in names}
    nb = normalize_batch(batch)

    @torch.no_grad()
    def endpoints():
        model.eval()
        core = model._encode_core(nb["images"], nb["questions"], nb["regions"],
                                  [None] * 4, model.cfg["prompt_kind"], False)
        prefix = model._llm_prefix(core)
        # Only the OPEN rows' answer positions, not visual prefixes.
        embeds, attention, labels = model._build_llm_inputs(prefix[2:], core["q_texts"][2:], nb["answers"][2:])
        logits = model.llm(inputs_embeds=embeds, attention_mask=attention).logits
        return {"closed": core["logits_closed"].detach().cpu(),
                "fracture": core["fracture_logit"].detach().cpu(),
                "open_answer_logits": logits[:, :-1][labels[:, 1:] != -100].float().detach().cpu(),
                "latent_delta_norm": float(core["fus"]["latent_delta"].norm(dim=-1).mean())}

    normal = endpoints()
    model.cfg["latent_force_zero"] = True
    neutral = endpoints()
    model.cfg["latent_force_zero"] = False
    differences = {key: float((normal[key] - neutral[key]).abs().max())
                   for key in ("closed", "fracture", "open_answer_logits")}
    folder.mkdir(parents=True)
    model.save(str(folder / "smoke.pt"), {"kind": "two-step TRAIN-only wiring audit, not performance evidence"})
    config["model"] = cfg
    config["data"]["val_split"] = "train"  # double-load audit remains TRAIN-only
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    report = {"scope": "GM+LF only, TRAIN-only four rows, two steps; no control superiority or clinical claim",
              "device": "cpu", "selected_train_indices": chosen, "steps": steps,
              "selected_parameter_updates": updates, "gate_zero_endpoint_max_differences": differences,
              "latent_delta_norm": normal["latent_delta_norm"],
              "seconds": time.perf_counter() - start,
              "passed": all(value > 0 for value in updates.values())
                        and all(value > 0 for value in steps[-1]["answer_only_gradient_max"].values())
                        and all(value > 0 for value in differences.values())
                        and 0 < normal["latent_delta_norm"] < 1.0}
    Path("refactor_v2/reports/gate_p_answer_route_real_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
