"""Frozen-weight TRAIN-only P interventions; NOT matched-training efficacy."""
import hashlib
import json
import time
from pathlib import Path

import torch

from src.data import build_dataset, collate_fn
from src.endpoints import is_canonical_fracture_presence_question
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.protocol import records_of
from src.train import build_split
from src.utils import normalize_answer


def state_hash(model):
    result = hashlib.sha256()
    for key, value in sorted(model.trainable_state_dict().items()):
        result.update(key.encode())
        result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    start = time.perf_counter()
    checkpoint = Path("refactor_v2/checkpoints/gate_p_answer_route_real_v3/smoke.pt")
    run = json.loads((checkpoint.parent / "config.json").read_text())
    dataset = build_split(build_dataset, run["data"]["dataset"], "train", None, True,
                          run["data"]["loader_kwargs"])
    records, indices, identities = records_of(dataset), [], set()
    for answer in ("yes", "no"):
        index = next(i for i, row in enumerate(records)
                     if is_canonical_fracture_presence_question(row["question"])
                     and normalize_answer(row["answer"]) == answer)
        indices.append(index)
        identities.add(records[index]["image_id"])
    for index, row in enumerate(records):
        if row.get("answer_type", "").upper() == "OPEN" and row["image_id"] not in identities:
            indices.append(index)
            identities.add(row["image_id"])
            if len(indices) == 4:
                break
    if len(indices) != 4 or len(identities) != 4:
        raise RuntimeError("Need four distinct training images for shuffled-context audit")
    batch = normalize_batch(collate_fn([dataset[i] for i in indices]))
    model = BoneVQAModel.load(str(checkpoint), device="cpu").eval()
    before = state_hash(model)

    @torch.no_grad()
    def evaluate(bank="train", context="conditioned", zero=False):
        model.cfg.update(latent_answer_bank_mode=bank, latent_context_control=context, latent_force_zero=zero)
        model.answer_bank_ready = False
        core = model._encode_core(batch["images"], batch["questions"], batch["regions"],
                                  [None] * 4, model.cfg["prompt_kind"], False)
        prefix = model._llm_prefix(core)
        embeds, attention, labels = model._build_llm_inputs(prefix[2:], core["q_texts"][2:], batch["answers"][2:])
        logits = model.llm(inputs_embeds=embeds, attention_mask=attention).logits
        return {"bank": model.latent.answer_bank.detach().clone(),
                "gm": core["x_lp"].detach().clone(), "delta": core["fus"]["latent_delta"].detach().clone(),
                "closed": core["logits_closed"].detach().clone(),
                "fracture": core["fracture_logit"].detach().clone(),
                "open": logits[:, :-1][labels[:, 1:] != -100].detach().float().clone()}

    reference = evaluate()
    conditions = {"repeat": evaluate(), "random_bank": evaluate(bank="random"),
                  "bank_off": evaluate(bank="empty"), "unconditioned_LF": evaluate(context="unconditioned"),
                  "shuffled_LF": evaluate(context="shuffled"), "gate_zero": evaluate(zero=True)}
    controls = {}
    for name, tensors in conditions.items():
        controls[name] = {
            "endpoint_max_differences": {key: float((reference[key] - tensors[key]).abs().max())
                                         for key in ("closed", "fracture", "open")},
            "residual_row_max_difference": float((tensors["delta"] - tensors["delta"][:1]).abs().max()),
            "gm_row_max_difference": float((tensors["gm"] - tensors["gm"][:1]).abs().max()),
            "bank_shape": list(tensors["bank"].shape),
        }
    random_bank = conditions["random_bank"]["bank"]
    norm_error = float((reference["bank"].norm(dim=-1) - random_bank.norm(dim=-1)).abs().max())
    reference_variation = float((reference["delta"] - reference["delta"][:1]).abs().max())
    after = state_hash(model)
    passed = (before == after and reference_variation > 0
              and all(value == 0 for value in controls["repeat"]["endpoint_max_differences"].values())
              and controls["unconditioned_LF"]["residual_row_max_difference"] < 1e-7
              and all(any(value > 0 for value in controls[name]["endpoint_max_differences"].values())
                      for name in ("random_bank", "bank_off", "unconditioned_LF", "shuffled_LF", "gate_zero"))
              and norm_error < 1e-5)
    report = {"scope": "frozen two-step smoke weights, four distinct TRAIN images; interventions not efficacy",
              "selected_train_indices": indices, "distinct_images": len(identities), "device": "cpu",
              "reference_residual_row_max_difference": reference_variation,
              "random_bank_row_norm_max_error": norm_error,
              "trainable_state_unchanged": before == after, "trainable_state_sha256": before,
              "controls": controls, "passed_technical_interventions": passed,
              "matched_training_superiority_measured": False, "seconds": time.perf_counter() - start,
              "limitations": ["bank-off removes bank attention and is not compute matched",
                              "shuffled control is a batch cyclic derangement; all four image IDs are distinct",
                              "unconditioned/random controls retain parameters and attention tensor shapes, not measured wall-clock equality",
                              "no held-out accuracy, clinical metrics, or significance claim"]}
    Path("refactor_v2/reports/p_controls_real_interventions_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
