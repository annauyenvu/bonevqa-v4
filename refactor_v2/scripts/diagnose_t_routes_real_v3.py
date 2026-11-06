"""Parameter-free T routing sensitivity at common initialization, TRAIN only."""
import json
import time
from pathlib import Path

import torch

from scripts.diagnose_p_controls_real_v3 import state_hash
from src.data import build_dataset, collate_fn
from src.data.prompts import textual_prompt
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.train import build_split
from src.utils import load_config


def main():
    start = time.perf_counter()
    config = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    dataset = build_split(build_dataset, config["data"]["dataset"], "train", None, True,
                          config["data"]["loader_kwargs"])
    # Same four distinct TRAIN images as the previous mechanism intervention.
    indices = [37, 0, 8, 13]
    batch = normalize_batch(collate_fn([dataset[index] for index in indices]))
    if len(set(batch["image_ids"])) != 4:
        raise RuntimeError("Distinct training-image audit changed; inspect manifests")
    checkpoint = "refactor_v2/checkpoints/common_initialization_seed42_fold0_v3/initial.pt"
    model = BoneVQAModel.load(checkpoint, device="cpu").eval()
    before = state_hash(model)

    @torch.no_grad()
    def evaluate(style):
        model.cfg.update(use_textual_prompt=style != "raw", textual_prompt_style="clinical" if style == "raw" else style)
        core = model._encode_core(batch["images"], batch["questions"], batch["regions"],
                                  [None] * 4, model.cfg["prompt_kind"], False)
        expected = [q if style == "raw" else textual_prompt(q, region=None, style=style) for q in batch["questions"]]
        if core["q_texts"] != expected:
            raise RuntimeError("Text encoder did not receive the declared full question representation")
        prompts = [model._format_llm_prompt(q) for q in core["q_texts"]]
        if any(prompt.count("Question:") != 1 for prompt in prompts):
            raise RuntimeError("Duplicated question wrappers")
        prefix = model._llm_prefix(core)
        embeds, attention, labels = model._build_llm_inputs(prefix[2:], core["q_texts"][2:], batch["answers"][2:])
        logits = model.llm(inputs_embeds=embeds, attention_mask=attention).logits
        return {"closed": core["logits_closed"].detach(), "fracture": core["fracture_logit"].detach(),
                "open": logits[:, :-1][labels[:, 1:] != -100].float().detach(),
                "question_texts": core["q_texts"]}

    styles = {style: evaluate(style) for style in ("raw", "clinical", "irrelevant_matched", "clinical_paraphrase")}
    repeat = evaluate("clinical")
    real = styles["clinical"]
    differences = {style: {key: float((real[key] - tensors[key]).abs().max())
                          for key in ("closed", "fracture", "open")}
                   for style, tensors in styles.items()}
    repeat_diff = {key: float((real[key] - repeat[key]).abs().max()) for key in ("closed", "fracture", "open")}
    unchanged = before == state_hash(model)
    report = {"scope": "parameter-free T route at fresh common initialization; sensitivity, NOT efficacy or robustness",
              "checkpoint": checkpoint, "device": "cpu", "train_indices": indices,
              "optimizer_steps": 0, "trainable_state_unchanged": unchanged,
              "endpoint_max_differences_from_clinical": differences, "repeat_max_differences": repeat_diff,
              "full_question_texts_by_style": {style: value["question_texts"] for style, value in styles.items()},
              "matched_token_budget_report": "textual_token_budget_fold0_v3.json",
              "seconds": time.perf_counter() - start,
              "passed_technical_routing": unchanged and all(value == 0 for value in repeat_diff.values())
                                          and all(all(value > 0 for value in differences[style].values())
                                                  for style in ("raw", "irrelevant_matched", "clinical_paraphrase")),
              "paraphrase_robustness_verified": False, "accuracy_superiority_measured": False}
    Path("refactor_v2/reports/t_routes_real_initialization_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "full_question_texts_by_style"}, indent=2))
    if not report["passed_technical_routing"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
