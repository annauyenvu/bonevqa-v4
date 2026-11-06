"""Stage2 answer-only bridge gradients and reload audit on pilot TRAIN QA."""
import hashlib
import argparse
import json
from pathlib import Path
import torch
from src.data import build_dataset, collate_fn
from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.train import build_split
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="refactor_v2/checkpoints/pmc_cpu_timing_v3")
    parser.add_argument("--output", default="refactor_v2/reports/pmc_cpu_timing_v3/bridge_verification.json")
    args = parser.parse_args()
    folder = Path(args.run_dir)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("Existing PMC bridge audit must not be overwritten")
    cfg = json.loads((folder / "config.json").read_text())
    ds = build_split(build_dataset, "pmc_vqa", "train", 32, True, cfg["data"]["loader_kwargs"])
    batch = collate_fn([ds[0], ds[1]])
    model = BoneVQAModel.load(str(folder / "last.pt"), device="cpu")
    set_seed(42)
    model.train()
    out = model(batch, stage=2)
    parameters = dict(model.named_parameters())
    names = ("qformer.query", "prefix_proj.0.weight")
    gradients = torch.autograd.grad(out["loss_answer"], [parameters[name] for name in names], allow_unused=True)
    maxima = {name: 0.0 if gradient is None else float(gradient.abs().max()) for name, gradient in zip(names, gradients)}
    nb = normalize_batch(batch)
    @torch.no_grad()
    def answer_logits(images):
        model.eval()
        core = model._encode_core(images, nb["questions"], nb["regions"], [None] * 2, model.cfg["prompt_kind"], False)
        inputs, attention, labels = model._build_llm_inputs(model._llm_prefix(core), core["q_texts"], nb["answers"])
        logits = model.llm(inputs_embeds=inputs, attention_mask=attention).logits
        return logits[:, :-1][labels[:, 1:] != -100].float().cpu()
    reference = answer_logits(nb["images"])
    repeated = answer_logits(nb["images"])
    swapped = answer_logits(list(reversed(nb["images"])))
    model.load_trainable(str(folder / "last.pt"))
    restored = answer_logits(nb["images"])
    errors = {"repeat": float((reference - repeated).abs().max()),
              "strict_reload": float((reference - restored).abs().max()),
              "image_swap": float((reference - swapped).abs().max())}
    report = {"scope": "two PMC TRAIN QA, technical stage2 answer bridge; no semantic superiority/clinical evidence",
        "checkpoint_sha256": hashlib.sha256((folder / "last.pt").read_bytes()).hexdigest(),
        "answer_only_gradient_max": maxima, "answer_logit_differences": errors,
        "passed": all(value > 0 for value in maxima.values()) and errors["repeat"] == 0
            and errors["strict_reload"] == 0 and errors["image_swap"] > 0}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
