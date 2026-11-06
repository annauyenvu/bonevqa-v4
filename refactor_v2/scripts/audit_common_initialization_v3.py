"""Audit actual pretrained-model shared initialization across prompt variants.

Construction only: no training, no image inference, no detector or performance
claim. Fold-0 train-only vocabulary is verified against the existing seal.
"""
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from src.data.sealed_manifest import load_sealed_records
from src.models.bonevqa import BoneVQAModel
from src.utils import load_config, set_seed


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path.cwd()
    config = load_config("refactor_v2/configs/fold0_b_sealed_v3.yaml")
    loader = config["data"]["loader_kwargs"]
    vocab_path = config["data"]["sealed_answer_vocab"]
    vocab = load_sealed_records(vocab_path, loader["seal_path"], root)
    directory = root / "refactor_v2/checkpoints/common_initialization_seed42_fold0_v3"
    checkpoint = directory / "initial.pt"
    if checkpoint.exists():
        raise RuntimeError("Initialization already exists; inspect its audit before any rerun")
    directory.mkdir(parents=True, exist_ok=True)
    variants = {"B": (False, False, False, False), "V": (True, False, False, False),
                "L": (False, True, False, False), "P": (False, False, True, False),
                "T": (False, False, False, True), "VP": (True, False, True, False)}
    reference, rows = None, []
    for name, flags in variants.items():
        start = time.perf_counter()
        model_config = copy.deepcopy(config["model"])
        model_config.update(prompt_variant=name, module_init_seed=config["seed"] + 1729,
                            llm_dtype="float32", grad_checkpoint=False, llm_grad_checkpoint=False)
        for key, enabled in zip(("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt"), flags):
            model_config[key] = enabled
        if model_config["prior_labels"]:
            raise RuntimeError("GM+LF scope forbids PF priors")
        set_seed(config["seed"])
        model = BoneVQAModel(model_config, vocab, device="cpu").to("cpu")
        state = model.trainable_state_dict()
        if reference is None:
            reference = {key: value.clone() for key, value in state.items()}
            model.save(str(checkpoint), {"seed": config["seed"], "fold": 0,
                                        "kind": "pretraining-free initialization audit, not trained evidence",
                                        "answer_vocab_sha256": sha256(vocab_path)})
        missing = sorted(set(reference) - set(state))
        differences = {key: float((reference[key].float() - state[key].float()).abs().max())
                       for key in reference if key in state and not torch.equal(reference[key], state[key])}
        rows.append({"variant": name, "shared_tensors": len(reference),
                     "missing_shared_tensors": missing, "different_shared_tensors": differences,
                     "additional_trainable_tensors": len(set(state) - set(reference)),
                     "construction_seconds": time.perf_counter() - start,
                     "passed": not missing and not differences})
        print(json.dumps(rows[-1]), flush=True)
        del model, state
        gc.collect()
    report = {"scope": "construction-only seed42 fold0; no trained checkpoint or prompt efficacy evidence",
              "seed": config["seed"], "device": "cpu", "vocabulary_entries": len(vocab),
              "vocabulary_sha256": sha256(vocab_path), "data_seal_sha256": sha256(loader["seal_path"]),
              "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": sha256(checkpoint),
              "base_cache_audit": "refactor_v2/reports/base_cache_hashes_v3.json",
              "base_cache_audit_sha256": sha256("refactor_v2/reports/base_cache_hashes_v3.json"),
              "variants": rows, "all_passed": all(row["passed"] for row in rows)}
    Path("refactor_v2/reports/common_initialization_seed42_fold0_v3.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
