"""Dataset provenance and split integrity gates."""

import hashlib
import json
import random
from collections import defaultdict
from typing import List

from torch.utils.data import Subset


IDENTITY_FIELDS = ("patient_id", "study_id", "component_id", "image_id")


def records_of(dataset) -> List[dict]:
    if isinstance(dataset, Subset):
        base = records_of(dataset.dataset)
        return [base[i] for i in dataset.indices]
    records = getattr(dataset, "records", None)
    if records is None:
        return []
    return list(records)


def identity(record: dict):
    for field in IDENTITY_FIELDS:
        value = record.get(field)
        if value not in (None, ""):
            return field, str(value)
    raise ValueError("Record thiếu patient/study/component/image identity.")


def assert_disjoint_splits(**splits):
    seen = {}
    for split_name, dataset in splits.items():
        for record in records_of(dataset):
            key = identity(record)
            prior = seen.get(key)
            if prior is not None and prior != split_name:
                raise RuntimeError(f"Leakage {key[0]}={key[1]} giữa {prior} và {split_name}.")
            seen[key] = split_name


def group_subsample(dataset, max_rows, seed):
    """Sample complete image/study groups instead of taking ordered QA rows."""
    if max_rows is None or max_rows >= len(dataset):
        return dataset
    records = records_of(dataset)
    if not records or len(records) != len(dataset):
        raise RuntimeError("Không thể group-subsample dataset thiếu record provenance.")
    groups = defaultdict(list)
    for idx, record in enumerate(records):
        groups[identity(record)].append(idx)
    keys = sorted(groups, key=lambda x: (x[0], x[1]))
    random.Random(seed).shuffle(keys)
    chosen = []
    for key in keys:
        rows = groups[key]
        if chosen and len(chosen) + len(rows) > max_rows:
            continue
        chosen.extend(rows)
        if len(chosen) >= max_rows:
            break
    if not chosen:
        raise RuntimeError("Group-aware subsampling tạo tập rỗng.")
    return Subset(dataset, sorted(chosen))


def manifest(dataset) -> dict:
    rows = records_of(dataset)
    identities = sorted({f"{k}:{v}" for k, v in map(identity, rows)})
    raw = json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return {
        "n_rows": len(rows),
        "n_groups": len(identities),
        "identities": identities,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate_prompt_protocol_config(cfg: dict) -> None:
    """Fail closed for the sealed v3 prompt-ablation protocol.

    Historical/pilot configurations do not opt in and remain loadable.  A v3
    run must declare its semantic module set explicitly so filenames such as
    ``None`` cannot silently mean "T-only", and V/L cannot consume benchmark
    annotations at automatic inference.
    """

    model = cfg.get("model", {})
    version = int(model.get("prompt_protocol_version", 0) or 0)
    if version < 3:
        return

    declared = str(model.get("prompt_variant") or "").upper()
    if not declared:
        raise RuntimeError("Prompt protocol v3 requires model.prompt_variant")
    if declared == "B":
        declared_set = set()
    else:
        declared_set = set(declared)
        unknown = declared_set - {"V", "L", "P", "T"}
        if unknown:
            raise RuntimeError(f"Unknown v3 prompt module(s): {sorted(unknown)}")

    actual = {
        key for key, enabled in (
            ("V", model.get("use_visual_prompt", False)),
            ("L", model.get("use_lens", False)),
            ("P", model.get("use_latent_prompt", False)),
            ("T", model.get("use_textual_prompt", False)),
        ) if enabled
    }
    if actual != declared_set:
        raise RuntimeError(
            f"prompt_variant={declared!r} but active modules are {''.join(sorted(actual)) or 'B'}"
        )

    if "L" in actual:
        if not model.get("lens_training_only", False):
            raise RuntimeError("v3 L requires original-image-only production inference")
        if float(model.get("lens_dropout", 0.0)) != 0.0:
            raise RuntimeError("v3 paired Lens delta requires lens_dropout=0 for a deterministic control")
    if "V" in actual:
        if float(model.get("qformer_dropout", 0.0) or 0.0) != 0.0:
            raise RuntimeError("v3 paired FAVP delta requires qformer_dropout=0")
        if model.get("visual_route") not in ("prefix_only", "fusion_only"):
            raise RuntimeError("v3 V must use one declared route; dual injection is not causal")
    if "P" in actual and model.get("prior_labels") and not model.get("prior_graph_path"):
        raise RuntimeError("Plain prior_labels are not LaPA prior fusion; provide a sealed graph or use []")
    if "P" in actual and model.get("latent_answer_bank_mode", "train") not in ("train", "empty", "random"):
        raise RuntimeError("v3 P latent_answer_bank_mode must be train, empty, or random")
    if model.get("latent_context_control", "conditioned") not in ("conditioned", "unconditioned", "shuffled"):
        raise RuntimeError("v3 latent_context_control must be conditioned, unconditioned, or shuffled")
    if "latent_context_control_eval" in model and model["latent_context_control_eval"] not in ("conditioned", "unconditioned"):
        raise RuntimeError("Scientific LF eval context must be per-image, never shuffled across evaluation rows")
    if "T" in actual and model.get("textual_prompt_style", "clinical") not in (
        "clinical", "clinical_paraphrase", "irrelevant", "irrelevant_matched",
    ):
        raise RuntimeError("v3 T textual_prompt_style is not a registered causal-control style")

    loader = cfg.get("data", {}).get("loader_kwargs", {})
    if actual & {"V", "L"}:
        if loader.get("with_masks", False) or loader.get("with_boxes", False):
            raise RuntimeError("Automatic v3 V/L must not consume annotation-derived ROI")
        if not loader.get("boxes_from") or not loader.get("boxes_as_masks", False):
            raise RuntimeError("Automatic v3 V/L requires detector proposals with boxes_as_masks=true")
    if model.get("use_region_in_prompt", False):
        raise RuntimeError("v3 automatic inference forbids annotation-derived region text")
