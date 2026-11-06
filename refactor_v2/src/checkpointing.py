"""Fail-closed checkpoint manifests and compatibility checks."""

import hashlib
import json
from typing import Dict, Iterable, Mapping, Sequence

import torch


SCHEMA_VERSION = 2
PREPROCESSING_VERSION = "bonevqa-raster-letterbox-v2"


def stable_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def checkpoint_manifest(config: dict, answer_vocab: Sequence[str], state: Mapping[str, torch.Tensor]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture_hash": stable_hash(config),
        "answer_vocab_hash": stable_hash(list(answer_vocab)),
        "preprocessing_version": PREPROCESSING_VERSION,
        "tensor_shapes": {k: list(v.shape) for k, v in sorted(state.items())},
    }


def _allowed(key: str, prefixes: Iterable[str]) -> bool:
    return any(key.startswith(prefix) for prefix in prefixes)


def compatible_state(
    expected: Mapping[str, torch.Tensor],
    incoming: Mapping[str, torch.Tensor],
    allowed_mismatch_prefixes: Iterable[str] = (),
) -> Dict[str, torch.Tensor]:
    """Return compatible tensors or raise with an auditable mismatch list."""
    allowed = tuple(allowed_mismatch_prefixes)
    missing = sorted(k for k in expected if k not in incoming and not _allowed(k, allowed))
    unexpected = sorted(k for k in incoming if k not in expected and not _allowed(k, allowed))
    wrong_shape = sorted(
        k for k in incoming
        if k in expected and incoming[k].shape != expected[k].shape and not _allowed(k, allowed)
    )
    if missing or unexpected or wrong_shape:
        raise RuntimeError(
            "Checkpoint không tương thích: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}, wrong_shape={wrong_shape[:20]}"
        )
    return {
        k: v for k, v in incoming.items()
        if k in expected and v.shape == expected[k].shape
    }


def validate_manifest(payload: dict, config: dict, answer_vocab: Sequence[str], require_preprocessing=True):
    manifest = payload.get("manifest")
    if not manifest:
        raise RuntimeError("Checkpoint thiếu manifest v2; không được nạp im lặng.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Sai checkpoint schema: {manifest.get('schema_version')} != {SCHEMA_VERSION}")
    if manifest.get("architecture_hash") != stable_hash(config):
        raise RuntimeError("architecture_hash không khớp checkpoint.")
    if manifest.get("answer_vocab_hash") != stable_hash(list(answer_vocab)):
        raise RuntimeError("answer_vocab_hash không khớp checkpoint.")
    if require_preprocessing and manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise RuntimeError("preprocessing_version không khớp checkpoint.")
