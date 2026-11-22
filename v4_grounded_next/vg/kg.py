"""Knowledge-graph latent prompt for the V4 model, without touching V4 code.

What is added
  * a bank of "knowledge cards": one English text per diagnosis (and per disease category) of
    kg_rgo/out/v2/bone_kg_v2.json, written from its reviewed edges and encoded once by the frozen
    PubMedBERT text tower (CLS state);
  * :class:`KGLatentPrompt`: learned latent queries read the question tokens, then the image tokens
    (global + proposal crops, as V4 fuses them), then retrieve from the card bank; their mean gives
    a gated residual ``gamma * tanh(W x)``;
  * the residual enters the SAME place as the original latent prompt of refactor_v2's
    ``MultimodalFusion``: the pooled representation before ``out_norm``, i.e. the vector x_f that the
    closed head, the fracture head, the type head and the Qwen prefix all read.

So the graph acts at the reasoning stage (after the image and the question are encoded), never in
the image encoder or the detector.  ``gamma`` starts at 0: at initialisation the model is exactly
the model without the graph (checked by scripts/kg_selftest.py).

Controls (``bank_mode``):
  kg        the real cards
  shuffled  each diagnosis name is paired with the facts of another diagnosis (wrong knowledge)
  random    the real bank after a fixed feature permutation and sign flip (same norms, no meaning)
  empty     no bank: the same new parameters, no knowledge (capacity control)

The model is built by the unchanged V4 protocol (``vg.trainer.build_model``) and then upgraded in
place by :func:`attach_kg`; V4 arms and their checkpoints are unaffected.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.layers import CrossAttnBlock, masked_mean

from .model import VNextGroundedModel, _isolated
from .paths import ROOT, read_json, sha256

KG_GRAPH = ROOT / "kg_rgo/out/v2/bone_kg_v2.json"
BANK_MODES = ("kg", "shuffled", "random", "empty")
CARD_MAX_TOKENS = 256          # context length of the BiomedCLIP text tower
EDGE_PROVENANCE = "assistant_draft_v0"   # reviewed edges; RGO gamut edges are left out of the cards
CONTROL_SEED = 20261116
SECTIONS = (  # (relation, strength or None, heading)
    ("may_cause", "specific", "Pathognomonic findings"),
    ("may_cause", "typical", "Typical findings"),
    ("may_cause", "possible", "Possible findings"),
    ("typical_location", None, "Location in bone"),
    ("typical_site", None, "Common sites"),
    ("typical_age", None, "Typical age"),
    ("differential_of", None, "Differential diagnosis"),
    ("may_be_complicated_by", None, "May be complicated by"),
)


# --------------------------------------------------------------------------- cards
def build_cards(graph_path=KG_GRAPH, mode: str = "kg") -> Tuple[List[dict], dict]:
    """One card per diagnosis and per category.  Returns (cards, info); a card is
    {"node": id, "type": ..., "name": English name, "text": the text that is encoded}."""
    assert mode in BANK_MODES
    g = read_json(graph_path)
    nodes = {n["id"]: n for n in g["nodes"]}
    out_edges: Dict[str, List[dict]] = {}
    members: Dict[str, List[str]] = {}
    for e in g["edges"]:
        if e["provenance"] != EDGE_PROVENANCE:
            continue
        s, t = nodes[e["source"]], nodes[e["target"]]
        if s["type"] == "diagnosis" and e["relation"] == "is_a":
            members.setdefault(t["id"], []).append(s["en"])
        elif s["type"] == "diagnosis":
            out_edges.setdefault(s["id"], []).append(e)
            if e["relation"] == "differential_of":   # stored once, read from both ends
                out_edges.setdefault(t["id"], []).append({**e, "source": t["id"], "target": s["id"]})
    category_of = {e["source"]: nodes[e["target"]]["en"] for e in g["edges"]
                   if e["relation"] == "is_a" and nodes[e["source"]]["type"] == "diagnosis"}
    cards = []
    for n in g["nodes"]:
        if n["type"] != "diagnosis":
            continue
        parts = [f"Category: {category_of[n['id']]}."]
        for rel, strength, head in SECTIONS:
            items = [nodes[e["target"]]["en"] for e in out_edges.get(n["id"], [])
                     if e["relation"] == rel and (strength is None or e["strength"] == strength)]
            if items:
                parts.append(f"{head}: {'; '.join(items)}.")
        cards.append({"node": n["id"], "type": "diagnosis", "name": n["en"], "facts": " ".join(parts)})
    for n in g["nodes"]:
        if n["type"] == "category" and members.get(n["id"]):
            cards.append({"node": n["id"], "type": "category", "name": n["en"],
                          "facts": f"Includes: {'; '.join(members[n['id']])}."})
    donors = list(range(len(cards)))
    if mode == "shuffled":   # a derangement among the diagnosis cards: every name gets another diagnosis's facts
        idx = [i for i, c in enumerate(cards) if c["type"] == "diagnosis"]
        rng = random.Random(CONTROL_SEED)
        while True:
            perm = idx[:]
            rng.shuffle(perm)
            if all(a != b for a, b in zip(idx, perm)):
                break
        for a, b in zip(idx, perm):
            donors[a] = b
    for i, c in enumerate(cards):
        c["facts_from"] = cards[donors[i]]["node"]
        c["text"] = f"{c['name']}. {cards[donors[i]]['facts']}"
    for c in cards:
        del c["facts"]
    info = {"graph": str(graph_path), "graph_sha256": sha256(graph_path), "graph_version": g["version"],
            "graph_content_sha256": g["sha256"], "cards": len(cards),
            "diagnosis_cards": sum(c["type"] == "diagnosis" for c in cards), "label_map": g["label_map"]}
    return cards, info


@torch.no_grad()
def encode_cards(text_encoder, cards: List[dict], device) -> torch.Tensor:
    """CLS state of the frozen PubMedBERT tower for every card.  No card may be truncated."""
    assert not any(p.requires_grad for p in text_encoder.model.parameters()), "the text tower must be frozen"
    tok = text_encoder.tokenizer
    texts = [c["text"] for c in cards]
    lengths = [len(x) for x in tok(texts, padding=False, truncation=False)["input_ids"]]
    assert max(lengths) <= CARD_MAX_TOKENS, f"card too long: {max(lengths)} tokens ({cards[lengths.index(max(lengths))]['node']})"
    was_training = text_encoder.model.training
    text_encoder.model.eval()
    try:
        rows = []
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], padding=True, truncation=True, max_length=CARD_MAX_TOKENS, return_tensors="pt")
            out = text_encoder.model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            rows.append(out.last_hidden_state[:, 0].float().cpu())
    finally:
        text_encoder.model.train(was_training)
    return torch.cat(rows, dim=0)


def placebo_bank(bank: torch.Tensor) -> torch.Tensor:
    """Fixed feature permutation and sign flip: every row keeps its norm, the semantics are destroyed."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(CONTROL_SEED)
    perm = torch.randperm(bank.shape[-1], generator=gen)
    signs = torch.randint(0, 2, (bank.shape[-1],), generator=gen).to(bank.dtype) * 2 - 1
    return bank[:, perm] * signs


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


# -------------------------------------------------------------------------- module
class KGLatentPrompt(nn.Module):
    def __init__(self, d_model: int, bank_dim: int, n_latent: int = 16, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
        self.ca_l = CrossAttnBlock(d_model, n_heads, dropout, ffn_mult=2)   # read the question
        self.ca_i = CrossAttnBlock(d_model, n_heads, dropout, ffn_mult=2)   # read the image tokens
        self.bank_proj = nn.Sequential(nn.Linear(bank_dim, d_model), nn.LayerNorm(d_model))
        self.norm_q = nn.LayerNorm(d_model)
        self.attn_kg = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)   # retrieval
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.gamma = nn.Parameter(torch.zeros(()))
        self.register_buffer("bank", torch.zeros(0, bank_dim), persistent=False)

    def forward(self, f_i: torch.Tensor, f_l: torch.Tensor, l_pad: Optional[torch.Tensor]):
        """f_i, f_l: image and question tokens after the fusion self-attention.  Returns the gated
        residual (B, d) and the retrieval weights over the cards (B, n_cards), or None without a bank."""
        b = f_i.shape[0]
        x = self.latent.to(f_i.dtype).unsqueeze(0).expand(b, -1, -1)
        x = self.ca_l(x, f_l, l_pad)
        x = self.ca_i(x, f_i)
        weights = None
        if self.bank.shape[0]:
            kv = self.bank_proj(self.bank.to(x.dtype)).unsqueeze(0).expand(b, -1, -1)
            a, w = self.attn_kg(self.norm_q(x), kv, kv, need_weights=True, average_attn_weights=True)
            x = x + self.drop(a)
            x = x + self.drop(self.ffn(self.norm2(x)))
            weights = w.mean(dim=1)
        return self.gamma * self.out(x.mean(dim=1)), weights


class VNextKGModel(VNextGroundedModel):
    """VNextGroundedModel + the KG residual on the final fusion call.  Never constructed directly:
    :func:`attach_kg` upgrades a model built by the V4 protocol."""

    def _kg_hook(self, fusion, args, output):
        self._kg_calls += 1
        if self._kg_calls != 2:     # call 1 is BoneVQAModel._encode_core (global tokens only); V4 discards its x_f
            return None
        l_mask = args[3]
        l_pad = None if l_mask is None else ~l_mask
        delta, weights = self.kg(output["f_i"], output["f_l"], l_pad)
        f_fi, f_fl, f_mm = output["f_fi"], output["f_fl"], output["x_ii"]
        mm_mask = None
        if l_mask is not None:
            mm_mask = torch.cat([torch.ones(f_fi.shape[:2], dtype=torch.bool, device=f_fi.device), l_mask], dim=1)
        pooled = masked_mean(f_mm, mm_mask)
        x_f = fusion.alpha * (pooled + delta.to(pooled.dtype)) + fusion.theta * f_fi[:, 0] + fusion.beta * f_fl[:, 0]
        self._kg_last = {"delta": delta, "weights": weights}
        return {**output, "x_f": fusion.out_norm(x_f), "latent_delta": delta}

    def _encode(self, nb: dict):
        self._kg_calls, self._kg_last = 0, None
        core = super()._encode(nb)
        assert self._kg_calls == 2, f"expected two fusion calls per forward, saw {self._kg_calls}"
        core["kg_delta"] = self._kg_last["delta"]
        core["kg_weights"] = self._kg_last["weights"]
        return core


def attach_kg(model: VNextGroundedModel, graph_path=KG_GRAPH, bank_mode: str = "kg", n_latent: int = 16,
              expect_bank_sha256: Optional[str] = None) -> dict:
    """Upgrade a V4 model in place.  New modules are created under an isolated RNG, so the base
    model's parameters and the global RNG stream are exactly those of the V4 protocol."""
    assert type(model) is VNextGroundedModel, type(model)
    c = model.cfg
    assert c["vnext_enabled"] and not c["vg_recovery"], "the KG residual targets the crop-path fusion of V4 / V4-CF"
    assert model.fusion.lp_out is None, "the base fusion must not already carry a latent prompt"
    cards, info = build_cards(graph_path, bank_mode)
    bank = encode_cards(model.text, cards, model.device)
    if bank_mode == "random":
        bank = placebo_bank(bank)
    elif bank_mode == "empty":
        bank = bank[:0]
    kg = _isolated(20261116, lambda: KGLatentPrompt(c["d_model"], bank.shape[1], n_latent, c["n_heads"], c["dropout"]))
    kg.bank = bank.to(model.device)
    model.kg = kg.to(model.device)
    model.__class__ = VNextKGModel
    model._kg_calls, model._kg_last = 0, None
    model.fusion.register_forward_hook(model._kg_hook)
    meta = {"graph": info["graph"], "graph_sha256": info["graph_sha256"], "graph_version": info["graph_version"],
            "bank_mode": bank_mode, "n_latent": n_latent, "cards": len(cards) if bank_mode != "empty" else 0,
            "bank_sha256": tensor_sha256(bank), "edge_provenance": EDGE_PROVENANCE}
    if expect_bank_sha256 is not None:
        assert meta["bank_sha256"] == expect_bank_sha256, "the card bank differs from the one this run was trained with"
    c["kg"] = meta
    model.kg_cards = cards
    model.kg_label_map = info["label_map"]
    return meta


def load_kg_arm(tag, fold, device):
    """scripts.mechanism_tests.load_arm, for runs with or without a graph (read from started.json)."""
    from .paths import RUNS
    from .trainer import build_model
    from .paths import seed_dir
    d = seed_dir(tag, fold)
    started, completed = read_json(d / "started.json"), read_json(d / "completed.json")
    ck = d / "LAST.pt"
    assert ck.exists(), f"{ck} missing (run trained with --no-save?)"
    model, _ = build_model(started["arm"], fold, device="cpu", overrides=started["overrides"])
    kg = started["model_cfg"].get("kg")
    if kg:
        attach_kg(model, kg["graph"], kg["bank_mode"], kg["n_latent"], expect_bank_sha256=kg["bank_sha256"])
    payload = torch.load(ck, map_location="cpu", weights_only=False)
    own = model.state_dict()
    res = model.load_state_dict({k: v for k, v in payload["state"].items() if k in own}, strict=False)
    assert not res.unexpected_keys and not (set(payload["state"]) - set(own))
    assert not kg or any(k.startswith("kg.") for k in payload["state"]), "checkpoint has no KG parameters"
    assert payload["extra"]["updates"] == completed["updates"]
    return model.to(device).eval(), started, sha256(ck)


def card_index(model, dataset: str, label: str) -> Optional[int]:
    """Row of the card that a dataset label maps to (for retrieval analysis), or None."""
    node = model.kg_label_map.get(dataset, {}).get(label)
    ids = [c["node"] for c in model.kg_cards]
    return ids.index(node) if node in ids else None
