"""v5 contract tests with lightweight deterministic encoders (no downloads)."""
import copy
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import src.models.bonevqa as bonevqa
from src.utils import set_seed

D, HEADS, SIZE = 32, 4, 64


class FakeVisual(nn.Module):
    def __init__(self, backbone="fake", d_model=D, lora_rank=4, image_size=SIZE, grad_checkpoint=False):
        super().__init__()
        self.kind, self.shared_clip, self.image_size = "fake", None, image_size
        self.patch = nn.Conv2d(3, d_model, 16, 16)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        for p in list(self.patch.parameters()) + [self.cls]:
            p.requires_grad = False
        self.lora_A = nn.Linear(d_model, 4, bias=False)
        self.lora_B = nn.Linear(4, d_model, bias=False)
        nn.init.normal_(self.lora_B.weight, std=0.02)
        self.head = nn.Linear(d_model, 16)
        self.head.weight.requires_grad = False
        self.head.bias.requires_grad = False

    def preprocess(self, images, device):
        arr = [torch.from_numpy(np.asarray(im.convert("RGB").resize((self.image_size, self.image_size)), dtype=np.float32) / 255.0)
               .permute(2, 0, 1) for im in images]
        return torch.stack(arr).to(device)

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], 1)
        t = t + self.lora_B(self.lora_A(t))
        return {"tokens": t, "embed": F.normalize(self.head(t[:, 0]), dim=-1)}


class FakeText(nn.Module):
    def __init__(self, visual=None, d_model=D, max_len=48, freeze=True):
        super().__init__()
        self.emb = nn.Embedding(997, d_model)
        self.emb.weight.requires_grad = False
        self.d_model = d_model

    def _ids(self, text):
        words = ["[cls]"] + str(text).lower().replace("?", " ").replace(".", " ").split()
        return [sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 997 for w in words][:16]

    def forward(self, texts, device):
        ids = [self._ids(t) for t in texts]
        L = max(len(i) for i in ids)
        mat = torch.zeros(len(ids), L, dtype=torch.long)
        mask = torch.zeros(len(ids), L, dtype=torch.bool)
        for r, i in enumerate(ids):
            mat[r, : len(i)] = torch.tensor(i)
            mask[r, : len(i)] = True
        tok = self.emb(mat.to(device))
        # contextual [cls] like a real BERT, so different texts (e.g. KG node names) get different embeds
        m = mask.to(device).unsqueeze(-1).float()
        tok = torch.cat([tok[:, :1] + (tok * m).sum(1, keepdim=True) / m.sum(1, keepdim=True), tok[:, 1:]], 1)
        return {"tokens": tok, "mask": mask.to(device), "embed": F.normalize(tok[:, 0], dim=-1)}

    @torch.no_grad()
    def encode_frozen(self, texts, device):
        return self.forward(texts, device)


@pytest.fixture(autouse=True)
def fake_encoders(monkeypatch):
    monkeypatch.setattr(bonevqa, "VisualEncoder", FakeVisual)
    monkeypatch.setattr(bonevqa, "TextEncoder", FakeText)


BASE = dict(d_model=D, n_heads=HEADS, n_query=4, image_size=SIZE, dropout=0.1, use_llm=False,
            use_visual_prompt=False, use_lens=False, use_latent_prompt=False, use_textual_prompt=False,
            use_fracture_head=True, prompt_protocol_version=3, grad_checkpoint=False, llm_grad_checkpoint=False)
VOCAB = ["yes", "no", "leg", "hand", "zero", "one"]
CAN = "Is there a fracture in this X-ray?"


def kg():
    from experiments.v5.bone_kg_v5 import build_bone_kg
    recs = [{"image_id": f"I{i}", "region": ["hand", "leg"][i % 2], "views": ["frontal"], "fractured": i % 2,
             "hardware": 0, "fracture_count": i % 2} for i in range(20)]
    return build_bone_kg(recs, min_cooccurrence=2)


def batch(n=4, seed=0, with_lens=False):
    rng = np.random.default_rng(seed)
    images, masks, lens = [], [], []
    for i in range(n):
        arr = (rng.random((80 + 8 * i, 96, 3)) * 255).astype(np.uint8)
        im = Image.fromarray(arr)
        images.append(im)
        m = np.zeros((1, im.height, im.width), dtype=np.uint8)
        m[0, 10:40, 20:60] = 1
        masks.append(m if i % 2 == 0 else np.zeros((0, im.height, im.width), dtype=np.uint8))
        lm = np.zeros((2, im.height, im.width), dtype=np.uint8)
        lm[0, 5:50, 5:50] = 1
        lm[1, 30:70, 40:90] = 1
        lens.append(lm)
    qs = [CAN, "Which body region is shown in this X-ray?", CAN, "How many fracture sites are visible?"][:n]
    ans = ["yes", "leg", "no", "zero"][:n]
    b = {"images": images, "questions": qs, "answers": ans, "answer_types": ["CLOSED"] * n,
         "regions": ["leg", "hand", "leg", "hand"][:n], "masks": masks, "boxes": [None] * n,
         "image_ids": [f"I{i}" for i in range(n)]}
    if with_lens:
        b["lens_masks"] = lens
    return b


def build(cls, **over):
    set_seed(7)
    cfg = dict(BASE)
    cfg.update(over)
    return cls(cfg, VOCAB, device="cpu")


def run_forward(m, b):
    set_seed(11)
    m.train()
    out = m(b, stage=3)
    return out["loss"].detach().clone()


def public(b):
    return {k: b[k] for k in ("images", "questions", "masks", "boxes")}


def test_all_off_equals_B():
    from experiments.v5.model_v5 import BoneVQAv5
    b = batch()
    mb, m5 = build(bonevqa.BoneVQAModel), build(BoneVQAv5)
    sb, s5 = mb.state_dict(), m5.state_dict()
    assert set(sb) == set(s5) and all(torch.equal(sb[k], s5[k]) for k in sb)
    assert torch.equal(run_forward(mb, b), run_forward(m5, b))
    # rows carry numpy masks, so compare the scalar outputs rather than whole dicts
    assert [(r["answer"], r.get("p_yes")) for r in mb.predict_batch(public(b))] == \
        [(r["answer"], r.get("p_yes")) for r in m5.predict_batch(public(b))]


def test_local_only_equals_v4():
    from experiments.v5.model_v5 import BoneVQAv5
    from experiments.vnext_model_v1 import VNextModel
    b = batch()
    m4, m5 = build(VNextModel, vnext_enabled=True, vnext_aux_weight=0.0), build(BoneVQAv5, v5_local=True)
    s4, s5 = m4.state_dict(), m5.state_dict()
    assert set(s4) == set(s5) and all(torch.equal(s4[k], s5[k]) for k in s4)
    assert torch.equal(run_forward(m4, b), run_forward(m5, b))
    p4 = [(r["answer"], r.get("p_yes")) for r in m4.predict_batch(public(b))]
    p5 = [(r["answer"], r.get("p_yes")) for r in m5.predict_batch(public(b))]
    assert p4 == p5


def _prefix(m, b):
    m.eval()
    masks = m._prepare_masks(b["images"], b["masks"], b["boxes"], b["image_ids"])
    with torch.no_grad():
        core = m._encode_core(b["images"], b["questions"], b["regions"], masks, "contour", False)
    return core["vis"]["prefix_vis"].clone()


def test_hag_feeds_local_tokens_into_qformer():
    from experiments.v5.model_v5 import BoneVQAv5
    b = batch()
    b2 = copy.deepcopy(b)
    arr = np.asarray(b2["images"][0]).copy()
    arr[10:40, 20:60] = 255 - arr[10:40, 20:60]          # change only the ROI pixels
    b2["images"][0] = Image.fromarray(arr)
    on, off = build(BoneVQAv5, v5_local=True, v5_hag="replace"), build(BoneVQAv5, v5_local=True)
    d_on = (_prefix(on, b) - _prefix(on, b2)).abs()
    d_off_row1 = (_prefix(off, b) - _prefix(off, b2))[1].abs().max()
    assert d_on[0].max() > 0
    assert d_off_row1 == 0                                  # unrelated row unchanged
    # an image without proposals gets exactly the B prefix under HAG
    base = build(BoneVQAv5)
    assert torch.allclose(_prefix(on, b)[1], _prefix(base, b)[1], atol=1e-5)


def test_lens_never_changes_inference():
    from experiments.v5.model_v5 import BoneVQAv5
    m = build(BoneVQAv5, v5_local=True, v5_hag="replace", v5_lens=True)
    b = batch(with_lens=True)
    before = [(r["answer"], r.get("p_yes")) for r in m.predict_batch(public(b))]
    with torch.no_grad():
        for p in m.v5_lens_fusion.parameters():
            p.add_(torch.randn_like(p))
    after = [(r["answer"], r.get("p_yes")) for r in m.predict_batch({**public(b), "lens_masks": b["lens_masks"]})]
    assert before == after


def test_gradients_reach_every_new_module():
    from experiments.v5.model_v5 import BoneVQAv5
    m = build(BoneVQAv5, v5_local=True, v5_hag="replace", v5_lens=True, v5_latent=True, v5_pf="graph", v5_kg=kg())
    b = batch(with_lens=True)
    m.train()
    (m(b, stage=3)["loss"] + m.forward_lens_align(b)["loss"]).backward()
    # (qformer/prefix only receive gradient through the LLM, absent in this fixture;
    # HAG routing into the Q-Former is covered by test_hag_feeds_local_tokens_into_qformer)
    names = ["local_tokens.attention.in_proj_weight", "v5_lens_fusion.compress.weight",
             "v5_gm.latent", "v5_lf.ca_l.attn.in_proj_weight", "v5_prior.gnn.0.w.weight",
             "v5_out_norm.weight", "visual.lora_B.weight", "fracture_head.net.0.weight"]
    params = dict(m.named_parameters())
    for n in names:
        g = params[n].grad
        assert g is not None and torch.count_nonzero(g) > 0, n


def test_prior_uses_graph_structure():
    from experiments.v5.bone_kg_v5 import degree_preserving_rewire
    from experiments.v5.model_v5 import BoneVQAv5
    g = kg()
    real = build(BoneVQAv5, v5_local=True, v5_latent=True, v5_pf="graph", v5_kg=g)
    rew = build(BoneVQAv5, v5_local=True, v5_latent=True, v5_pf="graph", v5_kg=degree_preserving_rewire(g))
    rew.load_state_dict(real.state_dict())
    b = batch()
    pr = [r.get("p_yes") for r in real.predict_batch(public(b))]
    pw = [r.get("p_yes") for r in rew.predict_batch(public(b))]
    assert pr != pw


def test_lens_fusion_shapes():
    from experiments.v5.modules_v5 import LensFusionV5
    x = torch.randn(2, 3, 1 + 14 * 14, D)
    assert LensFusionV5(D, heads=HEADS, grid=14, shuffle_r=4)(x).shape == (2, 1 + 16, D)
    assert LensFusionV5(D, heads=HEADS, grid=14, shuffle_r=2)(x).shape == (2, 1 + 49, D)


def test_save_load_roundtrip(tmp_path):
    from experiments.v5.model_v5 import BoneVQAv5
    m = build(BoneVQAv5, v5_local=True, v5_hag="replace", v5_latent=True, v5_pf="graph", v5_kg=kg())
    b = batch()
    ref = [(r["answer"], r.get("p_yes")) for r in m.predict_batch(public(b))]
    path = tmp_path / "v5.pt"
    m.save(str(path), {"model_class": "BoneVQAv5"})
    set_seed(7)  # fake frozen encoders are random; real ones load pretrained weights
    m2 = BoneVQAv5.load(str(path), device="cpu")
    assert [(r["answer"], r.get("p_yes")) for r in m2.predict_batch(public(b))] == ref


def test_full_model_overfits_tiny_set():
    from experiments.v5.model_v5 import BoneVQAv5
    m = build(BoneVQAv5, v5_local=True, v5_hag="replace", v5_lens=True, v5_latent=True, v5_pf="graph", v5_kg=kg())
    b = batch(with_lens=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    m.train()
    first = None
    for step in range(80):
        opt.zero_grad()
        loss = m(b, stage=3)["loss"]
        first = first if first is not None else float(loss)
        loss.backward()
        opt.step()
    assert float(loss) < 0.35 * first
    preds = [r["answer"] for r in m.predict_batch(public(b))]
    assert preds == b["answers"]


def test_kg_is_train_only_and_rewire_keeps_degrees():
    from experiments.v5.bone_kg_v5 import assert_train_only, degree_preserving_rewire, degrees
    g = kg()
    r = degree_preserving_rewire(g)
    assert degrees(g) == degrees(r) and g["edges"] != r["edges"]
    with pytest.raises(AssertionError):
        assert_train_only([{"image_id": "A"}], [{"image_id": "A"}])


def test_engine_detects_local_route_for_v4_and_v5():
    from experiments.v5.inference_v5 import uses_local_route
    assert uses_local_route({"vnext_enabled": True}) and uses_local_route({"v5_local": True})
    assert not uses_local_route({"use_visual_prompt": False})
