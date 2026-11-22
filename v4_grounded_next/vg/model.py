"""VNextGroundedModel: V4 crop tokens + causal grounding head + contrast + recovery.

Inference contract: inputs are ONLY images, questions and detector proposals
derived from the image.  Any ground-truth key in an eval-mode batch raises
``AssertionError`` (see :data:`GT_KEYS`).

Tensor contract (B batch, N_g = 1 + 14*14 global tokens, K proposal slots,
T = 4 pooled tokens per crop, d = 768, L question tokens, H_p = W_p = 14):

    G        (B, N_g, d)    BiomedCLIP global tokens (CLS + 14x14 patches)
    Q        (B, L, d)      PubMedBERT question tokens, mask (B, L)
    C        (B, K, T, d)   crop tokens (2x2 pooled patch grid per crop)  | flag: slot embeddings
    valid    (B, K)         proposal slot validity; m(I) = valid.any(1)
    G1       (B, N_g, d)    G + gate * CrossAttn(G + W_q q, C)    (V4 LocalTokens, unchanged)
    box_map  (B, 1, Hp, Wp) score-weighted proposal footprint -- diagnostics only, NOT a head input
    ell      (B, 1, Hp, Wp) grounding logits = f(G1 patches, C content, Q);  M_hat = sigmoid(ell)
    z_les    (B, d)         sum_p M_hat_p G1_p / sum_p M_hat_p   <- G1 (.) M_hat spatial pooling
    G2       (B, N_g, d)    [G1_cls + t_les(z_les), G1_p + M_hat_p * U(G1_p)]   (t_les, U zero-init)
    x_f      (B, d)         MultimodalFusion(G2, Q).x_f   -> ClosedHead, FractureHead, LLM prefix
    logit_local = FractureHead(x_f) + h_les(z_les)       (vector read-out, zero-init; no M_hat scalars)
                  [train only, vg_xf_droppath=p: FractureHead route dropped per row with prob p
                   around its output bias, kept rows x 1/(1-p); h_les route never dropped]
    p_global = sigmoid(GlobalFractureHead(Fusion(G, Q).x_f))      (detector independent)
    alpha    = g_alpha(mean(G), s_max, n/K, unc(C), max M_hat, fallback)  in [0,1]
    p_final  = (1 - alpha) * sg(p_global) + alpha * p_local       (R only)

So x_f, and therefore every answer head, is a differentiable function of
M_hat through two routes (the z_les token and the per-patch emphasis); the
fracture logit has a third route, h_les(z_les), which also reads a pooled
feature vector.  No scalar summary of M_hat (max / mean / LSE) reaches the
fracture logit, and the detector box map is not an input of the grounding head.
The gate checks these routes with finite interventions on ell.
"""
from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.vnext_model_v1 import LocalTokens  # sealed V4 module, imported read-only
from src.models.bonevqa import BoneVQAModel
from src.models.heads import FracturePresenceHead
from src.utils import normalize_answer

from . import geometry
from .paths import CANONICAL_Q, guard_write

# Keys that carry privileged annotation.  'masks'/'boxes' are the ambiguous
# legacy V4 names and are refused in every mode: proposals use 'proposals'.
GT_KEYS = ("gt_masks", "gt_boxes", "gt_polygons", "segmentation", "contrast_crops", "masks", "boxes")
TRAIN_ONLY_KEYS = ("gt_masks", "contrast_crops")
PROPOSAL_SOURCES = ("detector_cf", "detector_full", "placebo", "none", "test_fixture")

VG_DEFAULTS: Dict[str, Any] = {
    "vg_variant": "V4",
    "vnext_enabled": True,            # V4 LocalTokens crop path
    "vg_ground": False,
    "vg_pool_init": "identity",  # identity (zero-init, V4-exact at init) | legacy
    "vg_contrast": False,
    "vg_recovery": False,
    "vg_recovery_fallback": "global",  # 'global' (alpha=0 when m=0) | 'grid'
    "vg_token_mode": "crop",           # 'crop' | 'flag'
    # Shortcut controls (defaults close both shortcuts):
    #   vg_ground_input  'vlq'      : M_hat from vision-language features only -- G1 patches,
    #                                 crop-token CONTENT C (cross-attention, no box geometry,
    #                                 no detector score) and Q.  The detector box map is never
    #                                 an input (it is still built, for diagnostics only).
    #                    'footprint': legacy -- score-weighted box map concatenated to the head.
    #   vg_pool_read     'feature'  : FractureHead sees M_hat only through pooled feature
    #                                 vectors z_les = sum_p M_hat_p G1_p / sum_p M_hat_p and
    #                                 G1_p + M_hat_p U(G1_p); no max/mean/LSE scalar of M_hat
    #                                 reaches the fracture logit or the lesion token.
    #                    'scalar'   : legacy -- LSE/max/mean evidence into t_les and the logit.
    "vg_ground_input": "vlq",
    "vg_pool_read": "feature",
    # Training-only sample-wise DropPath on the FractureHead(x_f) logit route, centred on
    # its output bias: a dropped row's fracture logit is b + h_les(z_les), i.e. it rests on
    # the M_hat-pooled feature alone; kept rows are rescaled by 1/(1-p) so the expected
    # training logit equals the eval logit.  Identity in eval() (and at p = 0).
    "vg_xf_droppath": 0.0,
    # Read-side controls of the M_hat -> z_les / t_les / emphasis pooling (feature read only):
    #   vg_read_gamma   1.0 : w = M_hat / sum M_hat (exact original formula).
    #                   g>1 : w = M_hat^g / sum M_hat^g, computed in log-space as
    #                         softmax(g * log sigmoid(ell)) (the direct power underflows).
    #   vg_read_detach  True: the read uses stop-gradient(M_hat): L_VQA / fracture BCE reach
    #                         G1 patches, h_les, t_les and emphasis but never the GroundingHead,
    #                         which is then trained by L_ground only.
    "vg_read_gamma": 1.0,
    "vg_read_detach": False,
    "vg_max_k": 4,
    "vg_grid": 14,
    "vg_loss_res": 56,
    "vg_hidden": 128,
    "vg_lse_r": 5.0,
    "vg_ell0": -3.0,
    "vg_lambda_ground": 1.0, "vg_lambda_dice": 1.0, "vg_lambda_bce": 1.0,
    "vg_lambda_contrast": 0.5, "vg_tau": 0.07, "vg_contrast_dim": 256,
    "vg_local_loss_weight": 0.5,
    "vg_global_loss_weight": 1.0,
}


def pool_patches(tokens: torch.Tensor, out: int = 2) -> torch.Tensor:
    """(N, 1+S*S, d) -> (N, out*out, d) block mean.  Device-agnostic replacement
    for F.adaptive_avg_pool2d (MPS rejects non-divisible / some strided cases)."""
    patches = tokens[:, 1:]
    n, s2, d = patches.shape
    s = int(round(math.sqrt(s2)))
    assert s * s == s2 and s % out == 0, (s2, out)
    k = s // out
    x = patches.reshape(n, out, k, out, k, d)
    return x.mean(dim=(2, 4)).reshape(n, out * out, d).contiguous()


class GroundingHead(nn.Module):
    """f_ground(G1 patches, C content, Q) -> grounding logits (B,1,Hp,Wp).

    Patch queries (modulated by Q) cross-attend to the crop tokens C, so a patch
    can be marked because its features match a proposal's visual content -- not
    because it lies inside a box.  C carries no coordinates and no score.  With
    ``use_footprint`` (legacy) the score-weighted box map is appended instead."""

    def __init__(self, d: int, hidden: int, grid: int, ell0: float, n_heads: int = 8, use_footprint: bool = False):
        super().__init__()
        self.grid, self.use_footprint = grid, use_footprint
        self.norm = nn.LayerNorm(d)
        self.q_mod = nn.Linear(d, d)
        nn.init.zeros_(self.q_mod.weight); nn.init.zeros_(self.q_mod.bias)
        self.c_norm = nn.LayerNorm(d)
        self.c_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.reduce = nn.Linear(d, hidden)
        self.conv = nn.Sequential(
            nn.Conv2d(hidden + int(use_footprint), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.conv[-1].bias, ell0)

    def forward(self, patches, q, crops=None, valid=None, footprint=None):
        b, n, d = patches.shape
        h = self.norm(patches) * (1.0 + torch.tanh(self.q_mod(q))).unsqueeze(1)
        if crops is not None and valid is not None:
            kt = crops.reshape(b, -1, d)                                    # (B, K*T, d)
            vt = valid.unsqueeze(-1).expand(-1, -1, crops.shape[2]).reshape(b, -1)
            has = vt.any(1)
            pad = ~vt & has.unsqueeze(1)                                    # no all-masked rows
            kn = self.c_norm(kt)
            a, _ = self.c_attn(h, kn, kn, key_padding_mask=pad, need_weights=False)
            h = h + a * has[:, None, None].to(a.dtype)                      # m(I)=0 -> no crop term
        h = self.reduce(h).transpose(1, 2).reshape(b, -1, self.grid, self.grid).contiguous()
        if self.use_footprint:
            assert footprint is not None
            h = torch.cat([h, footprint], dim=1)
        return self.conv(h)


class LesionPooling(nn.Module):
    """The causal M_hat -> x_f link."""

    def __init__(self, d: int, lse_r: float, ell0: float, init: str = "identity", read: str = "feature",
                 gamma: float = 1.0, detach: bool = False):
        super().__init__()
        assert init in ("identity", "legacy") and read in ("feature", "scalar")
        assert gamma >= 1.0 and (read == "feature" or (gamma == 1.0 and not detach))
        self.r, self.ell0, self.init, self.read = lse_r, ell0, init, read
        self.gamma, self.detach = float(gamma), bool(detach)
        self.token = nn.Linear(d, d)
        self.evidence = nn.Linear(3, d)
        self.emphasis = nn.Linear(d, d)
        nn.init.normal_(self.emphasis.weight, std=0.02); nn.init.zeros_(self.emphasis.bias)
        if init == "identity":
            # Zero-init output maps (LoRA-B / zero-conv style): G2 == G1 and x_f == V4 at
            # init; the M_hat -> x_f routes open as soon as these weights move.
            for lin in (self.token, self.evidence, self.emphasis):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)

    def lse(self, ell_flat):
        return (torch.logsumexp(self.r * ell_flat, dim=1) - math.log(ell_flat.shape[1])) / self.r

    def forward(self, g1, ell):
        b = g1.shape[0]
        ell_flat = ell.reshape(b, -1)
        m_hat = torch.sigmoid(ell_flat)
        patches = g1[:, 1:]
        ell_read = ell_flat.detach() if self.detach else ell_flat
        m_read = m_hat.detach() if self.detach else m_hat
        if self.read == "feature" and self.gamma == 1.0:   # spatial pooling of G1 (.) M_hat, mass-normalised
            w = m_read / m_read.sum(1, keepdim=True).clamp_min(1e-6)
        elif self.read == "feature":  # sharpened: M_hat^g / sum M_hat^g in log-space
            w = torch.softmax(self.gamma * F.logsigmoid(ell_read), dim=1)
        else:
            w = torch.softmax(ell_flat, dim=1)
        z_les = torch.einsum("bp,bpd->bd", w, patches)
        self.last_w = w.detach()  # diagnostics (sharpen_readout_test)
        # Diagnostics only under read='feature' (never enters t_les or the logit).
        evidence = torch.stack([self.lse(ell_flat) - self.ell0, m_hat.max(1).values, m_hat.mean(1)], dim=1)
        t_les = self.token(z_les)
        if self.read == "scalar":
            t_les = t_les + self.evidence(evidence)
        emphasized = patches + m_read.unsqueeze(-1) * self.emphasis(patches)
        if self.init == "identity":  # lesion summary enters as a CLS residual (no extra token)
            g2 = torch.cat([(g1[:, 0] + t_les).unsqueeze(1), emphasized], dim=1)
        else:  # legacy: extra token -> perturbs fusion self-attention at init
            g2 = torch.cat([g1[:, :1], t_les.unsqueeze(1), emphasized], dim=1)
        return g2, z_les, evidence


class RecoveryGate(nn.Module):
    """alpha = g_alpha(mean(G), s_max, n/K, unc(C), max M_hat, is_fallback)."""

    def __init__(self, d: int):
        super().__init__()
        self.g = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 32))
        self.mlp = nn.Sequential(nn.Linear(32 + 5, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, g_mean, scalars):
        return torch.sigmoid(self.mlp(torch.cat([self.g(g_mean), scalars], dim=1))).squeeze(-1)


def _isolated(seed, factory):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


class VNextGroundedModel(BoneVQAModel):
    def __init__(self, cfg=None, answer_vocab=None, device=None):
        merged = copy.deepcopy(VG_DEFAULTS)
        merged.update(cfg or {})
        super().__init__(merged, answer_vocab, device)
        c = self.cfg
        assert not any(c[x] for x in ("use_visual_prompt", "use_lens", "use_latent_prompt", "use_textual_prompt"))
        assert c["vg_token_mode"] in ("crop", "flag") and c["vg_recovery_fallback"] in ("global", "grid")
        assert self.fracture_head is not None, "use_fracture_head must be on"
        if (c["vg_ground"] or c["vg_contrast"] or c["vg_recovery"]):
            assert c["vnext_enabled"], "G/C/R extend the V4 crop path"
        d = c["d_model"]
        if c["vnext_enabled"]:
            # Identical seed/class to sealed VNextModel -> exact V4 parity when G=C=R=off.
            self.local_tokens = _isolated(20261025, lambda: LocalTokens(d, c["n_heads"]))
        if c["vg_token_mode"] == "flag":
            self.slot_embed = _isolated(20261111, lambda: nn.Parameter(torch.randn(c["vg_max_k"], 4, d) * 0.02))
        if c["vg_ground"]:
            assert c["vg_ground_input"] in ("vlq", "footprint") and c["vg_pool_read"] in ("feature", "scalar")
            self.ground_head = _isolated(20261112, lambda: GroundingHead(
                d, c["vg_hidden"], c["vg_grid"], c["vg_ell0"], c["n_heads"], c["vg_ground_input"] == "footprint"))
            self.lesion_pool = _isolated(20261113, lambda: LesionPooling(
                d, c["vg_lse_r"], c["vg_ell0"], c["vg_pool_init"], c["vg_pool_read"],
                c["vg_read_gamma"], c["vg_read_detach"]))
            if c["vg_pool_read"] == "scalar":
                self.w_map = nn.Parameter(torch.tensor(0.0 if c["vg_pool_init"] == "identity" else 0.5))
            else:  # vector read-out of the pooled lesion feature, zero-init (identity at init)
                def _les_head():
                    h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
                    nn.init.zeros_(h[1].weight); nn.init.zeros_(h[1].bias)
                    return h
                self.les_head = _isolated(20261118, _les_head)
        if c["vg_contrast"]:
            self.contrast_img = _isolated(20261114, lambda: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, c["vg_contrast_dim"])))
            self.contrast_query = _isolated(20261115, lambda: nn.Linear(d, c["vg_contrast_dim"]))
        if c["vg_recovery"]:
            self.global_fracture_head = _isolated(20261116, lambda: FracturePresenceHead(d, c["dropout"]))
            self.recovery_gate = _isolated(20261117, lambda: RecoveryGate(d))
        self._intervention = None
        self._fp32_guard()

    # ------------------------------------------------------------------ guards
    def _fp32_guard(self):
        bad = [n for n, p in self.named_parameters() if p.is_floating_point() and p.dtype != torch.float32]
        assert not bad, f"fp32-only contract (MPS half/bf16 backward is unsafe): {bad[:5]}"

    def check_batch(self, batch: dict):
        present = [k for k in GT_KEYS if k in batch and batch[k] is not None and any(v is not None for v in batch[k])]
        if not self.training:
            assert not present, f"Leakage guard: ground-truth keys passed to model in eval mode: {present}"
        for k in ("masks", "boxes"):
            assert k not in present, f"Ambiguous legacy key '{k}': use 'proposals' (detector) or gt_* (train-only)"
        for src in batch.get("proposal_source") or []:
            assert src in PROPOSAL_SOURCES, f"proposal_source {src!r} not allowed (GT-derived proposals forbidden)"

    @contextmanager
    def intervene(self, ell: Optional[torch.Tensor] = None, crop_tokens: str = None):
        """Mechanistic interventions on internal tensors (never on inputs/GT)."""
        self._intervention = {"ell": ell, "crop_tokens": crop_tokens}
        try:
            yield
        finally:
            self._intervention = None

    def sync_global_head(self):
        """Initialise the detector-independent head from the (B/PREtask) fracture head."""
        if self.cfg["vg_recovery"]:
            self.global_fracture_head.load_state_dict(self.fracture_head.state_dict())

    # ------------------------------------------------------------------ encoders
    def _crop_tokens(self, images, proposals):
        c = self.cfg
        b, kmax, d = len(images), c["vg_max_k"], c["d_model"]
        dev = self.device
        feats = torch.zeros(b, kmax, 4, d, device=dev)
        valid = torch.zeros(b, kmax, dtype=torch.bool, device=dev)
        scalars = torch.zeros(b, 3, device=dev)  # s_max, n/K, is_fallback
        foot = torch.zeros(b, 1, c["vg_grid"], c["vg_grid"], device=dev)
        crops, slots, boxes_used = [], [], []
        for i, (im, props) in enumerate(zip(images, proposals)):
            props = [p for p in (props or []) if p[2] > p[0] and p[3] > p[1]][:3]
            fallback = False
            if not props and c["vg_recovery"] and c["vg_recovery_fallback"] == "grid":
                gray = np.asarray(im.convert("L"))
                props = [bx + [0.0] for bx in geometry.multiscale_grid_boxes(gray)]
                fallback = True
            boxes_used.append(props)
            row = []
            for p in props[:kmax]:
                x1, y1, x2, y2 = [int(round(v)) for v in p[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(im.width, max(x1 + 1, x2)), min(im.height, max(y1 + 1, y2))
                row.append(len(crops)); crops.append(im.crop((x1, y1, x2, y2)))
            slots.append(row)
            scores = [float(p[4]) if len(p) > 4 else 1.0 for p in props]
            scalars[i, 0] = 0.0 if fallback or not scores else max(scores)
            scalars[i, 1] = len(row) / kmax
            scalars[i, 2] = float(fallback)
            if c["vg_ground"] and c["vg_token_mode"] == "crop" and props:  # box map: diagnostics / legacy input
                fp = geometry.boxes_to_grid([p[:4] for p in props], im.size, c["image_size"], c["vg_grid"],
                                            weights=[max(s, 0.25) for s in scores])
                foot[i, 0] = torch.from_numpy(fp).to(dev)
        for i, row in enumerate(slots):
            for j, _ in enumerate(row):
                valid[i, j] = True
        mode = (self._intervention or {}).get("crop_tokens") or c["vg_token_mode"]
        if mode == "flag":
            feats = self.slot_embed.unsqueeze(0).expand(b, -1, -1, -1) * valid[:, :, None, None]
        elif crops:
            raw = self.visual(self.visual.preprocess(crops, dev))["tokens"]
            pooled = pool_patches(raw, 2)
            for i, row in enumerate(slots):
                for j, slot in enumerate(row):
                    feats[i, j] = pooled[slot]
        return feats, valid, scalars, foot, boxes_used

    def _local_tokens_active_only(self, G, text, text_mask, feats, valid):
        """Sealed LocalTokens applied to the rows that have a proposal only.

        LocalTokens multiplies its output by ``active`` (= valid.any(1)), so for a row
        with no proposal g1 == G and the row contributes exactly zero gradient to
        LocalTokens.  Evaluating the module on such rows anyway is harmless on CPU but
        not on MPS: with every row inactive the backward pass through the masked
        attention over all-zero crop memory returns arbitrary values (up to FLT_MAX)
        for ``local_tokens.norm`` instead of zero (runs/arms/fold0/V4-G nonfinite dump;
        scripts/replay_nonfinite.py: CPU 0.0, MPS 6.2 / 12.6 / 3.4e38).  Every operation
        in LocalTokens is per-row, so restricting it to active rows is the same function
        with the same (correct) gradients on every device."""
        active = valid.any(1)
        if bool(active.all()):
            return self.local_tokens(G, text, text_mask, feats, valid)
        delta = torch.zeros_like(G)
        aux = self.local_tokens.aux_head.bias.to(G.dtype).expand(G.shape[0]).clone()  # aux_head(0)
        if bool(active.any()):
            idx = torch.nonzero(active).squeeze(1)
            _, aux_a, act_a, delta_a = self.local_tokens(G[idx], text[idx], text_mask[idx], feats[idx], valid[idx])
            assert bool(act_a.all())
            # scatter by a one-hot matmul (index_copy / index_put backward are not on MPS)
            sel = F.one_hot(idx, G.shape[0]).to(G.dtype).t()            # (B, n_active)
            delta = torch.einsum("ba,and->bnd", sel, delta_a.to(G.dtype))
            aux = aux * (~active).to(aux.dtype) + sel @ aux_a.to(aux.dtype)
        return G + delta, aux, active, delta

    def _encode(self, nb: dict):
        c = self.cfg
        images, questions = nb["images"], nb["questions"]
        b = len(images)
        empty = [np.zeros((0, im.height, im.width), dtype=np.uint8) for im in images]
        core = BoneVQAModel._encode_core(self, images, questions, [None] * b, empty, c["prompt_kind"],
                                         self.training, context_group_ids=nb.get("component_ids"))
        core["x_global"] = core["fus"]["x_f"]
        if not c["vnext_enabled"]:
            core["logit_local"] = core["fracture_logit"]
            return core
        G = core["vis"]["global_tokens"]
        txt = core["txt"]
        feats, valid, scalars, foot, boxes_used = self._crop_tokens(images, nb.get("proposals") or [[]] * b)
        if c["vg_token_mode"] == "flag":
            foot = torch.zeros_like(foot)  # count-only control: no geometry
        g1, aux, active, delta = self._local_tokens_active_only(G, txt["tokens"], txt["mask"], feats, valid)
        if c.get("vnext_force_zero", False):
            g1 = G
        tokens = g1
        core.update(active=active, valid=valid, prop_scalars=scalars, boxes_used=boxes_used, local_delta=delta)
        if c["vg_ground"]:
            q = (txt["tokens"] * txt["mask"].unsqueeze(-1)).sum(1) / txt["mask"].sum(1, keepdim=True).clamp_min(1)
            ell = self.ground_head(g1[:, 1:], q, feats, valid,
                                   foot if c["vg_ground_input"] == "footprint" else None)
            iv = (self._intervention or {}).get("ell")
            if iv is not None:
                ell = iv.to(ell)
            tokens, z_les, evidence = self.lesion_pool(g1, ell)
            core.update(ell=ell, m_hat=torch.sigmoid(ell), z_les=z_les, evidence=evidence, box_map=foot)
        fus = self.fusion(tokens, txt["tokens"], None, txt["mask"], None, context_group_ids=nb.get("component_ids"))
        core["vis"]["f_i"] = tokens
        core["fus"] = fus
        core["logits_closed"] = self.closed_head(fus["x_f"])
        core["type_logit"] = self.type_head(fus["x_f"])
        logit_local = self.fracture_head(fus["x_f"])
        p_drop = float(c.get("vg_xf_droppath", 0.0))
        if p_drop > 0 and self.training:
            assert c["vg_ground"] and c["vg_pool_read"] == "feature", "x_f DropPath needs the h_les(z_les) route"
            b_out = self.fracture_head.net[-1].bias
            keep = (torch.rand(logit_local.shape[0], device=logit_local.device) >= p_drop).to(logit_local.dtype)
            logit_local = b_out + (logit_local - b_out) * keep / (1.0 - p_drop)
            core["xf_keep"] = keep
        if c["vg_ground"]:
            if c["vg_pool_read"] == "scalar":
                logit_local = logit_local + self.w_map * core["evidence"][:, 0]
            else:
                logit_local = logit_local + self.les_head(core["z_les"]).squeeze(-1)
        core["fracture_logit"] = logit_local
        core["logit_local"] = logit_local
        core["p_local"] = torch.sigmoid(logit_local.float())
        if c["vg_recovery"]:
            logit_g = self.global_fracture_head(core["x_global"])
            p_global = torch.sigmoid(logit_g.float())
            unc = core["p_local"] * (1 - core["p_local"]) * 4.0
            m_max = core["m_hat"].flatten(1).max(1).values if c["vg_ground"] else torch.zeros_like(unc)
            gate_in = torch.stack([scalars[:, 0], scalars[:, 1], unc.detach(), m_max, scalars[:, 2]], dim=1)
            alpha = self.recovery_gate(G.mean(1), gate_in)
            has_local = valid.any(1)
            alpha = torch.where(has_local, alpha, torch.zeros_like(alpha))  # m(I)=0 & no grid -> pure global
            p_final = (1 - alpha) * p_global.detach() + alpha * core["p_local"]
            core.update(logit_global=logit_g, p_global=p_global, alpha=alpha, p_final=p_final)
        else:
            core["p_final"] = core["p_local"]
        core["fracture_prob"] = core["p_final"]
        return core

    # ------------------------------------------------------------------ losses
    def _ground_loss(self, core, nb, rows):
        c = self.cfg
        targets = nb.get("gt_masks")
        if not rows or targets is None:
            return core["ell"].sum() * 0
        res = c["vg_loss_res"]
        idx = torch.tensor(rows, device=self.device)
        logits = F.interpolate(core["ell"][idx], size=(res, res), mode="bilinear", align_corners=False).flatten(1)
        tgt = torch.stack([torch.as_tensor(targets[i], dtype=torch.float32) for i in rows]).to(self.device).flatten(1)
        bce = F.binary_cross_entropy_with_logits(logits, tgt)
        pos = tgt.sum(1) > 0
        dice = logits.sum() * 0
        if pos.any():
            p = torch.sigmoid(logits[pos]); t = tgt[pos]
            dice = (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
        return c["vg_lambda_dice"] * dice + c["vg_lambda_bce"] * bce, dice.detach(), bce.detach()

    def _contrast_loss(self, nb, rows):
        c = self.cfg
        groups = [nb["contrast_crops"][i] for i in rows if nb["contrast_crops"][i]]
        groups = [g for g in groups if g.get("negatives")]
        anchor = self.local_tokens.question.weight.sum() * 0
        if not groups:
            return anchor
        crops, spans = [], []
        for g in groups:
            spans.append((len(crops), len(g["negatives"]) + 1))
            crops.append(g["positive"]); crops.extend(g["negatives"])
        z = self.visual(self.visual.preprocess(crops, self.device))["tokens"][:, 1:].mean(1)
        z = F.normalize(self.contrast_img(z), dim=-1)
        qt = self.text.encode_frozen([CANONICAL_Q], self.device)
        qv = (qt["tokens"] * qt["mask"].unsqueeze(-1)).sum(1) / qt["mask"].sum(1, keepdim=True)
        qf = F.normalize(self.contrast_query(self.local_tokens.question(qv)), dim=-1)[0]
        losses = []
        for start, n in spans:
            sims = (z[start:start + n] @ qf) / c["vg_tau"]
            losses.append(F.cross_entropy(sims.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=self.device)))
        return torch.stack(losses).mean() + anchor

    def forward(self, batch, stage: int = 3):
        """Training forward.  Re-uses BoneVQAModel.forward for L_VQA (closed CE,
        type, LM, contrastive image-text) via a one-shot core override, then adds
        L_ground, L_contrast and the recovery dual loss."""
        self.check_batch(batch)
        c = self.cfg
        nb = dict(batch)
        core_holder = {}

        original = self._encode_core

        def patched(*args, **kwargs):
            core = self._encode(nb)
            core_holder["core"] = core
            return core

        self._encode_core = patched
        try:
            out = BoneVQAModel.forward(self, {k: nb[k] for k in ("images", "questions", "answers", "answer_types",
                                                                 "image_ids", "component_ids") if nb.get(k) is not None}, stage)
        finally:
            self._encode_core = original
        core = core_holder["core"]
        dev = self.device
        canonical = [i for i, q in enumerate(nb["questions"]) if q == CANONICAL_Q
                     and normalize_answer(nb["answers"][i]) in ("yes", "no")]
        loss = out["loss"]
        terms = {"task": out["loss"]}  # non-detached per-loss terms (gradient diagnostics only)
        zero = torch.zeros((), device=dev)
        out.update(loss_ground=zero, loss_dice=zero, loss_gbce=zero, loss_contrast=zero, loss_recovery=zero)
        if c["vg_ground"] and self.training:
            seen, rows = set(), []
            for i in canonical:
                if nb["image_ids"][i] not in seen:
                    seen.add(nb["image_ids"][i]); rows.append(i)
            lg = self._ground_loss(core, nb, rows)
            if isinstance(lg, tuple):
                lg, ld, lb = lg
                out.update(loss_dice=ld, loss_gbce=lb)
            loss = loss + c["vg_lambda_ground"] * lg
            terms["ground"] = lg
            out["loss_ground"] = lg.detach()
        if c["vg_contrast"] and self.training and nb.get("contrast_crops"):
            pos = [i for i in canonical if normalize_answer(nb["answers"][i]) == "yes"]
            lc = self._contrast_loss(nb, pos)
            loss = loss + c["vg_lambda_contrast"] * lc
            terms["contrast"] = lc
            out["loss_contrast"] = lc.detach()
        if c["vg_recovery"] and canonical:
            idx = torch.tensor(canonical, device=dev)
            y = torch.tensor([normalize_answer(nb["answers"][i]) == "yes" for i in canonical], dtype=torch.float32, device=dev)
            pw = float(c.get("fracture_positive_weight", 1.0))
            wts = torch.where(y > 0, torch.full_like(y, pw), torch.ones_like(y))
            pf = core["p_final"][idx].clamp(1e-6, 1 - 1e-6)
            l_final = F.binary_cross_entropy(pf, y, weight=wts)
            l_global = F.binary_cross_entropy_with_logits(core["logit_global"][idx].float(), y, pos_weight=torch.tensor(pw, device=dev))
            loss = loss + l_final + c["vg_global_loss_weight"] * l_global
            terms["recovery"] = l_final + c["vg_global_loss_weight"] * l_global
            out["loss_recovery"] = (l_final + l_global).detach()
        # BoneVQAModel already adds BCE(FractureHead) == BCE(logit_local) on canonical rows
        # (weight fracture_loss_weight); under R this is the auxiliary local loss.
        out["loss"] = loss
        out["terms"] = terms
        out["core"] = core
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def infer(self, batch) -> Dict[str, torch.Tensor]:
        """Fracture endpoint for canonical rows. eval() only; GT keys forbidden."""
        assert not self.training, "infer() is evaluation-only; call model.eval()"
        self.check_batch(batch)
        core = self._encode(dict(batch))
        keys = ("p_final", "p_local", "p_global", "alpha", "m_hat", "evidence", "active", "box_map", "valid")
        return {k: core[k].detach().float().cpu() if torch.is_tensor(core.get(k)) else None for k in keys} | {
            "logits_closed": core["logits_closed"].detach().float().cpu(), "core": core}

    def predict_batch(self, batch, **_) -> List[Dict[str, Any]]:
        self.eval()
        self.check_batch(batch)
        with torch.no_grad():
            r = self.infer(batch)
        out = []
        probs = torch.softmax(r["logits_closed"], dim=-1)
        thr = float(self.cfg.get("fracture_yes_threshold", 0.5))
        for i, q in enumerate(batch["questions"]):
            row = {}
            if q == CANONICAL_Q:
                p = float(r["p_final"][i])
                row.update(answer="yes" if p >= thr else "no", p_yes=p, answer_type="CLOSED")
            else:
                j = int(probs[i].argmax())
                row.update(answer=self.answer_vocab[j], answer_type="CLOSED", confidence=float(probs[i, j]))
            out.append(row)
        return out

    def save(self, path: str, extra: Optional[dict] = None):
        guard_write(path)
        return super().save(path, extra)
