"""BoneVQA-Prompt v5: B + paper-faithful FAVP / Localization Lens / LaPA routes.

Compatibility contract (enforced by tests/test_v5_model.py):
* all ``v5_*`` switches off  -> forward/predict identical to B (BoneVQAModel);
* only ``v5_local`` (crop, hag off) -> identical to v4 (experiments/vnext_model_v1);
* Localization Lens never touches inference: ``predict_batch`` is invariant to
  lens masks, lens modules and lens weights.

Deviations corrected relative to earlier ports (see reports/V5_FIDELITY_MAP.md):
* FAVP HAG: local + global ViT tokens are concatenated into the Q-Former that
  feeds the LLM (FAVP Eq. 5-7).  v4 only fed local tokens to the fusion.
* FAVP visual prompt kinds (crop / box / circle / contour / mask) selectable.
* Lens: anatomical views for every image, region-text DCL alignment phase,
  QA InfoNCE phase, pixel shuffle r = 4, original-image-only inference.
* LaPA: eta = 0.1, latent size 32, L -> I -> MM order, PF = GAT over a bone
  knowledge graph (not SLAKE), X_F = alpha*X_II + theta*F_FI + beta*F_FL with
  alpha = 1, theta = beta = 0.1 ("lapa" mode).  "residual" mode keeps the v3
  gated residual for the ablation.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bonevqa import BoneVQAModel, normalize_batch
from src.models.latent_prompt import LatentPromptGenerator
from src.models.layers import invert_mask, masked_mean
from src.models.losses import lens_alignment_loss

from .bone_kg_v5 import adjacency
from .modules_v5 import BoneKnowledgePrior, LapaLatentFusion, LensFusionV5, LocalTokens

# torch 2.6 MPS bug (found 2026-10-30): layer_norm backward returns wrong weight/bias gradients
# (zeros, garbage or inf) when the INPUT does not require grad -- e.g. fusion.txt_sa.norm1 on frozen
# text tokens, v5_gm.answer_sa.norm1 on the answer bank, local_tokens.norm on crop features.
# Giving such inputs a (discarded) grad makes MPS take the correct kernel path; forward values and
# all other gradients are unchanged, and CPU is untouched.  BONEVQA_MPS_LN_FIX=0 disables it
# (used only to show the bug in the MPS-vs-CPU gradient qualification).
_layer_norm = F.layer_norm


def _mps_safe_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    if (input.device.type == "mps" and torch.is_grad_enabled() and not input.requires_grad
            and any(p is not None and p.requires_grad for p in (weight, bias))):
        input = input.detach().requires_grad_(True)
    return _layer_norm(input, normalized_shape, weight, bias, eps)


if __import__("os").environ.get("BONEVQA_MPS_LN_FIX", "1") != "0":
    F.layer_norm = _mps_safe_layer_norm

V5_DEFAULTS: Dict = {
    # FAVP / local visual prompt
    "v5_local": False,
    "v5_local_view": "crop",          # crop | box | circle | contour | mask
    "v5_local_content": "image",      # image | slot_constant (fixed per-slot vectors, no pixels from the regions)
    "v5_local_max": 3,
    "v5_hag": "off",                  # off | replace | gated
    # Localization Lens (training only)
    "v5_lens": False,
    "v5_lens_shuffle_r": 4,
    "v5_lens_layers": 2,
    "v5_lens_weight": 0.1,
    "v5_lens_modes": ["unicolor", "multicolor", "masked"],
    "v5_lens_phase1_loss": "dcl",     # dcl (paper) | infonce (Lens Table 3 ablation)
    # LaPA latent prompt
    "v5_latent": False,
    "v5_latent_mode": "lapa",         # lapa | residual
    "v5_n_latent": 32,
    "v5_eta": 0.1,
    "v5_pf": "none",                  # none | graph | empty
    "v5_kg": None,                    # graph dict from bone_kg_v5 (real or rewired)
    "v5_pf_tokens": 4,
    "v5_alpha": 1.0,
    "v5_theta": 0.1,
    "v5_beta": 0.1,
    "v5_init_seed": 20261030,
}
LEGACY_SWITCHES = ("use_visual_prompt", "use_lens", "use_latent_prompt")  # T stays the base textual prompt


def _isolated(factory, seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return factory()


class BoneVQAv5(BoneVQAModel):
    def __init__(self, cfg: Optional[dict] = None, answer_vocab=None, device=None):
        merged = dict(V5_DEFAULTS)
        merged.update(cfg or {})
        super().__init__(merged, answer_vocab, device)
        c = self.cfg
        assert not any(c[k] for k in LEGACY_SWITCHES), "v5 replaces the legacy V/L/P switches"
        assert c["v5_local_view"] in ("crop", "box", "circle", "contour", "mask")
        assert c["v5_hag"] in ("off", "replace", "gated")
        assert c["v5_latent_mode"] in ("lapa", "residual")
        assert c["v5_pf"] in ("none", "graph", "empty")
        if c["v5_hag"] != "off" and not c["v5_local"]:
            raise ValueError("v5_hag requires v5_local")
        d, h, p = c["d_model"], c["n_heads"], c["dropout"]
        seed = int(c["v5_init_seed"])
        if c["v5_local"]:
            # Same seed and construction as v4 VNextModel -> identical init.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261025)
                self.local_tokens = LocalTokens(d, h)
        if c["v5_hag"] == "gated":
            self.gamma_hag = nn.Parameter(torch.zeros(()))
        if c["v5_lens"]:
            self.v5_lens_fusion = _isolated(lambda: LensFusionV5(
                d, 3, c["v5_lens_layers"], h, 0.0, grid=c["image_size"] // 16,
                shuffle_r=c["v5_lens_shuffle_r"]), seed + 1)
        if c["v5_latent"]:
            self.v5_gm = _isolated(lambda: LatentPromptGenerator(d, c["v5_n_latent"], h, p, prior_labels=[]), seed + 2)
            self.v5_lf = _isolated(lambda: LapaLatentFusion(d, h, p), seed + 3)
            if c["v5_pf"] != "none":
                if not c.get("v5_kg"):
                    raise ValueError("v5_pf requires v5_kg (bone graph from TRAIN)")
                n_nodes = len(c["v5_kg"]["nodes"])
                self.v5_prior = _isolated(lambda: BoneKnowledgePrior(d, n_nodes, h, p, n_out=c["v5_pf_tokens"]), seed + 4)
            if c["v5_latent_mode"] == "lapa":
                self.v5_out_norm = _isolated(lambda: nn.LayerNorm(d), seed + 5)
            else:
                self.v5_lp_out = _isolated(lambda: nn.Sequential(nn.Linear(d, d), nn.Tanh()), seed + 5)
                self.v5_gamma_latent = nn.Parameter(torch.zeros(()))
        self.v5_ready = False
        self._v5_cache = None

    # ------------------------------------------------------------------ setup
    @property
    def v5_any(self) -> bool:
        c = self.cfg
        return bool(c["v5_local"] or c["v5_latent"])

    @torch.no_grad()
    def refresh_answer_bank(self):
        super().refresh_answer_bank()
        self.refresh_v5()

    @torch.no_grad()
    def refresh_v5(self):
        c = self.cfg
        if c["v5_latent"]:
            pooled = []
            for i in range(0, len(self.answer_vocab), 128):
                out = self.text.encode_frozen(self.answer_vocab[i:i + 128], self.device)
                pooled.append(out["tokens"][:, 0].float())
            self.v5_gm.set_answer_bank(torch.cat(pooled, 0))
            if c["v5_pf"] != "none":
                kg = c["v5_kg"]
                feats = self.text.encode_frozen(list(kg["nodes"]), self.device)["tokens"][:, 0].float()
                adj = adjacency(kg) if c["v5_pf"] == "graph" else torch.eye(len(kg["nodes"]), dtype=torch.bool)
                self.v5_prior.set_graph(feats, adj.to(self.device))
        self.v5_ready = True

    def load_trainable(self, path, allowed_mismatch_prefixes=()):
        extra = super().load_trainable(path, allowed_mismatch_prefixes)
        self.v5_ready = False
        return extra

    def _prepare_masks(self, images, masks, boxes, image_ids):
        # Production path: detector boxes -> rectangle masks for the local route
        # (the legacy method only converts boxes when legacy V/L are on).
        if self.cfg["v5_local"] and (masks is None or all(m is None for m in masks)) and boxes is not None:
            out = []
            for im, b in zip(images, boxes):
                ms = []
                for box in (b or [])[: self.cfg["v5_local_max"]]:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
                    x1, x2 = max(0, min(im.width, x1)), max(0, min(im.width, x2))
                    y1, y2 = max(0, min(im.height, y1)), max(0, min(im.height, y2))
                    if x2 > x1 and y2 > y1:
                        m = np.zeros((im.height, im.width), dtype=np.uint8)
                        m[y1:y2, x1:x2] = 1
                        ms.append(m)
                out.append(np.stack(ms) if ms else np.zeros((0, im.height, im.width), dtype=np.uint8))
            return out
        return super()._prepare_masks(images, masks, boxes, image_ids)

    # ------------------------------------------------------------ local views
    def _local_views(self, images, mask_list):
        from src.data.prompts import build_visual_prompts

        kind = self.cfg["v5_local_view"]
        views, slots = [], []
        for image, masks in zip(images, mask_list):
            s = []
            for mask in list(masks)[: self.cfg["v5_local_max"]]:
                arr = np.asarray(mask)
                yy, xx = np.where(arr > 0)
                if not len(xx):
                    continue
                if kind == "crop":
                    x1, x2 = int(xx.min()), int(xx.max()) + 1
                    y1, y2 = int(yy.min()), int(yy.max()) + 1
                    view = image.crop((x1, y1, x2, y2))
                else:
                    view = build_visual_prompts(image, arr[None].astype(np.uint8), kind=kind)[0]
                s.append(len(views))
                views.append(view)
            slots.append(s)
        return views, slots

    # ------------------------------------------------------------- encoding
    def _encode_core(self, images, questions, regions, mask_list, prompt_kind, training,
                     include_lens_views: bool = False, context_group_ids=None):
        core = super()._encode_core(images, questions, regions, mask_list, prompt_kind, training,
                                    include_lens_views, context_group_ids)
        if not self.v5_any:
            return core
        c = self.cfg
        if not self.v5_ready:
            self.refresh_v5()
        vis, txt = core["vis"], core["txt"]
        global_tokens = vis["global_tokens"]
        b, _, d = global_tokens.shape
        fi, aux, active, delta = global_tokens, None, None, None
        fus = core["fus"]
        if c["v5_local"]:
            views, slots = self._local_views(images, mask_list)
            k = c["v5_local_max"]
            features = global_tokens.new_zeros(b, k, 4, d)
            valid = torch.zeros(b, k, dtype=torch.bool, device=global_tokens.device)
            raw = None
            if views and c["v5_local_content"] == "slot_constant":
                # Availability-only control: which slots exist reaches the model, region pixels do not.
                gen = torch.Generator().manual_seed(20261103)
                constant = torch.randn(k, 4, d, generator=gen).to(features)
                for i, s in enumerate(slots):
                    for j, _ in enumerate(s):
                        features[i, j] = constant[j]
                        valid[i, j] = True
            elif views:
                raw = self.visual(self.visual.preprocess(views, self.device))["tokens"]
                patches = raw[:, 1:]
                side = int(patches.shape[1] ** 0.5)
                assert side * side == patches.shape[1]
                grid = patches.transpose(1, 2).reshape(len(views), d, side, side)
                if grid.device.type == "mps":
                    # torch 2.6 MPS adaptive_avg_pool2d is wrong (|err| ~0.6) on this strided view;
                    # contiguous only on MPS so CPU stays bit-identical to v4.
                    grid = grid.contiguous()
                pooled = F.adaptive_avg_pool2d(grid, (2, 2)).flatten(2).transpose(1, 2)
                for i, s in enumerate(slots):
                    for j, slot in enumerate(s):
                        features[i, j] = pooled[slot]
                        valid[i, j] = True
            fi, aux, active, delta = self.local_tokens(global_tokens, txt["tokens"], txt["mask"], features, valid)
            fus = self.fusion(fi, txt["tokens"], None, txt["mask"], None, context_group_ids=context_group_ids)
            if c["v5_hag"] != "off":
                vis = dict(vis)
                vis["prefix_vis"] = self._hag_prefix(global_tokens, raw, slots, vis["prefix_vis"])
        if c["v5_latent"]:
            fus = self._lapa(fi, fus, txt)
        vis = dict(vis)
        vis["f_i"] = fi
        core["vis"], core["fus"] = vis, fus
        core["logits_closed"] = self.closed_head(fus["x_f"])
        core["type_logit"] = self.type_head(fus["x_f"])
        core["fracture_logit"] = self.fracture_head(fus["x_f"]) if self.fracture_head is not None else None
        self._v5_cache = {"aux": aux, "active": active, "delta": delta, "x_lp": fus.get("x_lp_v5"),
                          "global_tokens": global_tokens}
        return core

    def _hag_prefix(self, global_tokens, raw, slots, prefix_base):
        """FAVP Eq. 5-6: Q-Former over concat(local_1..local_N, global) tokens."""
        b, n, d = global_tokens.shape
        lengths = [n * (1 + len(s)) for s in slots]
        L = max(lengths)
        memory = global_tokens.new_zeros(b, L, d)
        pad = torch.ones(b, L, dtype=torch.bool, device=global_tokens.device)
        for i, s in enumerate(slots):
            parts = [global_tokens[i]] + ([raw[j] for j in s] if raw is not None else [])
            seq = torch.cat(parts, 0)
            memory[i, : seq.shape[0]] = seq
            pad[i, : seq.shape[0]] = False
        joint = self.qformer(memory, pad)
        if self.cfg["v5_hag"] == "replace":
            return joint
        return prefix_base + self.gamma_hag * (joint - prefix_base)

    def _lapa(self, e_i, fus, txt):
        c = self.cfg
        f_l, f_i, f_fi, f_fl = fus["f_l"], fus["f_i"], fus["f_fi"], fus["f_fl"]
        l_pad = invert_mask(txt["mask"])
        f_mm = torch.cat([f_fi, f_fl], 1)
        mm_pad = torch.cat([torch.zeros(f_fi.shape[:2], dtype=torch.bool, device=f_fi.device), l_pad], 1)
        x_lp, _ = self.v5_gm(e_i)                                    # GM, Eq. 1-2
        x = self.v5_lf(x_lp, f_l, l_pad, f_i, f_mm, mm_pad)          # LF, Eq. 7-9
        if c["v5_pf"] != "none":
            x = torch.cat([x, self.v5_prior(x_lp)], 1)               # PF, Eq. 10-11
        if c["v5_latent_mode"] == "lapa":                            # Eq. 12
            x_f = self.v5_out_norm(c["v5_alpha"] * x.mean(1) + c["v5_theta"] * f_fi[:, 0]
                                   + c["v5_beta"] * f_fl[:, 0])
        else:
            pooled = masked_mean(f_mm, invert_mask(mm_pad))
            fu = self.fusion
            x_f = fu.out_norm(fu.alpha * (pooled + self.v5_gamma_latent * self.v5_lp_out(x.mean(1)))
                              + fu.theta * f_fi[:, 0] + fu.beta * f_fl[:, 0])
        out = dict(fus)
        out["x_f"] = x_f
        out["x_ii"] = torch.cat([f_mm, x], 1)
        out["x_lp_v5"] = x_lp
        return out

    # -------------------------------------------------------------- training
    def forward(self, batch, stage: int = 2):
        output = super().forward(batch, stage)
        c = self.cfg
        nb = normalize_batch(batch)
        zero = output["loss"].new_zeros(())
        loss_cs, loss_lens = zero, zero
        if c["v5_latent"] and stage != 1 and self._v5_cache and self._v5_cache["x_lp"] is not None:
            target = self.text.encode_frozen(nb["answers"], self.device)
            loss_cs = LatentPromptGenerator.consistency_loss(self._v5_cache["x_lp"], target["tokens"], target["mask"])
        if c["v5_lens"] and self.training and nb.get("lens_masks") is not None:
            loss_lens = self.lens_loss(nb, phase="qa")
        output["loss"] = output["loss"] + c["v5_eta"] * loss_cs + c["v5_lens_weight"] * loss_lens
        output["loss_v5_consistency"] = loss_cs.detach()
        output["loss_v5_lens"] = loss_lens.detach()
        if self._v5_cache and self._v5_cache.get("delta") is not None:
            output["local_delta_norm"] = self._v5_cache["delta"].detach().norm(dim=-1).mean()
            output["local_active_rows"] = self._v5_cache["active"].sum().detach()
        return output

    def lens_loss(self, nb, phase: str = "qa") -> torch.Tensor:
        """Localization Lens alignment (paper Sec. 2.3).

        phase 'region': DCL between fused lens views and the body-region text
        (the organ-alignment pre-training phase).  phase 'qa': InfoNCE between
        fused lens views and question-answer text.  Lens views are built from
        ``lens_masks`` that the data pipeline supplies for TRAIN rows only.
        """
        from src.data.prompts import build_lens

        c = self.cfg
        first: Dict[str, int] = {}
        for i, iid in enumerate(nb["image_ids"]):
            if nb["lens_masks"][i] is not None and len(nb["lens_masks"][i]) > 0:
                first.setdefault(str(iid), i)
        rows = list(first.values())
        if not rows:
            return torch.zeros((), device=self.device)
        imgs = []
        for i in rows:
            im, ms = nb["images"][i], np.asarray(nb["lens_masks"][i])
            modes = random.sample(list(c["v5_lens_modes"]), 2)
            imgs += [im] + [build_lens(im, ms, mode=m) for m in modes]
        tok = self.visual(self.visual.preprocess(imgs, self.device))["tokens"]
        tok = tok.reshape(len(rows), 3, tok.shape[1], tok.shape[2])
        fused = self.v5_lens_fusion(tok)
        img_vec = self.img_proj(fused.mean(1))
        if phase == "region":
            texts = [str(nb["regions"][i] or "bone") for i in rows]
            txt_vec = self.txt_proj(self.text.encode_frozen(texts, self.device)["tokens"][:, 0])
            return lens_alignment_loss(img_vec, txt_vec, texts, c["v5_lens_phase1_loss"])
        if phase != "qa":
            raise ValueError(phase)
        pos = {str(nb["image_ids"][i]): k for k, i in enumerate(rows)}
        use = [i for i, iid in enumerate(nb["image_ids"]) if str(iid) in pos]
        texts = [f"{nb['questions'][i]} {nb['answers'][i]}".strip() for i in use]
        txt_vec = self.txt_proj(self.text.encode_frozen(texts, self.device)["tokens"][:, 0])
        img_rows = img_vec[[pos[str(nb["image_ids"][i])] for i in use]]
        groups = [str(nb["image_ids"][i]) for i in use]
        return lens_alignment_loss(img_rows, txt_vec, groups, "infonce")

    def forward_lens_align(self, batch) -> Dict[str, torch.Tensor]:
        """Phase-1 Localization Lens step: region-text DCL only."""
        nb = normalize_batch(batch)
        loss = self.lens_loss(nb, phase="region")
        return {"loss": loss, "loss_v5_lens_region": loss.detach()}
