from typing import Optional

import torch
import torch.nn as nn

from .layers import CrossAttnBlock, SelfAttnBlock, invert_mask, masked_mean
from ..context_controls import component_disjoint_donors


class MultimodalFusion(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1,
                 alpha: float = 1.0, theta: float = 0.1, beta: float = 0.1,
                 use_latent: bool = True):
        super().__init__()
        self.alpha, self.theta, self.beta = alpha, theta, beta
        self.img_sa = SelfAttnBlock(d_model, n_heads, dropout)
        self.txt_sa = SelfAttnBlock(d_model, n_heads, dropout)
        self.img_ca = CrossAttnBlock(d_model, n_heads, dropout)
        self.txt_ca = CrossAttnBlock(d_model, n_heads, dropout)
        self.proj_i = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.proj_l = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.lp_ca_l = CrossAttnBlock(d_model, n_heads, dropout) if use_latent else None
        self.lp_ca_i = CrossAttnBlock(d_model, n_heads, dropout) if use_latent else None
        self.lp_ca_mm = CrossAttnBlock(d_model, n_heads, dropout) if use_latent else None
        self.lp_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh()) if use_latent else None
        self.gamma_latent = nn.Parameter(torch.zeros(())) if use_latent else None
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, e_i: torch.Tensor, e_l: torch.Tensor, x_lp: Optional[torch.Tensor],
                l_mask: Optional[torch.Tensor] = None, prior: Optional[torch.Tensor] = None,
                latent_context_control: str = "conditioned", context_group_ids=None):
        if latent_context_control not in ("conditioned", "unconditioned", "shuffled"):
            raise ValueError("Unknown latent context control: " + latent_context_control)
        l_pad = invert_mask(l_mask)
        f_i = self.img_sa(e_i)
        f_l = self.txt_sa(e_l, key_padding_mask=l_pad)
        f_fi = self.proj_i(self.img_ca(f_i, f_l, l_pad))
        f_fl = self.proj_l(self.txt_ca(f_l, f_i))
        f_mm = torch.cat([f_fi, f_fl], dim=1)
        mm_pad = None
        if l_pad is not None:
            mm_pad = torch.cat([torch.zeros(f_fi.shape[:2], dtype=torch.bool, device=f_fi.device), l_pad], dim=1)
        # Baseline is never replaced by P.  P supplies a gated residual to the
        # same representation consumed by both the closed classifier and LLM.
        pooled_base = masked_mean(f_mm, invert_mask(mm_pad))
        x_ii = f_mm
        latent_delta = torch.zeros_like(pooled_base)
        if x_lp is not None:
            # Intervene ONLY on LF context: the baseline answer representation,
            # learned parameters and attention tensor shapes remain unchanged.
            lf_l, lf_i, lf_mm, lf_pad, lf_mm_pad = f_l, f_i, f_mm, l_pad, mm_pad
            if latent_context_control == "unconditioned":
                lf_l, lf_i, lf_mm = torch.zeros_like(f_l), torch.zeros_like(f_i), torch.zeros_like(f_mm)
                lf_pad = None if l_pad is None else torch.zeros_like(l_pad)
                lf_mm_pad = None if mm_pad is None else torch.zeros_like(mm_pad)
            elif latent_context_control == "shuffled":
                if f_i.shape[0] < 2:
                    raise ValueError("Shuffled LF context requires at least two samples")
                # Legacy no-ID execution remains diagnostic-only. Scientific
                # training stays blocked at the training entry point until a
                # common component-aware batch schedule is locked for ALL arms.
                if context_group_ids is None:
                    donors = torch.arange(f_i.shape[0], device=f_i.device).roll(1)
                else:
                    if len(context_group_ids) != f_i.shape[0]:
                        raise ValueError("LF donor group count must match batch size")
                    donors = torch.tensor(component_disjoint_donors(context_group_ids), device=f_i.device)
                lf_l, lf_i, lf_mm = (value.index_select(0, donors) for value in (f_l, f_i, f_mm))
                lf_pad = None if l_pad is None else l_pad.index_select(0, donors)
                lf_mm_pad = None if mm_pad is None else mm_pad.index_select(0, donors)
            x = self.lp_ca_l(x_lp, lf_l, lf_pad)
            x = self.lp_ca_i(x, lf_i)
            x = self.lp_ca_mm(x, lf_mm, lf_mm_pad)
            if prior is not None:
                x = torch.cat([x, prior], dim=1)
            x_ii = torch.cat([f_mm, x], dim=1)
            latent_delta = self.gamma_latent * self.lp_out(x.mean(dim=1))
        x_f = self.alpha * (pooled_base + latent_delta) + self.theta * f_fi[:, 0] + self.beta * f_fl[:, 0]
        return {"x_f": self.out_norm(x_f), "x_ii": x_ii, "f_fi": f_fi, "f_fl": f_fl,
                "f_i": f_i, "f_l": f_l, "latent_delta": latent_delta}
