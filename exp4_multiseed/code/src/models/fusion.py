from typing import Optional

import torch
import torch.nn as nn

from .layers import CrossAttnBlock, SelfAttnBlock, invert_mask, masked_mean


class MultimodalFusion(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 8, dropout: float = 0.1,
                 alpha: float = 1.0, theta: float = 0.1, beta: float = 0.1):
        super().__init__()
        self.alpha, self.theta, self.beta = alpha, theta, beta
        self.img_sa = SelfAttnBlock(d_model, n_heads, dropout)
        self.txt_sa = SelfAttnBlock(d_model, n_heads, dropout)
        self.img_ca = CrossAttnBlock(d_model, n_heads, dropout)
        self.txt_ca = CrossAttnBlock(d_model, n_heads, dropout)
        self.proj_i = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.proj_l = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.lp_ca_l = CrossAttnBlock(d_model, n_heads, dropout)
        self.lp_ca_i = CrossAttnBlock(d_model, n_heads, dropout)
        self.lp_ca_mm = CrossAttnBlock(d_model, n_heads, dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, e_i: torch.Tensor, e_l: torch.Tensor, x_lp: Optional[torch.Tensor],
                l_mask: Optional[torch.Tensor] = None, prior: Optional[torch.Tensor] = None):
        l_pad = invert_mask(l_mask)
        f_i = self.img_sa(e_i)
        f_l = self.txt_sa(e_l, key_padding_mask=l_pad)
        f_fi = self.proj_i(self.img_ca(f_i, f_l, l_pad))
        f_fl = self.proj_l(self.txt_ca(f_l, f_i))
        f_mm = torch.cat([f_fi, f_fl], dim=1)
        mm_pad = None
        if l_pad is not None:
            mm_pad = torch.cat([torch.zeros(f_fi.shape[:2], dtype=torch.bool, device=f_fi.device), l_pad], dim=1)
        if x_lp is None:
            x_ii = f_mm
            pooled_ii = masked_mean(f_mm, invert_mask(mm_pad))
        else:
            x = self.lp_ca_l(x_lp, f_l, l_pad)
            x = self.lp_ca_i(x, f_i)
            x = self.lp_ca_mm(x, f_mm, mm_pad)
            if prior is not None:
                x = torch.cat([x, prior], dim=1)
            x_ii = x
            pooled_ii = x_ii.mean(dim=1)
        x_f = self.alpha * pooled_ii + self.theta * f_fi[:, 0] + self.beta * f_fl[:, 0]
        return {"x_f": self.out_norm(x_f), "x_ii": x_ii, "f_fi": f_fi, "f_fl": f_fl, "f_i": f_i, "f_l": f_l}
