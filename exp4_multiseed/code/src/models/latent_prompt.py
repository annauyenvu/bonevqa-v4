from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CrossAttnBlock, SelfAttnBlock, masked_mean


class LatentPromptGenerator(nn.Module):
    def __init__(self, d_model: int = 768, n_latent: int = 32, n_heads: int = 8, dropout: float = 0.1,
                 prior_labels: Optional[List[str]] = None):
        super().__init__()
        self.n_latent = n_latent
        self.latent = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
        self.answer_sa = SelfAttnBlock(d_model, n_heads, dropout)
        self.answer_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.cross = CrossAttnBlock(d_model, n_heads, dropout)
        self.prior_labels = list(prior_labels or [])
        if self.prior_labels:
            self.prior_emb = nn.Embedding(len(self.prior_labels), d_model)
            self.prior_cross = CrossAttnBlock(d_model, n_heads, dropout)
        self.register_buffer("answer_bank", torch.zeros(0, d_model), persistent=False)

    def set_answer_bank(self, pooled_answers: torch.Tensor):
        self.answer_bank = pooled_answers.detach().float()

    def forward(self, batch_size: int):
        x = self.latent.unsqueeze(0).expand(batch_size, -1, -1)
        if self.answer_bank.numel() == 0:
            return x, None
        bank = self.answer_bank.to(x.device, x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        f_ta = self.answer_proj(self.answer_sa(bank))
        x_hat = self.cross(x, f_ta)
        prior = None
        if self.prior_labels:
            f_g = self.prior_emb.weight.to(x.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            prior = self.prior_cross(x_hat, f_g)
        return x_hat, prior

    @staticmethod
    def consistency_loss(x_hat: torch.Tensor, target_answer_tokens: torch.Tensor, target_mask=None) -> torch.Tensor:
        p = x_hat.float().mean(dim=1)
        t = masked_mean(target_answer_tokens.float(), target_mask)
        return (1.0 - F.cosine_similarity(p, t, dim=-1)).mean()
