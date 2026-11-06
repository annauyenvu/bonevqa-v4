"""Paper-faithful building blocks for BoneVQA-Prompt v5.

Each class names the paper equation it implements so reviewers can audit the
mapping.  All modules are built inside isolated RNG scopes by ``model_v5`` so
switching one module on never changes the initialisation of shared tensors.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import CrossAttnBlock, SelfAttnBlock, masked_mean


# --------------------------------------------------------------------------- #
# FAVP / v4 local tokens.  Identical layout and init to experiments/vnext_model_v1
# so v5 with only the local route enabled reproduces v4 bit-for-bit.
# --------------------------------------------------------------------------- #
class LocalTokens(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, heads, dropout=0, batch_first=True)
        self.question = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4)
        # Kept for state-layout parity with v4 checkpoints (auxiliary weight 0).
        self.mil_score = nn.Linear(d, 1)
        self.aux_head = nn.Linear(d, 1)

    def forward(self, global_tokens, text, text_mask, crops, valid):
        b, k, t, d = crops.shape
        active = valid.any(1)
        memory = crops.reshape(b, k * t, d)
        padding = ~valid.repeat_interleave(t, dim=1)
        safe = padding.clone()
        safe[~active, 0] = False
        q = (text * text_mask.unsqueeze(-1)).sum(1) / text_mask.sum(1, keepdim=True).clamp_min(1)
        delta = self.attention(global_tokens + self.question(q).unsqueeze(1), self.norm(memory),
                               self.norm(memory), key_padding_mask=safe, need_weights=False)[0]
        delta = delta * torch.sigmoid(self.gate(global_tokens.mean(1))).unsqueeze(1) * active[:, None, None]
        crop_vectors = crops.mean(2)
        scores = self.mil_score(crop_vectors).squeeze(-1).masked_fill(~valid, -1e4)
        weights = scores.softmax(1) * active[:, None]
        local = (crop_vectors * weights.unsqueeze(-1)).sum(1)
        aux = self.aux_head(local).squeeze(-1)
        return global_tokens + delta, aux, active, delta


# --------------------------------------------------------------------------- #
# LaPA: latent prompt fusion (Sec. 3.2, Eq. 7-9) and prior knowledge fusion
# (Sec. 3.3, Eq. 10-11).
# --------------------------------------------------------------------------- #
class LapaLatentFusion(nn.Module):
    """Eq. 7-9 in the order the paper found best (Table 3): L -> I -> MM."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.ca_l = CrossAttnBlock(d, heads, dropout)
        self.ca_i = CrossAttnBlock(d, heads, dropout)
        self.ca_mm = CrossAttnBlock(d, heads, dropout)

    def forward(self, x_lp, f_l, l_pad, f_i, f_mm, mm_pad):
        x = self.ca_l(x_lp, f_l, l_pad)
        x = self.ca_i(x, f_i)
        return self.ca_mm(x, f_mm, mm_pad)


class GATLayer(nn.Module):
    """Multi-head graph attention restricted to graph neighbours (+ self loop).

    Masked-neighbour attention is the textbook GAT; the audit of the LaPA
    author route showed non-edges could exchange messages, so this variant is
    named explicitly in reports.
    """

    def __init__(self, d: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.w = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dh))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        h = self.w(x).view(n, self.h, self.dh)
        e = (h * self.a_src).sum(-1).unsqueeze(1) + (h * self.a_dst).sum(-1).unsqueeze(0)  # [n, n, H]
        e = F.leaky_relu(e, 0.2).masked_fill(~adj.unsqueeze(-1), float("-inf"))
        att = self.drop(torch.softmax(e, dim=1))
        out = torch.einsum("ijh,jhd->ihd", att, h).reshape(n, d)
        return self.norm(x + F.elu(out))


class BoneKnowledgePrior(nn.Module):
    """LaPA PF: F_G = GNN(F_OD, X_adj);  X̂_II = [X̃_II ; Proj(CA(F_G, X_LP, X_LP))].

    Node features F_OD are frozen biomedical text embeddings of node names
    (LaPA embeds organ/disease names with RoBERTa; here PubMedBERT of
    BiomedCLIP, shared with the rest of the model).  ``compress`` maps the
    node axis to ``n_out`` tokens (the "Compress" block in LaPA Fig. 2c).
    """

    def __init__(self, d: int, n_nodes: int, heads: int, dropout: float, n_layers: int = 2, n_out: int = 4):
        super().__init__()
        self.inp = nn.Linear(d, d)
        self.gnn = nn.ModuleList([GATLayer(d, heads, dropout) for _ in range(n_layers)])
        self.ca = CrossAttnBlock(d, heads, dropout)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.compress = nn.Linear(n_nodes, n_out)
        self.register_buffer("node_feat", torch.zeros(n_nodes, d), persistent=False)
        self.register_buffer("adj", torch.eye(n_nodes, dtype=torch.bool), persistent=False)
        self.ready = False

    def set_graph(self, node_feat: torch.Tensor, adj: torch.Tensor):
        self.node_feat = node_feat.detach().float().to(self.inp.weight.device)
        self.adj = adj.to(self.inp.weight.device)
        self.ready = True

    def forward(self, x_lp: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            raise RuntimeError("Bone KG prior used before set_graph(); refusing silent zero graph")
        g = self.inp(self.node_feat.to(x_lp.dtype))
        for layer in self.gnn:
            g = layer(g, self.adj)
        g = g.unsqueeze(0).expand(x_lp.shape[0], -1, -1)
        prior = self.proj(self.ca(g, x_lp))                    # [B, n_nodes, d]
        return self.compress(prior.transpose(1, 2)).transpose(1, 2)  # [B, n_out, d]


# --------------------------------------------------------------------------- #
# Localization Lens: view fusion with pixel shuffle (Eq. 4-5).
# --------------------------------------------------------------------------- #
class LensFusionV5(nn.Module):
    """E_fused = v(E_orig, E_rep1, E_rep2); then pixel (un)shuffle by factor r.

    The paper uses r = 4.  A 14x14 ViT grid is not divisible by 4, so the
    patch grid is zero-padded to the next multiple (16x16 -> 4x4 tokens).
    The padding is recorded; r = 2 needs no padding (7x7 tokens).
    """

    def __init__(self, d: int, n_views: int = 3, n_layers: int = 2, heads: int = 8,
                 dropout: float = 0.0, grid: int = 14, shuffle_r: int = 4):
        super().__init__()
        self.grid, self.r = grid, shuffle_r
        self.padded = int(math.ceil(grid / shuffle_r) * shuffle_r)
        self.view_emb = nn.Parameter(torch.zeros(n_views, 1, d))
        nn.init.trunc_normal_(self.view_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.compress = nn.Linear(d * shuffle_r * shuffle_r, d)
        self.cls_proj = nn.Linear(d, d)

    @property
    def n_out_tokens(self) -> int:
        return 1 + (self.padded // self.r) ** 2

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        b, v, n, d = view_tokens.shape
        x = view_tokens + self.view_emb[:v].unsqueeze(0).to(view_tokens.dtype)
        x = self.encoder(x.reshape(b, v * n, d))
        x = self.norm(x).reshape(b, v, n, d).mean(dim=1)
        cls_tok = self.cls_proj(x[:, 0:1])
        p = x[:, 1:1 + self.grid * self.grid].transpose(1, 2).reshape(b, d, self.grid, self.grid)
        if self.padded != self.grid:
            p = F.pad(p, (0, self.padded - self.grid, 0, self.padded - self.grid))
        p = F.pixel_unshuffle(p, self.r).flatten(2).transpose(1, 2)
        return torch.cat([cls_tok, self.compress(p)], dim=1)


def pool_tokens(tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_mean(tokens, mask)
