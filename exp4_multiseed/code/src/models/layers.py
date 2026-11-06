import torch
import torch.nn as nn


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * ffn_mult, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * ffn_mult, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_padding_mask=None) -> torch.Tensor:
        kvn = self.norm_kv(kv)
        a, _ = self.attn(self.norm_q(q), kvn, kvn, key_padding_mask=kv_padding_mask, need_weights=False)
        q = q + self.drop(a)
        q = q + self.drop(self.ffn(self.norm2(q)))
        return q


def masked_mean(x: torch.Tensor, mask=None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def invert_mask(mask):
    return None if mask is None else ~mask
