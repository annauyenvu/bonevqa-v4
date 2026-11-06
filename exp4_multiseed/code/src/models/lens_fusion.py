import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import CrossAttnBlock, SelfAttnBlock


class LensFusion(nn.Module):
    def __init__(self, d_model: int = 768, n_views: int = 3, n_layers: int = 2, n_heads: int = 8,
                 dropout: float = 0.1, grid: int = 14, shuffle_r: int = 2):
        super().__init__()
        self.n_views = n_views
        self.grid = grid
        self.shuffle_r = shuffle_r
        self.view_emb = nn.Parameter(torch.zeros(n_views, 1, d_model))
        nn.init.trunc_normal_(self.view_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.compress = nn.Linear(d_model * shuffle_r * shuffle_r, d_model)
        self.cls_proj = nn.Linear(d_model, d_model)

    @property
    def n_out_tokens(self) -> int:
        return 1 + (self.grid // self.shuffle_r) ** 2

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        b, v, n, d = view_tokens.shape
        x = view_tokens + self.view_emb[:v].unsqueeze(0).to(view_tokens.dtype)
        x = self.encoder(x.reshape(b, v * n, d))
        x = self.norm(x).reshape(b, v, n, d).mean(dim=1)
        cls_tok = self.cls_proj(x[:, 0:1])
        patches = x[:, 1:1 + self.grid * self.grid]
        patches = patches.transpose(1, 2).reshape(b, d, self.grid, self.grid)
        patches = F.pixel_unshuffle(patches, self.shuffle_r)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.compress(patches)
        return torch.cat([cls_tok, patches], dim=1)


class QFormerLite(nn.Module):
    def __init__(self, d_model: int = 768, n_query: int = 32, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_query, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([SelfAttnBlock(d_model, n_heads, dropout), CrossAttnBlock(d_model, n_heads, dropout)]))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_tokens: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        q = self.query.unsqueeze(0).expand(visual_tokens.shape[0], -1, -1).to(visual_tokens.dtype)
        for sa, ca in self.layers:
            q = sa(q)
            q = ca(q, visual_tokens, key_padding_mask)
        return self.norm(q)
