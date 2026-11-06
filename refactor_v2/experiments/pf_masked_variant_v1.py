"""Explicit algorithm variants, not unchanged author LaPA.

mask_only retains author's dim1 normalization to isolate masking.
masked_neighbors additionally normalizes over neighbor keys (last dimension).
Parameter names/shapes are unchanged, permitting exact matched initialization.
"""
import types
import torch
from torch.nn import functional as F


def masked_forward(self, h, adj):
    wh = h @ self.W
    logits = self.leakyrelu(wh @ self.a[:self.out_features] +
                           (wh @ self.a[self.out_features:]).transpose(1, 2))
    allowed = adj > 0
    if not allowed.any(-1).all() or not allowed.any(-2).all():
        raise ValueError('Every row/column needs an allowed edge (self loops)')
    weights = logits.masked_fill(~allowed, float('-inf')).softmax(self.normalization_dim)
    weights = F.dropout(weights, self.dropout, training=self.training)
    result = weights @ wh
    return F.elu(result) if self.concat else result


def configure_gat(gat, variant):
    if variant not in ('mask_only', 'masked_neighbors'):
        raise ValueError(variant)
    for layer in [*gat.attentions, gat.out_att]:
        layer.normalization_dim = 1 if variant == 'mask_only' else -1
        layer.forward = types.MethodType(masked_forward, layer)
    return gat
