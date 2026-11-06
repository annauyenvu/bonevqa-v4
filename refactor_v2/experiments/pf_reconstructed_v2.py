"""Reconstructed PF variant: masked-neighbor GAT and graph-query residual.

Explicit algorithmic correction for graph-signal attenuation in this port.
Not the unchanged author PF. No fitted residual scale or additional parameters.
"""
from torch import nn
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.pf_masked_variant_v1 import configure_gat


class GraphResidualAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.cross_attn = source.cross_attn
        self.layer_norm = source.layer_norm

    def forward(self, query, key, value):
        return self.layer_norm(query + self.cross_attn(query, key, value))


class ReconstructedPFRunner(PFRunner):
    def __init__(self, mode='real'):
        super().__init__(mode)
        configure_gat(self.slake_pf.pf.GAN_layer, 'masked_neighbors')
        self.slake_pf.pf.cross_attn_kw = GraphResidualAttention(self.slake_pf.pf.cross_attn_kw)
