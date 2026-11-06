"""Four explicit mechanisms with identical parameter structure and initialization."""
from experiments.pf_slake_integrated_v1 import PFRunner
from experiments.pf_masked_variant_v1 import configure_gat
from experiments.pf_reconstructed_v2 import GraphResidualAttention

ARMS=('author_real','masked_real','residual_real','residual_rewired')


def build(arm):
    if arm not in ARMS: raise ValueError(arm)
    model=PFRunner('rewired' if arm=='residual_rewired' else 'real')
    if arm!='author_real':configure_gat(model.slake_pf.pf.GAN_layer,'masked_neighbors')
    if arm.startswith('residual_'):
        model.slake_pf.pf.cross_attn_kw=GraphResidualAttention(model.slake_pf.pf.cross_attn_kw)
    return model
