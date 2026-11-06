"""Endpoint-aligned residual routing for V/L prompt modules."""

from typing import Optional

import torch
import torch.nn as nn


class ResidualPromptMixer(nn.Module):
    """Add prompt evidence to the same visual stream used by every endpoint.

    Inputs are *semantic deltas*, never raw duplicate global features.  Gates
    start at exactly zero.  Therefore enabling a module does not replace the
    baseline representation at initialization, while the gate itself still
    receives a gradient on the first optimization step.
    """

    def __init__(self, d_model: int, use_visual: bool = True, use_lens: bool = True,
                 zero_preserving: bool = False):
        super().__init__()
        # Bias-free projection is required when the input represents a semantic
        # delta: zero evidence must remain *exactly* zero even after a learned
        # gate opens.  Historical v2 checkpoints retain the biased layout;
        # sealed v3 configs opt into ``zero_preserving``.
        use_bias = not zero_preserving
        self.visual_proj = (nn.Sequential(nn.Linear(d_model, d_model, bias=use_bias), nn.Tanh())
                            if use_visual else None)
        self.lens_proj = (nn.Sequential(nn.Linear(d_model, d_model, bias=use_bias), nn.Tanh())
                          if use_lens else None)
        self.gamma_visual = nn.Parameter(torch.zeros(())) if use_visual else None
        self.gamma_lens = nn.Parameter(torch.zeros(())) if use_lens else None

    @staticmethod
    def _pool(tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(
        self,
        base_tokens: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
        lens_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta = base_tokens.new_zeros(base_tokens.shape[0], base_tokens.shape[-1])
        if visual_tokens is not None and self.visual_proj is not None:
            delta = delta + self.gamma_visual * self.visual_proj(self._pool(visual_tokens))
        if lens_tokens is not None and self.lens_proj is not None:
            delta = delta + self.gamma_lens * self.lens_proj(self._pool(lens_tokens))
        return base_tokens + delta.unsqueeze(1)

    def contributions(
        self,
        base_tokens: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
        lens_tokens: Optional[torch.Tensor] = None,
    ):
        """Return endpoint-space contributions for QC and causal audits."""

        zeros = base_tokens.new_zeros(base_tokens.shape[0], base_tokens.shape[-1])
        visual = zeros
        lens = zeros
        if visual_tokens is not None and self.visual_proj is not None:
            visual = self.gamma_visual * self.visual_proj(self._pool(visual_tokens))
        if lens_tokens is not None and self.lens_proj is not None:
            lens = self.gamma_lens * self.lens_proj(self._pool(lens_tokens))
        return {"visual": visual, "lens": lens}

    def gate_values(self):
        return {
            "visual": None if self.gamma_visual is None else float(self.gamma_visual.detach()),
            "lens": None if self.gamma_lens is None else float(self.gamma_lens.detach()),
        }
