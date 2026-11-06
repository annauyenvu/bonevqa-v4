import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClosedHead(nn.Module):
    def __init__(self, d_model: int, n_answers: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_answers))
        nn.init.constant_(self.net[-1].bias, -math.log(max(n_answers - 1, 1)))

    def forward(self, x_f: torch.Tensor) -> torch.Tensor:
        return self.net(x_f)


class AnswerTypeHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ContrastiveProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)
