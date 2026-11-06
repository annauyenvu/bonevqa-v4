import torch
import torch.nn.functional as F


def decoupled_contrastive_loss(img: torch.Tensor, txt: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    logits = img @ txt.t() / temperature
    n = logits.shape[0]
    if n < 2:
        return logits.new_zeros(())
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    pos = logits.diagonal()
    neg_i2t = logits.masked_fill(eye, float("-inf")).logsumexp(dim=1)
    neg_t2i = logits.t().masked_fill(eye, float("-inf")).logsumexp(dim=1)
    return ((neg_i2t - pos) + (neg_t2i - pos)).mean() * 0.5


def info_nce_loss(img: torch.Tensor, txt: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    logits = img @ txt.t() / temperature
    n = logits.shape[0]
    if n < 2:
        return logits.new_zeros(())
    target = torch.arange(n, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def bce_multi_label(logits: torch.Tensor, target_index: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.sum() == 0:
        return logits.new_zeros(())
    logits = logits[valid].float()
    target = torch.zeros_like(logits)
    target[torch.arange(logits.shape[0], device=logits.device), target_index[valid]] = 1.0
    return F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / logits.shape[0]
