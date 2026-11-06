import torch
import torch.nn.functional as F


def lens_alignment_loss(img, txt, groups, mode="infonce"):
    """Repeated QA from one image are not contrastive negatives.

    With only one distinct image, contrastive losses collapse to a constant
    even when there are multiple QA rows. Use the declared single-image
    cosine fallback for that entire case, not only for a singleton row.
    """
    if len({str(group) for group in groups}) < 2:
        return (1.0 - F.cosine_similarity(img.float(), txt.float(), dim=-1)).mean()
    if mode == "dcl":
        return multi_positive_decoupled_loss(img, txt, groups)
    return multi_positive_info_nce_loss(img, txt, groups)


def endpoint_masks(answer_types, target_index: torch.Tensor):
    is_closed = torch.tensor([str(t).upper() == "CLOSED" for t in answer_types], device=target_index.device)
    return {
        "closed": is_closed & (target_index >= 0),
        "open": ~is_closed,
        "is_closed": is_closed,
    }


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


def multi_positive_decoupled_loss(img: torch.Tensor, txt: torch.Tensor, groups, temperature: float = 0.1):
    """Decoupled contrastive loss that excludes every same-group positive."""
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    logits = img @ txt.t() / temperature
    n = logits.shape[0]
    if n < 2:
        return logits.new_zeros(())
    positive = torch.tensor(
        [[str(groups[i]) == str(groups[j]) for j in range(n)] for i in range(n)],
        dtype=torch.bool,
        device=logits.device,
    )

    def direction(scores, pos_mask):
        has_negative = (~pos_mask).any(dim=1)
        if not has_negative.any():
            return scores.new_zeros(())
        pos = torch.logsumexp(scores.masked_fill(~pos_mask, float("-inf")), dim=1)
        pos = pos - pos_mask.sum(dim=1).float().log()
        neg = torch.logsumexp(scores.masked_fill(pos_mask, float("-inf")), dim=1)
        return (neg[has_negative] - pos[has_negative]).mean()

    return 0.5 * (direction(logits, positive) + direction(logits.t(), positive.t()))


def info_nce_loss(img: torch.Tensor, txt: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    logits = img @ txt.t() / temperature
    n = logits.shape[0]
    if n < 2:
        return logits.new_zeros(())
    target = torch.arange(n, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def multi_positive_info_nce_loss(img: torch.Tensor, txt: torch.Tensor, groups, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE where repeated QA rows from one image are positives, not negatives."""
    img = F.normalize(img.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    logits = img @ txt.t() / temperature
    n = logits.shape[0]
    if n < 2:
        return logits.new_zeros(())
    positive = torch.tensor(
        [[str(groups[i]) == str(groups[j]) for j in range(n)] for i in range(n)],
        dtype=torch.bool,
        device=logits.device,
    )

    def direction(scores, mask):
        log_den = torch.logsumexp(scores, dim=1)
        log_num = torch.logsumexp(scores.masked_fill(~mask, float("-inf")), dim=1)
        return (log_den - log_num).mean()

    return 0.5 * (direction(logits, positive) + direction(logits.t(), positive.t()))


def bce_multi_label(logits: torch.Tensor, target_index: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.sum() == 0:
        return logits.new_zeros(())
    logits = logits[valid].float()
    target = torch.zeros_like(logits)
    target[torch.arange(logits.shape[0], device=logits.device), target_index[valid]] = 1.0
    return F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / logits.shape[0]


def single_label_cross_entropy(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy for the mutually-exclusive closed-answer vocabulary."""
    if valid.sum() == 0:
        return logits.new_zeros(())
    per_sample = F.cross_entropy(logits[valid].float(), target_index[valid], reduction="none")
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight[valid].float()
    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)
