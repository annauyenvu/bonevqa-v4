"""Author-layout classifier and objective for the disclosed shared-backbone port."""
import torch
from torch import nn
from torch.nn import functional as F
from experiments.lapa_author_route_fixture_v1 import author_primitives


class ReferenceAnswerHead(nn.Sequential):
    def __init__(self, width, classes):
        super().__init__(nn.Linear(width, width*2), nn.LayerNorm(width*2),
                         nn.GELU(), nn.Linear(width*2, classes))
        self.apply(author_primitives()['init_weights'])


def reference_loss(logits, targets, generated_mean, raw_target_words, eta=0.1):
    """Raw target words include the prospectively specified padding/special tokens.

    Training targets only. Never call this function inside public inference.
    Single existing BoneVQA answers use one-hot scores; this is a disclosed
    dataset adaptation of the author's potentially soft VQA score vectors.
    """
    if targets.shape != logits.shape or raw_target_words.ndim != 3:
        raise ValueError('Incorrect target shapes')
    if generated_mean.shape != raw_target_words.mean(1).shape:
        raise ValueError('Incorrect consistency shapes')
    if not torch.isfinite(targets).all() or not ((targets >= 0) & (targets <= 1)).all():
        raise ValueError('VQA scores must be finite in [0,1]')
    answer = F.binary_cross_entropy_with_logits(logits, targets)*logits.shape[1]
    consistency = (1-F.cosine_similarity(generated_mean, raw_target_words.mean(1))).sum()
    return {'total': answer + eta*consistency, 'answer': answer, 'consistency_sum': consistency}
