import torch

from src.models.losses import lens_alignment_loss, multi_positive_info_nce_loss


def test_repeated_qa_single_image_retains_lens_supervision():
    image = torch.tensor([[1., 0., 0.], [1., 0., 0.]], requires_grad=True)
    text = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
    assert multi_positive_info_nce_loss(image, text, ["a", "a"]).item() == 0.0
    loss = lens_alignment_loss(image, text, ["a", "a"])
    loss.backward()
    assert loss.item() > 0
    assert image.grad.isfinite().all() and image.grad.abs().max() > 0


def test_multiple_images_keep_original_contrastive_objective():
    image, text = torch.randn(3, 5), torch.randn(3, 5)
    groups = ["a", "a", "b"]
    assert torch.equal(lens_alignment_loss(image, text, groups),
                       multi_positive_info_nce_loss(image, text, groups))
