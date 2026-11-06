from itertools import product
import pytest
import torch
from src.context_controls import component_disjoint_donors, component_balanced_batches, ComponentBatchSampler
from src.models.fusion import MultimodalFusion


def test_donors_exhaustive_small_group_assignments():
    for count in range(2, 8):
        for groups in product(range(3), repeat=count):
            feasible = max(groups.count(group) for group in set(groups)) * 2 <= count
            if feasible:
                donors = component_disjoint_donors(groups)
                assert sorted(donors) == list(range(count))
                assert all(groups[i] != groups[j] for i, j in enumerate(donors))
                assert donors == component_disjoint_donors(groups)
            else:
                with pytest.raises(ValueError, match="dominant"):
                    component_disjoint_donors(groups)


def test_missing_component_ids_rejected():
    with pytest.raises(ValueError, match="present"):
        component_disjoint_donors([None, "a"])


@pytest.mark.parametrize("count", [7, 8, 9, 15, 16, 17, 30])
def test_common_batch_schedule_preserves_rows_and_seed(count):
    groups = [index % 5 for index in range(count)]
    batches = component_balanced_batches(groups, 8, 42)
    assert sorted(i for batch in batches for i in batch) == list(range(count))
    assert batches == component_balanced_batches(groups, 8, 42)
    assert all(2 <= len(batch) <= 8 for batch in batches)
    for batch in batches:
        donors = component_disjoint_donors([groups[i] for i in batch])
        assert all(groups[batch[i]] != groups[batch[j]] for i, j in enumerate(donors))


def test_impossible_batch_schedule_fails_not_silently_drops():
    with pytest.raises(ValueError, match="without dropping"):
        component_balanced_batches(["a"] * 9 + ["b"], 8, 42)


def test_sampler_epoch_resume_and_dataloader_preserve_component_ids():
    from torch.utils.data import DataLoader
    groups = [index % 5 for index in range(17)]
    sampler = ComponentBatchSampler(groups, 8, 42)
    initial = list(sampler)
    sampler.set_epoch(1)
    expected = list(sampler)
    restored = ComponentBatchSampler(groups, 8, 42)
    restored.set_epoch(1)
    assert list(restored) == expected
    assert initial != expected
    assert restored.binding() == sampler.binding()
    loader = DataLoader(list(range(17)), batch_sampler=restored)
    assert [batch.tolist() for batch in loader] == expected


def test_lf_donor_mask_and_context_follow_same_permutation():
    torch.manual_seed(4)
    fusion = MultimodalFusion(8, 2, 0.0, use_latent=True).eval()
    fusion.gamma_latent.data.fill_(0.1)
    image, text = torch.randn(4, 3, 8), torch.randn(4, 4, 8)
    prompt = torch.randn(1, 2, 8).expand(4, -1, -1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    groups = ["a", "a", "b", "c"]
    donors = component_disjoint_donors(groups)
    reference = fusion(image[donors], text[donors], prompt, mask[donors])
    control = fusion(image, text, prompt, mask, latent_context_control="shuffled", context_group_ids=groups)
    assert torch.equal(reference["latent_delta"], control["latent_delta"])
    baseline = fusion(image, text, prompt, mask)
    assert torch.equal(baseline["f_fi"], control["f_fi"])
    assert torch.equal(baseline["f_fl"], control["f_fl"])
