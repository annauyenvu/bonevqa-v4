import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "train_detector_v2.py"
    spec = importlib.util.spec_from_file_location("train_detector_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_positive_anchored_sampler_covers_negatives_and_never_emits_all_negative():
    module = _module()

    class Dataset:
        muc = [
            ("p0", ("p0.png", [[0, 0, 1, 1]])),
            ("p1", ("p1.png", [[0, 0, 1, 1]])),
            ("n0", ("n0.png", [])),
            ("n1", ("n1.png", [])),
            ("n2", ("n2.png", [])),
            ("n3", ("n3.png", [])),
            ("n4", ("n4.png", [])),
        ]

    sampler = module.PositiveAnchoredBatchSampler(Dataset(), batch_size=3, seed=7)
    batches = list(sampler)
    negative_seen = []
    for batch in batches:
        assert len(batch) == 3
        assert any(Dataset.muc[i][1][1] for i in batch)
        negative_seen.extend(i for i in batch if not Dataset.muc[i][1][1])
    assert set(negative_seen) == {2, 3, 4, 5, 6}
