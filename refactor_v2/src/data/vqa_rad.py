import random

from .common import DATA_ROOT, RecordVQADataset, infer_region, is_yes_no, load_json

VQA_RAD_ROOT = DATA_ROOT / "vqa_rad"
SPLIT_FILES = {"train": "train.json", "val": "train.json", "test": "test.json"}
VAL_FRACTION = 0.1
VAL_SEED = 42


def _parse_records(rows):
    records = []
    for r in rows:
        answer = str(r["answer"]).strip()
        answer_type = r.get("answer_type") or ("CLOSED" if is_yes_no(answer) else "OPEN")
        records.append({
            "image": r["image"],
            "image_id": r["image"].split("/")[-1].rsplit(".", 1)[0],
            "question": r["question"].strip(),
            "answer": answer,
            "answer_type": answer_type.upper(),
            "region": infer_region(r["question"] + " " + answer),
            "boxes": None,
        })
    return records


def _tach_val_theo_anh(records, fraction=VAL_FRACTION, seed=VAL_SEED):
    anh = sorted({r["image_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(anh)
    n_val = max(1, int(round(len(anh) * fraction)))
    anh_val = set(anh[:n_val])
    train = [r for r in records if r["image_id"] not in anh_val]
    val = [r for r in records if r["image_id"] in anh_val]
    return train, val


def load_vqa_rad_records(split, root=VQA_RAD_ROOT, val_from_train=True):
    path = root / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy VQA-RAD split {split} tại {path}")
    records = _parse_records(load_json(path))
    if split == "test" or not val_from_train:
        return records
    train, val = _tach_val_theo_anh(records)
    return val if split == "val" else train


class VQARADDataset(RecordVQADataset):
    def __init__(self, split="train", max_samples=None, transform=None, max_side=None, root=VQA_RAD_ROOT,
                 val_from_train=True):
        records = load_vqa_rad_records(split, root, val_from_train=val_from_train)
        if max_samples is not None:
            # Few-shot adaptation must not be determined by source-file order.
            # Keep the held-out image-level validation split unchanged and draw
            # a reproducible, answer-type-stratified subset from train only.
            if split == "train" and max_samples < len(records):
                rng = random.Random(42)
                by_type = {}
                for row in records:
                    by_type.setdefault(row["answer_type"], []).append(row)
                picked = []
                remaining = max_samples
                groups = list(by_type.items())
                for i, (_, group) in enumerate(groups):
                    rng.shuffle(group)
                    n = round(max_samples * len(group) / len(records)) if i < len(groups) - 1 else remaining
                    n = min(len(group), max(0, n))
                    picked.extend(group[:n])
                    remaining -= n
                records = picked
            else:
                records = records[:max_samples]
        super().__init__(records, root, transform=transform, max_side=max_side, name="vqa_rad")
