import glob
import os
import random
from collections import Counter

from .common import DATA_ROOT, RecordVQADataset, infer_region, is_yes_no, load_json, normalize_answer, save_json

PMC_ROOT = DATA_ROOT / "pmc_vqa"
SPLIT_FILES = {"train": "train.json", "val": "val.json", "test": "test.json",
               "train_lon": "train_lon.json"}
FALLBACK_LIMIT = {"train": 5000, "test": 2000, "train_lon": 5000}
VAL_FRACTION = 0.05
TOP_VOCAB = 300
MIN_FREQ = 3
CHOICE_KEYS = ("A", "B", "C", "D")


def _clean(s):
    return (s or "").strip()


def _strip_choice_prefix(text):
    for key in CHOICE_KEYS:
        if text.upper().startswith(f"{key}:"):
            return text[2:].strip()
    return text


def export_from_parquet(split, root=PMC_ROOT, limit=None):
    import pyarrow.parquet as pq

    parquet_dir = root / "hf" / "data"
    files = sorted(glob.glob(str(parquet_dir / f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"Không có parquet PMC-VQA cho split {split} tại {parquet_dir}")
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    limit = limit or FALLBACK_LIMIT[split]
    rows = []
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg).to_pylist():
                if len(rows) >= limit:
                    break
                name = r["Figure_path"]
                p = img_dir / name
                if not p.exists():
                    with open(p, "wb") as f:
                        f.write(r["image"]["bytes"])
                choices = {k: _clean(r.get(f"Choice {k}")) for k in CHOICE_KEYS}
                rows.append({
                    "image": "images/" + name,
                    "question": _clean(r["Question"]),
                    "answer": _clean(r["Answer"]),
                    "choices": choices,
                    "answer_label": _clean(r.get("Answer_label")),
                })
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    save_json(rows, root / SPLIT_FILES[split])
    return rows


def _resolve_answer(r):
    answer = _strip_choice_prefix(_clean(r.get("answer")))
    choices = r.get("choices") or {}
    label = _clean(r.get("answer_label")).upper().strip(" .:()")
    if label in choices and choices[label]:
        text = _strip_choice_prefix(choices[label])
        if text:
            answer = text
    return answer


def _load_rows(split, root, allow_export):
    if split == "train_lon":
        path = root / SPLIT_FILES["train_lon"]
        if path.exists():
            return load_json(path)
        raise FileNotFoundError(f"Chưa sinh {path}; chạy scripts/sinh_pmc_train_lon.py trước")
    if split == "test":
        path = root / SPLIT_FILES["test"]
        if path.exists():
            return load_json(path)
        if allow_export:
            return export_from_parquet("test", root)
        raise FileNotFoundError(f"Không tìm thấy PMC-VQA test tại {path}")
    path = root / SPLIT_FILES["train"]
    if path.exists():
        rows = load_json(path)
    elif allow_export:
        rows = export_from_parquet("train", root)
    else:
        raise FileNotFoundError(f"Không tìm thấy PMC-VQA train tại {path}")
    val_path = root / SPLIT_FILES["val"]
    if val_path.exists():
        val_rows = load_json(val_path)
        if split == "val":
            return val_rows
        val_images = {r["image"] for r in val_rows}
        return [r for r in rows if r["image"] not in val_images]
    images = sorted({r["image"] for r in rows})
    rng = random.Random(42)
    rng.shuffle(images)
    val_images = set(images[: max(1, int(len(images) * VAL_FRACTION))])
    if split == "val":
        return [r for r in rows if r["image"] in val_images]
    return [r for r in rows if r["image"] not in val_images]


def load_pmc_vqa_records(split, root=PMC_ROOT, allow_export=True):
    rows = _load_rows(split, root, allow_export)
    resolved = [(r, _resolve_answer(r)) for r in rows]
    freq = Counter(normalize_answer(a) for _, a in resolved if len(a.split()) <= 2)
    top = {a for a, _ in freq.most_common(TOP_VOCAB)}
    records = []
    for r, answer in resolved:
        norm = normalize_answer(answer)
        short = len(answer.split()) <= 2 and (norm in top or freq[norm] >= MIN_FREQ)
        answer_type = "CLOSED" if (is_yes_no(answer) or short) else "OPEN"
        question = _clean(r.get("question"))
        records.append({
            "image": r["image"],
            "image_id": os.path.splitext(os.path.basename(r["image"]))[0],
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "region": infer_region(question + " " + answer),
            "choices": r.get("choices"),
            "boxes": None,
        })
    return records


class PMCVQADataset(RecordVQADataset):
    def __init__(self, split="train", max_samples=None, transform=None, max_side=None, root=PMC_ROOT):
        records = load_pmc_vqa_records(split, root)
        if max_samples is not None:
            records = records[:max_samples]
        super().__init__(records, root, transform=transform, max_side=max_side, name="pmc_vqa")
