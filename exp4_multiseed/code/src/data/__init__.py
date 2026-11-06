from collections import Counter
from pathlib import Path

from .common import DATA_ROOT, RecordVQADataset, infer_region, load_json, normalize_answer, save_json
from .fracatlas_vqa import FracAtlasVQADataset, build_fracatlas_vqa
from .pmc_vqa import PMCVQADataset, load_pmc_vqa_records
from .prompts import build_lens, build_visual_prompts, draw_boxes, masks_to_boxes, overlay_masks, textual_prompt
from .roco import ROCODataset, load_roco_records
from .vqa_rad import VQARADDataset, load_vqa_rad_records

DATASET_NAMES = ("fracatlas", "fracatlas_content", "vqa_rad", "roco", "pmc_vqa")
ANSWER_VOCAB = {}


LOP_DATASET = {}


def _loc_kwargs(lop, kw):
    import inspect

    cho_phep = set(inspect.signature(lop.__init__).parameters)
    return {k: v for k, v in kw.items() if k in cho_phep}


def build_dataset(name, split="train", max_samples=None, transform=None, **kw):
    name = name.lower()
    if "+" in name:
        from torch.utils.data import ConcatDataset

        cac_ten = [t.strip() for t in name.split("+") if t.strip()]
        cac_ds = [build_dataset(t, split, max_samples, transform, **kw) for t in cac_ten]
        gop = ConcatDataset(cac_ds)
        gop.name = name
        gop.records = [r for ds in cac_ds for r in getattr(ds, "records", [])]
        return gop
    if name == "fracatlas":
        return FracAtlasVQADataset(split=split, max_samples=max_samples, transform=transform, **_loc_kwargs(FracAtlasVQADataset, kw))
    if name == "fracatlas_content":
        root = DATA_ROOT / "fracatlas_content" / "FracAtlas"
        return FracAtlasVQADataset(split=split, max_samples=max_samples, transform=transform, root=root,
                                   **_loc_kwargs(FracAtlasVQADataset, kw))
    if name == "vqa_rad":
        return VQARADDataset(split=split, max_samples=max_samples, transform=transform, **_loc_kwargs(VQARADDataset, kw))
    if name == "roco":
        return ROCODataset(split=split, max_samples=max_samples, transform=transform, **_loc_kwargs(ROCODataset, kw))
    if name == "pmc_vqa":
        return PMCVQADataset(split=split, max_samples=max_samples, transform=transform, **_loc_kwargs(PMCVQADataset, kw))
    raise ValueError(f"Dataset không hỗ trợ: {name}. Chọn một trong {DATASET_NAMES}")


def collate_fn(batch):
    keys = ("images", "questions", "answers", "answer_types", "regions", "masks", "boxes", "image_ids")
    out = {k: [] for k in keys}
    for sample in batch:
        out["images"].append(sample["image"])
        out["questions"].append(sample["question"])
        out["answers"].append(sample["answer"])
        out["answer_types"].append(sample["answer_type"])
        out["regions"].append(sample.get("region"))
        out["masks"].append(sample.get("masks"))
        out["boxes"].append(sample.get("boxes"))
        out["image_ids"].append(sample["image_id"])
    return out


def _train_records(name):
    name = name.lower()
    if "+" in name:
        ra = []
        for t in name.split("+"):
            ra.extend(_train_records(t.strip()))
        return ra
    if name == "fracatlas":
        return build_fracatlas_vqa()["train"]
    if name == "fracatlas_content":
        root = DATA_ROOT / "fracatlas_content" / "FracAtlas"
        return build_fracatlas_vqa(root)["train"]
    if name == "vqa_rad":
        return load_vqa_rad_records("train")
    if name == "roco":
        return load_roco_records("train")
    if name == "pmc_vqa":
        return load_pmc_vqa_records("train")
    raise ValueError(f"Dataset không hỗ trợ: {name}")


def build_answer_vocab(dataset_names, top_k=None, closed_only=False, cache_dir=DATA_ROOT, force=False):
    if isinstance(dataset_names, str):
        dataset_names = [t.strip() for t in dataset_names.split("+") if t.strip()]
    names = [n.lower() for n in dataset_names]
    key = "_".join(names) + ("_closed" if closed_only else "") + (f"_top{top_k}" if top_k else "")
    if key in ANSWER_VOCAB and not force:
        return ANSWER_VOCAB[key]
    cache_path = Path(cache_dir) / f"answer_vocab_{key}.json"
    if cache_path.exists() and not force:
        vocab = load_json(cache_path)
        ANSWER_VOCAB[key] = vocab
        return vocab
    counter = Counter()
    for n in names:
        for r in _train_records(n):
            if closed_only and r.get("answer_type") != "CLOSED":
                continue
            counter[normalize_answer(r["answer"])] += 1
    ordered = [a for a, _ in counter.most_common()]
    for special in ("yes", "no"):
        if special in ordered:
            ordered.remove(special)
    vocab = ["yes", "no"] + ordered
    if top_k:
        vocab = vocab[:top_k]
    save_json(vocab, cache_path)
    ANSWER_VOCAB[key] = vocab
    return vocab


__all__ = [
    "DATA_ROOT", "DATASET_NAMES", "ANSWER_VOCAB", "RecordVQADataset",
    "build_dataset", "collate_fn", "build_answer_vocab",
    "FracAtlasVQADataset", "VQARADDataset", "ROCODataset", "PMCVQADataset",
    "build_fracatlas_vqa", "load_vqa_rad_records", "load_roco_records", "load_pmc_vqa_records",
    "build_visual_prompts", "build_lens", "draw_boxes", "masks_to_boxes", "overlay_masks", "textual_prompt",
    "infer_region",
]
