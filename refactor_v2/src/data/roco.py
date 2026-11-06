import glob
import io
import os
from pathlib import Path

from .common import DATA_ROOT, RecordVQADataset, infer_region, load_json, save_json

ROCO_ROOT = DATA_ROOT / "roco"
SPLIT_FILES = {"train": "train.json", "val": "val.json", "test": "test.json"}
PARQUET_SPLIT = {"train": "train", "val": "validation", "test": "test"}
CAPTION_QUESTION = "Describe this radiology image."
FALLBACK_LIMIT = {"train": 5000, "val": 1000, "test": 1000}


def export_from_parquet(split, root=ROCO_ROOT, limit=None):
    import pyarrow.parquet as pq
    from PIL import Image

    parquet_dir = root / "hf" / "data"
    files = sorted(glob.glob(str(parquet_dir / f"{PARQUET_SPLIT[split]}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"Không có parquet ROCO cho split {split} tại {parquet_dir}")
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
                b = r["image"]["bytes"]
                im = Image.open(io.BytesIO(b))
                fmt = (im.format or "JPEG").lower()
                fmt = "jpg" if fmt == "jpeg" else fmt
                name = f"{r['image_id']}.{fmt}"
                p = img_dir / name
                if not p.exists():
                    with open(p, "wb") as f:
                        f.write(b)
                rows.append({"image": "images/" + name, "image_id": r["image_id"], "caption": r["caption"], "cui": r.get("cui")})
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    save_json(rows, root / SPLIT_FILES[split])
    return rows


def load_roco_records(split, root=ROCO_ROOT, allow_export=True, rows=None):
    path = Path(root) / SPLIT_FILES.get(split, split + ".json")
    if rows is not None:
        pass
    elif path.exists():
        rows = load_json(path)
    elif allow_export:
        rows = export_from_parquet(split, root)
    else:
        raise FileNotFoundError(f"Không tìm thấy ROCO split {split} tại {path}")
    records = []
    for r in rows:
        caption = str(r.get("caption", "")).strip()
        image = r["image"]
        records.append({
            "image": image,
            "image_id": str(r.get("image_id") or os.path.splitext(os.path.basename(image))[0]),
            "question": CAPTION_QUESTION,
            "answer": caption,
            "answer_type": "OPEN",
            "region": infer_region(caption),
            "boxes": None,
        })
    return records


class ROCODataset(RecordVQADataset):
    def __init__(self, split="train", max_samples=None, transform=None, max_side=None, root=ROCO_ROOT,
                 sealed_records_path=None, seal_path=None, project_root=None, flat_image_prefix=None):
        self.flat_image_prefix = flat_image_prefix
        rows = None
        if sealed_records_path is not None:
            if seal_path is None or project_root is None:
                raise ValueError('Sealed ROCO requires seal_path and project_root')
            from .sealed_manifest import load_sealed_records
            rows = load_sealed_records(sealed_records_path, seal_path, project_root)
        records = load_roco_records(split, root, rows=rows)
        if max_samples is not None:
            records = records[:max_samples]
        super().__init__(records, root, transform=transform, max_side=max_side, name="roco")
