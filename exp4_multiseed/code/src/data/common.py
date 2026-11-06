import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

DATA_ROOT = Path(os.environ.get("BONEVQA_DATA", r"D:\BaoYen_work\data"))

REGION_KEYWORDS = [
    ("chest", ["chest", "lung", "pulmonary", "thorax", "thoracic", "cardiac", "heart", "mediastin", "pleura", "rib"]),
    ("abdomen", ["abdomen", "abdominal", "liver", "hepatic", "kidney", "renal", "bowel", "colon", "intestin", "spleen", "pancrea", "gallbladder", "stomach", "gastric"]),
    ("brain", ["brain", "head", "cranial", "skull", "cerebr", "intracranial", "ventricle", "cerebell", "brainstem", "sinus"]),
    ("neck", ["neck", "cervical", "thyroid", "larynx", "pharyn"]),
    ("spine", ["spine", "spinal", "vertebra", "lumbar", "thoracolumbar", "sacral", "disc"]),
    ("pelvis", ["pelvis", "pelvic", "bladder", "uterus", "prostate", "ovar"]),
    ("hip", ["hip", "femoral head", "femoral neck", "acetabul"]),
    ("hand", ["hand", "wrist", "finger", "metacarp", "phalan", "carpal", "thumb"]),
    ("shoulder", ["shoulder", "clavicle", "humer", "scapula", "glenoid", "acromio"]),
    ("knee", ["knee", "patella", "tibial plateau", "meniscus"]),
    ("leg", ["leg", "tibia", "fibula", "femur", "ankle", "foot", "calcane", "metatars"]),
    ("elbow", ["elbow", "radius", "ulna", "forearm", "olecranon"]),
    ("breast", ["breast", "mammo"]),
]


def infer_region(text):
    if not text:
        return None
    lowered = text.lower()
    for region, keys in REGION_KEYWORDS:
        for key in keys:
            if re.search(r"\b" + re.escape(key), lowered):
                return region
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def normalize_answer(answer):
    return re.sub(r"\s+", " ", str(answer).strip().lower()).strip(" .")


def is_yes_no(answer):
    return normalize_answer(answer) in ("yes", "no")


def resize_sample(image, masks, boxes, max_side):
    if not max_side:
        return image, masks, boxes
    w, h = image.size
    scale = max_side / float(max(w, h))
    if scale >= 1.0:
        return image, masks, boxes
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    image = image.resize((new_w, new_h), Image.BILINEAR)
    if masks is not None:
        resized = [np.array(Image.fromarray(m).resize((new_w, new_h), Image.NEAREST), dtype=np.uint8) for m in masks]
        masks = np.stack(resized, axis=0) if resized else None
    if boxes is not None:
        boxes = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in boxes]
    return image, masks, boxes


class RecordVQADataset(Dataset):
    def __init__(self, records, root, transform=None, max_side=None, name="dataset"):
        self.records = records
        self.root = Path(root)
        self.transform = transform
        self.max_side = max_side
        self.name = name

    def __len__(self):
        return len(self.records)

    def load_image(self, record):
        path = self.root / record["image"]
        with Image.open(path) as im:
            return im.convert("RGB")

    def build_masks(self, record, image):
        return None, record.get("boxes")

    def __getitem__(self, idx):
        record = self.records[idx]
        image = self.load_image(record)
        masks, boxes = self.build_masks(record, image)
        image, masks, boxes = resize_sample(image, masks, boxes, self.max_side)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "question": record["question"],
            "answer": record["answer"],
            "answer_type": record.get("answer_type", "OPEN"),
            "region": record.get("region"),
            "masks": masks,
            "boxes": boxes,
            "image_id": str(record.get("image_id", record["image"])),
        }
