import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import DATA_ROOT, RecordVQADataset, load_json, save_json

FRACATLAS_ROOT = DATA_ROOT / "fracatlas" / "FracAtlas"
SPLIT_RATIO = (0.70, 0.15, 0.15)
SEED = 42
REGIONS = ["hand", "leg", "hip", "shoulder"]
VIEWS = ["frontal", "lateral", "oblique"]
NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

try:
    import cv2
except Exception:
    cv2 = None


def polygons_to_masks(polygons_per_ann, height, width):
    masks = []
    for polygons in polygons_per_ann:
        mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            if cv2 is not None:
                cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            else:
                pil_mask = Image.fromarray(mask)
                ImageDraw.Draw(pil_mask).polygon([tuple(p) for p in pts.tolist()], fill=1)
                mask = np.array(pil_mask, dtype=np.uint8)
        masks.append(mask)
    if not masks:
        return None
    return np.stack(masks, axis=0)


def _read_meta(root):
    rows = {}
    with open(root / "dataset.csv", "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows[r["image_id"]] = {k: (int(v) if k != "image_id" else v) for k, v in r.items()}
    return rows


def _read_coco(root):
    coco = load_json(root / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json")
    id_to_file = {im["id"]: im for im in coco["images"]}
    per_file = defaultdict(list)
    for ann in coco["annotations"]:
        im = id_to_file[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        per_file[im["file_name"]].append({
            "box": [x, y, x + w, y + h],
            "segmentation": ann["segmentation"],
            "width": im["width"],
            "height": im["height"],
        })
    return per_file


def _image_rel_path(root, file_name, fractured):
    sub = "Fractured" if fractured else "Non_fractured"
    rel = Path("images") / sub / file_name
    if (root / rel).exists():
        return str(rel).replace("\\", "/")
    other = Path("images") / ("Non_fractured" if fractured else "Fractured") / file_name
    if (root / other).exists():
        return str(other).replace("\\", "/")
    return str(rel).replace("\\", "/")


def _regions_of(meta):
    return [r for r in REGIONS if meta.get(r, 0) == 1]


def _views_of(meta):
    return [v for v in VIEWS if meta.get(v, 0) == 1]


def _position(box, width, height):
    cx = (box[0] + box[2]) / 2.0 / max(width, 1)
    cy = (box[1] + box[3]) / 2.0 / max(height, 1)
    vertical = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    horizontal = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return f"middle-{horizontal}"
    if horizontal == "center":
        return f"{vertical}-center"
    return f"{vertical}-{horizontal}"


def _article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _join_words(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


class YesNoBalancer:
    def __init__(self):
        self.counts = Counter()

    def prefer(self):
        return "yes" if self.counts["yes"] <= self.counts["no"] else "no"

    def add(self, answer):
        self.counts[answer] += 1


def _closed_candidates(info, rng):
    regions = info["regions"]
    region_text = _join_words(regions) if regions else "body"
    wrong_regions = [r for r in REGIONS if r not in regions]
    fractured = info["fractured"] == 1
    fixed = [
        ("Is there a fracture in this X-ray?", "yes" if fractured else "no"),
    ]
    optional = []
    if regions:
        optional.append((f"Is this an X-ray of the {region_text}?", "yes"))
        optional.append((f"Is the {region_text} fractured?", "yes" if fractured else "no"))
    if wrong_regions:
        wr = rng.choice(wrong_regions)
        optional.append((f"Is this an X-ray of the {wr}?", "no"))
        optional.append((f"Is the {wr} fractured?", "no"))
    optional.append(("Is there any orthopedic hardware or implant visible?", "yes" if info["hardware"] == 1 else "no"))
    if fractured:
        optional.append(("Are there multiple fractures?", "yes" if info["fracture_count"] > 1 else "no"))
    views = info["views"]
    if views:
        v = rng.choice(views)
        optional.append((f"Is this {_article(v)} {v} view X-ray?", "yes"))
        wrong_views = [x for x in VIEWS if x not in views]
        if wrong_views:
            wv = rng.choice(wrong_views)
            optional.append((f"Is this {_article(wv)} {wv} view X-ray?", "no"))
    optional.append(("Does this image contain more than one scan?", "yes" if info["multiscan"] == 1 else "no"))
    return fixed, optional


def _open_candidates(info):
    regions = info["regions"]
    region_text = _join_words(regions) if regions else "unknown"
    fractured = info["fractured"] == 1
    n = info["fracture_count"]
    positions = []
    for p in info["positions"]:
        if p not in positions:
            positions.append(p)
    qa = [
        ("Which body region is shown in this X-ray?", region_text),
        ("What abnormality is seen in this X-ray?", "fracture" if fractured else "no abnormality"),
        ("What type of imaging is this?", "x-ray"),
    ]
    if fractured:
        qa.append(("How many fracture sites are visible?", NUMBER_WORDS.get(n, str(n))))
        if positions:
            qa.append(("Where is the fracture located?", f"{_join_words(positions)} of the {region_text}"))
        if n > 1:
            desc = f"{NUMBER_WORDS.get(n, str(n)).capitalize()} fractures are seen in the {region_text}"
        else:
            desc = f"A fracture is seen in the {region_text}"
        if positions:
            desc += f" at the {_join_words(positions)}"
        qa.append(("Describe the finding.", desc + "."))
    else:
        qa.append(("How many fracture sites are visible?", "zero"))
        qa.append(("Describe the finding.", f"No fracture is seen in the {region_text}."))
    return qa


def generate_qa(info, rng, balancer):
    fixed, optional = _closed_candidates(info, rng)
    n_closed_extra = rng.randint(1, 2)
    chosen = list(fixed)
    for q, a in chosen:
        balancer.add(a)
    rng.shuffle(optional)
    for _ in range(n_closed_extra):
        pref = balancer.prefer()
        pick = None
        for cand in optional:
            if cand[1] == pref:
                pick = cand
                break
        if pick is None and optional:
            pick = optional[0]
        if pick is None:
            break
        optional.remove(pick)
        chosen.append(pick)
        balancer.add(pick[1])
    closed = [{"question": q, "answer": a, "answer_type": "CLOSED"} for q, a in chosen]
    open_pool = _open_candidates(info)
    n_open = rng.randint(1, min(3, len(open_pool)))
    must = [open_pool[0]] if rng.random() < 0.5 else [open_pool[-1]]
    rest = [x for x in open_pool if x not in must]
    rng.shuffle(rest)
    open_chosen = must + rest[: max(0, n_open - 1)]
    opens = [{"question": q, "answer": a, "answer_type": "OPEN"} for q, a in open_chosen]
    return closed + opens


def _split_images(image_ids, meta, seed=SEED):
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for iid in sorted(image_ids):
        groups[meta[iid]["fractured"]].append(iid)
    splits = {"train": [], "val": [], "test": []}
    for _, ids in sorted(groups.items()):
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * SPLIT_RATIO[0]))
        n_val = int(round(n * SPLIT_RATIO[1]))
        splits["train"] += ids[:n_train]
        splits["val"] += ids[n_train:n_train + n_val]
        splits["test"] += ids[n_train + n_val:]
    for ids in splits.values():
        rng.shuffle(ids)
    return splits


def build_fracatlas_vqa(root=FRACATLAS_ROOT, force=False):
    root = Path(root)
    out_dir = root.parent
    outputs = {s: out_dir / f"vqa_{s}.json" for s in ("train", "val", "test")}
    if not force and all(p.exists() for p in outputs.values()):
        return {s: load_json(p) for s, p in outputs.items()}
    if not (root / "dataset.csv").exists():
        raise FileNotFoundError(f"Không tìm thấy FracAtlas tại {root} (thiếu dataset.csv)")
    meta = _read_meta(root)
    coco = _read_coco(root)
    splits = _split_images(list(meta.keys()), meta)
    balancer = YesNoBalancer()
    rng = random.Random(SEED)
    result = {}
    for split in ("train", "val", "test"):
        records = []
        for iid in splits[split]:
            m = meta[iid]
            anns = coco.get(iid, [])
            fractured = m["fractured"]
            rel = _image_rel_path(root, iid, fractured)
            width = anns[0]["width"] if anns else None
            height = anns[0]["height"] if anns else None
            if width is None:
                try:
                    with Image.open(root / rel) as im:
                        width, height = im.size
                except Exception:
                    width, height = 1, 1
            boxes = [a["box"] for a in anns]
            positions = [_position(b, width, height) for b in boxes]
            info = {
                "regions": _regions_of(m),
                "views": _views_of(m),
                "fractured": fractured,
                "hardware": m["hardware"],
                "multiscan": m["multiscan"],
                "fracture_count": m["fracture_count"] if not anns else max(m["fracture_count"], len(anns)),
                "positions": positions,
            }
            region_text = _join_words(info["regions"]) if info["regions"] else None
            for qa in generate_qa(info, rng, balancer):
                records.append({
                    "image": rel,
                    "image_id": iid.rsplit(".", 1)[0],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answer_type": qa["answer_type"],
                    "region": region_text,
                    "fractured": fractured,
                    "hardware": m["hardware"],
                    "fracture_count": info["fracture_count"],
                    "views": info["views"],
                    "boxes": boxes if boxes else None,
                    "segmentation": [a["segmentation"] for a in anns] if anns else None,
                    "width": width,
                    "height": height,
                })
        save_json(records, outputs[split])
        result[split] = records
    save_json({k: sorted(v) for k, v in splits.items()}, out_dir / "vqa_image_splits.json")
    return result


class FracAtlasVQADataset(RecordVQADataset):
    def __init__(self, split="train", max_samples=None, transform=None, max_side=None, root=FRACATLAS_ROOT,
                 with_masks=True, with_boxes=True, boxes_from=None, force_rebuild=False,
                 image_mode="normal", roi_control="real", roi_seed=20260920):
        data = build_fracatlas_vqa(root, force=force_rebuild)
        records = data[split]
        if max_samples is not None:
            records = records[:max_samples]
        self.with_masks = with_masks
        self.with_boxes = with_boxes
        self.box_du_doan = load_json(boxes_from) if boxes_from else None
        if image_mode not in ("normal", "gray"):
            raise ValueError(f"image_mode không hợp lệ: {image_mode}")
        if roi_control not in ("real", "shuffled", "generic"):
            raise ValueError(f"roi_control không hợp lệ: {roi_control}")
        self.image_mode = image_mode
        self.roi_control = roi_control
        # Permute at image level, not QA level: repeated questions for one
        # radiograph always receive the same label-independent donor ROI.
        unique = {}
        for r in records:
            unique.setdefault(r["image_id"], r)
        ids = sorted(unique)
        donors = list(ids)
        rng = random.Random(roi_seed)
        if len(ids) > 1:
            # A true derangement is essential: no recipient may keep its own
            # ROI, while every donor ROI is still used exactly once.
            for _ in range(100):
                rng.shuffle(donors)
                if all(iid != donor for iid, donor in zip(ids, donors)):
                    break
            else:
                donors = ids[1:] + ids[:1]
        self.roi_donors = {iid: unique[donor] for iid, donor in zip(ids, donors)}
        super().__init__(records, root, transform=transform, max_side=max_side, name="fracatlas")

    @staticmethod
    def _resize_masks(masks, size):
        w, h = size
        if masks is None or len(masks) == 0:
            return np.zeros((0, h, w), dtype=np.uint8)
        return np.stack([np.asarray(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST), dtype=np.uint8)
                         for m in masks])

    def load_image(self, record):
        image = super().load_image(record)
        if self.image_mode == "gray":
            return Image.new("RGB", image.size, (128, 128, 128))
        return image

    def build_masks(self, record, image):
        if self.box_du_doan is not None:
            hop = self.box_du_doan.get(record["image_id"]) or None
            return None, hop
        boxes = record.get("boxes") if self.with_boxes else None
        source = self.roi_donors.get(record["image_id"], record) if self.roi_control == "shuffled" else record
        seg = source.get("segmentation")
        if self.roi_control == "generic":
            w, h = image.size
            x1, x2 = int(w * 0.25), int(w * 0.75)
            y1, y2 = int(h * 0.25), int(h * 0.75)
            mask = np.zeros((1, h, w), dtype=np.uint8)
            mask[0, y1:y2, x1:x2] = 1
            return mask, None
        if not seg or not self.with_masks:
            return np.zeros((0, image.height, image.width), dtype=np.uint8), boxes
        # Donor polygons are in donor-image coordinates; rasterize there then
        # resize to the recipient image so shuffle preserves normalized shape.
        donor_w, donor_h = int(source.get("width") or image.width), int(source.get("height") or image.height)
        masks = polygons_to_masks(seg, donor_h, donor_w)
        return self._resize_masks(masks, image.size), boxes


def summarize(records):
    c = Counter()
    for r in records:
        c["qa"] += 1
        c[f"type_{r['answer_type']}"] += 1
        if r["answer_type"] == "CLOSED":
            c[f"ans_{r['answer']}"] += 1
    c["images"] = len({r["image_id"] for r in records})
    c["images_fractured"] = len({r["image_id"] for r in records if r["fractured"] == 1})
    return dict(c)
