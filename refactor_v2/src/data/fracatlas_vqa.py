import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import DATA_ROOT, RecordVQADataset, load_json, save_json
from ..preprocessing import canonicalize_raster

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
                 with_masks=True, with_boxes=True, boxes_from=None, boxes_as_masks=False, force_rebuild=False,
                 image_mode="normal", roi_control="real", roi_seed=20260920, records_path=None,
                 sealed_records_path=None, seal_path=None, project_root=None):
        if sealed_records_path is not None:
            if records_path is not None:
                raise ValueError('Use records_path or sealed_records_path, not both')
            if seal_path is None or project_root is None:
                raise ValueError('Sealed FracAtlas requires seal_path and project_root')
            from .sealed_manifest import load_sealed_records
            records = load_sealed_records(sealed_records_path, seal_path, project_root)
        elif records_path is not None:
            records = load_json(records_path)
        else:
            data = build_fracatlas_vqa(root, force=force_rebuild)
            records = data[split]
        if max_samples is not None:
            records = records[:max_samples]
        self.with_masks = with_masks
        self.with_boxes = with_boxes
        self.box_du_doan = load_json(boxes_from) if boxes_from else None
        self.boxes_as_masks = boxes_as_masks
        if image_mode not in ("normal", "gray"):
            raise ValueError(f"image_mode không hợp lệ: {image_mode}")
        if roi_control not in ("real", "shuffled", "generic", "matched", "matched_all"):
            raise ValueError(f"roi_control không hợp lệ: {roi_control}")
        if roi_control in ("matched", "matched_all") and self.box_du_doan is None:
            raise ValueError(f"roi_control={roi_control!r} yêu cầu boxes_from; từ chối control không có tác dụng")
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
        positive_ids = [iid for iid in ids if self.box_du_doan and self.box_du_doan.get(iid)]
        self.positive_roi_donors = {}
        if positive_ids:
            for index, iid in enumerate(ids):
                # Stable, label-independent assignment from the empirical
                # detector-positive geometry pool.  This lets negative images
                # receive a same-distribution placebo without using lesion
                # annotations or inventing full-image boxes.
                self.positive_roi_donors[iid] = unique[positive_ids[index % len(positive_ids)]]
        self.canonical_sizes = {}
        if roi_control in ("shuffled", "matched_all"):
            for iid, candidate in unique.items():
                path = Path(root) / candidate["image"]
                if path.exists():
                    with Image.open(path) as raw:
                        self.canonical_sizes[iid] = canonicalize_raster(raw).size
                else:
                    self.canonical_sizes[iid] = (
                        int(candidate.get("width") or 1), int(candidate.get("height") or 1)
                    )
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

    @staticmethod
    def _boxes_to_masks(boxes, size):
        """Rasterize proposal boxes after clipping them to the recipient image."""
        width, height = size
        masks = []
        for box in boxes or []:
            x1, y1, x2, y2 = box[:4]
            x1 = max(0, min(width, int(round(x1))))
            x2 = max(0, min(width, int(round(x2))))
            y1 = max(0, min(height, int(round(y1))))
            y2 = max(0, min(height, int(round(y2))))
            mask = np.zeros((height, width), dtype=np.uint8)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
                masks.append(mask)
        return np.stack(masks) if masks else np.zeros((0, height, width), dtype=np.uint8)

    @staticmethod
    def _scale_boxes(boxes, source_size, target_size):
        """Map donor proposals by normalized coordinates into a recipient image."""
        if not boxes:
            return None
        source_w, source_h = source_size
        target_w, target_h = target_size
        if source_w <= 0 or source_h <= 0:
            source_w, source_h = target_w, target_h
        scale_x, scale_y = target_w / float(source_w), target_h / float(source_h)
        scaled = []
        for box in boxes:
            mapped = [box[0] * scale_x, box[1] * scale_y,
                      box[2] * scale_x, box[3] * scale_y]
            # Preserve detector confidence/metadata if present.
            mapped.extend(box[4:])
            scaled.append(mapped)
        return scaled

    @staticmethod
    def _matched_placebo_boxes(boxes, size):
        """Relocate proposals while preserving their clipped pixel geometry.

        Candidate positions are chosen to minimize overlap with every original
        proposal and then maximize displacement.  The transformation is fully
        deterministic, which keeps repeated QA rows for an image identical.
        An empty proposal set remains empty; this control never invents ROI
        presence where the detector produced none.
        """
        width, height = size
        clipped = []
        for box in boxes or []:
            x1, y1, x2, y2 = box[:4]
            x1 = max(0, min(width, int(round(x1))))
            x2 = max(0, min(width, int(round(x2))))
            y1 = max(0, min(height, int(round(y1))))
            y2 = max(0, min(height, int(round(y2))))
            if x2 > x1 and y2 > y1:
                clipped.append((x1, y1, x2, y2, list(box[4:])))
        if not clipped:
            return None

        originals = [box[:4] for box in clipped]

        def overlap_area(candidate):
            total = 0
            for original in originals:
                ix1, iy1 = max(candidate[0], original[0]), max(candidate[1], original[1])
                ix2, iy2 = min(candidate[2], original[2]), min(candidate[3], original[3])
                total += max(0, ix2 - ix1) * max(0, iy2 - iy1)
            return total

        relocated = []
        for x1, y1, x2, y2, tail in clipped:
            box_w, box_h = x2 - x1, y2 - y1
            xs = sorted({0, max(0, width - box_w), max(0, (width - box_w) // 2)})
            ys = sorted({0, max(0, height - box_h), max(0, (height - box_h) // 2)})
            candidates = [[x, y, x + box_w, y + box_h] for x in xs for y in ys]
            old_cx, old_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # First minimize overlap with real proposals; among ties choose the
            # farthest position.  Lexicographic coordinates make ties stable.
            candidates.sort(key=lambda c: (
                overlap_area(c),
                -((c[0] + c[2]) / 2.0 - old_cx) ** 2
                -((c[1] + c[3]) / 2.0 - old_cy) ** 2,
                c[0], c[1],
            ))
            chosen = candidates[0]
            relocated.append(chosen + tail)
        return relocated

    def _controlled_detector_boxes(self, record, image):
        """Apply ROI controls to detector proposals before mask rasterization."""
        image_id = record["image_id"]
        real_boxes = self.box_du_doan.get(image_id) or None
        if self.roi_control == "real":
            # Preserve the pre-fix/default behavior exactly.
            return real_boxes
        if self.roi_control == "generic":
            width, height = image.size
            return [[int(width * 0.25), int(height * 0.25),
                     int(width * 0.75), int(height * 0.75)]]
        if self.roi_control == "matched":
            return self._matched_placebo_boxes(real_boxes, image.size)
        if self.roi_control == "matched_all":
            placebo_source = real_boxes
            if not placebo_source:
                donor = self.positive_roi_donors.get(image_id)
                if donor is None:
                    raise RuntimeError("matched_all không có detector-positive donor")
                donor_boxes = self.box_du_doan.get(donor["image_id"]) or None
                donor_size = self.canonical_sizes.get(
                    donor["image_id"],
                    (int(donor.get("width") or image.width), int(donor.get("height") or image.height)),
                )
                placebo_source = self._scale_boxes(donor_boxes, donor_size, image.size)
            return self._matched_placebo_boxes(placebo_source, image.size)

        donor = self.roi_donors.get(image_id, record)
        donor_boxes = self.box_du_doan.get(donor["image_id"]) or None
        donor_size = self.canonical_sizes.get(
            donor["image_id"],
            (int(donor.get("width") or image.width), int(donor.get("height") or image.height)),
        )
        return self._scale_boxes(donor_boxes, donor_size, image.size)

    def build_masks(self, record, image):
        if self.box_du_doan is not None:
            hop = self._controlled_detector_boxes(record, image)
            if not self.boxes_as_masks or not hop:
                return np.zeros((0, image.height, image.width), dtype=np.uint8), hop
            return self._boxes_to_masks(hop, image.size), hop
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
