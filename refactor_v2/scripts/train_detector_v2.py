import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data import build_fracatlas_vqa
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from src.preprocessing import RASTER_PREPROCESSING_VERSION, canonicalize_raster_and_boxes
from src.detector_protocol import load_detector_manifest, verify_detector_annotation_parity, file_sha256
from src.utils import get_device, save_json, set_seed, setup_env_dirs, setup_logger

setup_env_dirs()
logger = setup_logger()

CANH_DAI = 320


class BoPhatHienDataset(Dataset):
    def __init__(self, split, root=FRACATLAS_ROOT, canh_dai=CANH_DAI, tang_cuong=False, allowed_ids=None):
        data = build_fracatlas_vqa(root)
        anh = {}
        rows = data[split] if allowed_ids is None else [r for part in ("train", "val", "test") for r in data[part]]
        for r in rows:
            if allowed_ids is not None and r["image_id"] not in allowed_ids:
                continue
            # Keep negative radiographs with an empty target. Never replace an
            # empty lesion target by a full-image positive box.
            if r["image_id"] not in anh:
                anh[r["image_id"]] = (r["image"], r.get("boxes") or [])
        self.muc = sorted(anh.items())
        if allowed_ids is not None and set(anh) != set(allowed_ids):
            raise RuntimeError("Detector annotation source missing requested image IDs: " +
                               str(sorted(set(allowed_ids) - set(anh))[:20]))
        self.root = root
        self.canh_dai = canh_dai
        self.tang_cuong = tang_cuong

    def __len__(self):
        return len(self.muc)

    def __getitem__(self, i):
        image_id, (rel, boxes) = self.muc[i]
        with Image.open(os.path.join(self.root, rel)) as im:
            im, boxes = canonicalize_raster_and_boxes(im, boxes)
        w, h = im.size
        ty_le = self.canh_dai / max(w, h)
        nw, nh = max(1, int(round(w * ty_le))), max(1, int(round(h * ty_le)))
        resized = im.resize((nw, nh), Image.BILINEAR)
        pad_x = (self.canh_dai - nw) // 2
        pad_y = (self.canh_dai - nh) // 2
        im = Image.new("RGB", (self.canh_dai, self.canh_dai), (0, 0, 0))
        im.paste(resized, (pad_x, pad_y))
        hop = []
        for b in boxes:
            x1, y1, x2, y2 = [v * ty_le for v in b[:4]]
            x1, x2 = sorted((max(0.0, x1), min(float(nw), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(nh), y2)))
            x1, x2 = x1 + pad_x, x2 + pad_x
            y1, y2 = y1 + pad_y, y2 + pad_y
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                hop.append([x1, y1, x2, y2])
        anh = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        if self.tang_cuong and np.random.rand() < 0.5:
            anh = torch.flip(anh, dims=[2])
            hop = [[self.canh_dai - b[2], b[1], self.canh_dai - b[0], b[3]] for b in hop]
        boxes_t = torch.tensor(hop, dtype=torch.float32).reshape(-1, 4)
        muc_tieu = {"boxes": boxes_t,
                    "labels": torch.ones(len(hop), dtype=torch.int64),
                    "image_id": torch.tensor([i])}
        return anh, muc_tieu, image_id


def collate(batch):
    return tuple(zip(*batch))


class PositiveAnchoredBatchSampler:
    """Keep negative images while avoiding an all-negative Faster-RCNN batch.

    TorchVision's empty-target RPN path currently triggers an internal MPS
    error when every image in a training batch is negative. Each batch gets a
    real positive image and up to ``batch_size - 1`` distinct negatives. This
    is also close to class-balanced for FracAtlas (about 1:3 positive:negative).
    """

    def __init__(self, dataset, batch_size, seed=42):
        if batch_size < 2:
            raise ValueError("batch_size phải >= 2 để ghép ảnh dương và âm")
        self.positive = [i for i, item in enumerate(dataset.muc) if item[1][1]]
        self.negative = [i for i, item in enumerate(dataset.muc) if not item[1][1]]
        if not self.positive:
            raise RuntimeError("Detector training split không có ảnh dương")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        negative_batches = math.ceil(len(self.negative) / (self.batch_size - 1))
        return max(len(self.positive), negative_batches)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        positive = self.positive.copy()
        negative = self.negative.copy()
        rng.shuffle(positive)
        rng.shuffle(negative)
        width = self.batch_size - 1
        for batch_index in range(len(self)):
            batch = [positive[batch_index % len(positive)]]
            start = batch_index * width
            # Keep batch cardinality fixed so MPS can reuse one compiled graph.
            # Only the short tail is cycled (92/2393 negatives at batch_size=6
            # on the current clean split), and every original negative remains
            # covered at least once per epoch.
            if negative:
                batch.extend(negative[j % len(negative)] for j in range(start, start + width))
            yield batch


def tao_mo_hinh(architecture="mobilenet_320", weights="DEFAULT"):
    if weights is not None:
        enum = (torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                if architecture == "mobilenet_320" else torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        cached = os.path.join(torch.hub.get_dir(), "checkpoints", enum.url.rsplit("/", 1)[-1])
        if not os.path.isfile(cached):
            raise FileNotFoundError("Offline detector initialization requires cached weights: " + cached)
    weights_backbone = None if weights is None else "DEFAULT"
    if architecture == "mobilenet_320":
        mo_hinh = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, weights_backbone=weights_backbone)
    elif architecture == "resnet50_640":
        mo_hinh = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights, weights_backbone=weights_backbone)
    else:
        raise ValueError(f"Detector architecture không hỗ trợ: {architecture}")
    so_dac_trung = mo_hinh.roi_heads.box_predictor.cls_score.in_features
    mo_hinh.roi_heads.box_predictor = FastRCNNPredictor(so_dac_trung, 2)
    if weights is not None:
        mo_hinh.initial_weight_provenance = {"path": cached, "sha256": file_sha256(cached),
                                            "source": str(enum)}
    return mo_hinh


def iou_hop(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    giao = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    dt_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    dt_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    hop_ = dt_a + dt_b - giao
    return giao / hop_ if hop_ > 0 else 0.0


@torch.no_grad()
def danh_gia(mo_hinh, loader, device, nguong=0.05):
    mo_hinh.eval()
    iou_tot, so_anh, trung, negatives, false_positive = [], 0, 0, 0, 0
    for anhs, muc_tieus, _ in loader:
        anhs = [a.to(device) for a in anhs]
        ra = mo_hinh(anhs)
        for r, mt in zip(ra, muc_tieus):
            that = mt["boxes"].tolist()
            du_doan = [b for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()) if s >= nguong]
            so_anh += 1
            if not that:
                negatives += 1
                false_positive += int(bool(du_doan))
                continue
            if not du_doan:
                iou_tot.append(0.0)
                continue
            tot = max(iou_hop(d, t) for d in du_doan[:5] for t in that)
            iou_tot.append(tot)
            trung += int(tot >= 0.5)
    return {"so_anh": so_anh, "so_anh_duong": len(iou_tot), "so_anh_am": negatives,
            "iou_tot_nhat_tb": round(float(np.mean(iou_tot)) if iou_tot else 0.0, 4),
            "ty_le_iou_tren_0_5": round(trung / max(len(iou_tot), 1), 4),
            "ty_le_khong_tim_thay": round(float(np.mean([i == 0.0 for i in iou_tot])) if iou_tot else 0.0, 4),
            "negative_specificity": round(1.0 - false_positive / max(negatives, 1), 4),
            "negative_false_positive_rate": round(false_positive / max(negatives, 1), 4)}


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện bộ phát hiện gãy xương trên FracAtlas")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="D:/BaoYen_work/checkpoints/bo_phat_hien_gay.pt")
    parser.add_argument("--bao_cao", default="reports/bo_phat_hien_gay.json")
    parser.add_argument("--split_manifest", help="JSON {train,val,test: [image_id,...]} từ clean split")
    parser.add_argument("--architecture", choices=["mobilenet_320", "resnet50_640"], default="mobilenet_320")
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_train_images", type=int)
    parser.add_argument("--max_eval_images", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--torch_threads", type=int, default=10)
    parser.add_argument("--progress", help="JSON tiến độ được cập nhật sau mỗi epoch")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Yêu cầu MPS nhưng torch.backends.mps.is_available() = False")
    logger.info("Detector backend: %s | torch threads: %d", device, torch.get_num_threads())
    split_ids, split_payload, split_manifest_sha256 = None, {}, None
    if args.split_manifest:
        split_payload, manifest_ids, split_manifest_sha256 = load_detector_manifest(args.split_manifest)
        split_ids = {key: set(value) for key, value in manifest_ids.items()}
        if split_payload.get("protocol_version", 0) >= 3 and (args.max_train_images or args.max_eval_images):
            raise RuntimeError("Use an explicit TRAIN-only smoke manifest, not silently capped v3 partitions")
    ids = lambda split: None if split_ids is None else split_ids[split]
    train_ds = BoPhatHienDataset("train", canh_dai=args.image_size, tang_cuong=True, allowed_ids=ids("train"))
    val_ds = BoPhatHienDataset("val", canh_dai=args.image_size, allowed_ids=ids("val"))
    test_ds = BoPhatHienDataset("test", canh_dai=args.image_size, allowed_ids=ids("test"))
    if split_payload.get("protocol_version", 0) >= 3:
        for role, dataset in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            verify_detector_annotation_parity(dataset, split_payload["source_vqa"][role]["path"])
    if args.max_train_images is not None:
        train_ds.muc = train_ds.muc[:args.max_train_images]
    if args.max_eval_images is not None:
        val_ds.muc = val_ds.muc[:args.max_eval_images]
        test_ds.muc = test_ds.muc[:args.max_eval_images]
    logger.info("Bộ phát hiện: train %d ảnh, val %d, test %d", len(train_ds), len(val_ds), len(test_ds))
    progress_path = args.progress or f"{args.out}.progress.json"
    save_json({"status": "running", "completed_epochs": 0, "requested_epochs": args.epochs,
               "train_images": len(train_ds), "val_images": len(val_ds), "test_images": len(test_ds),
               "device": str(device), "batch_size": args.batch_size}, progress_path)

    train_sampler = PositiveAnchoredBatchSampler(train_ds, args.batch_size, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                             num_workers=args.num_workers)

    mo_hinh = tao_mo_hinh(args.architecture).to(device)
    tham_so = [p for p in mo_hinh.parameters() if p.requires_grad]
    toi_uu = torch.optim.SGD(tham_so, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lich = torch.optim.lr_scheduler.CosineAnnealingLR(toi_uu, T_max=args.epochs)

    tot_nhat, lich_su, epochs_without_improvement = -1.0, [], 0
    for epoch in range(args.epochs):
        mo_hinh.train()
        t0, tong, n = time.time(), 0.0, 0
        for anhs, muc_tieus, _ in train_loader:
            anhs = [a.to(device) for a in anhs]
            muc_tieus = [{k: v.to(device) for k, v in t.items()} for t in muc_tieus]
            mat_mat = sum(mo_hinh(anhs, muc_tieus).values())
            if not torch.isfinite(mat_mat):
                raise RuntimeError("Non-finite detector loss; preserve artifacts and inspect")
            toi_uu.zero_grad(set_to_none=True)
            mat_mat.backward()
            torch.nn.utils.clip_grad_norm_(tham_so, 5.0)
            toi_uu.step()
            tong += float(mat_mat.detach())
            n += 1
        lich.step()
        m = danh_gia(mo_hinh, val_loader, device)
        m.update({"epoch": epoch + 1, "loss": round(tong / max(n, 1), 4), "time_s": round(time.time() - t0, 1)})
        lich_su.append(m)
        save_json({"status": "running", "completed_epochs": epoch + 1, "requested_epochs": args.epochs,
                   "best_val_iou": max(tot_nhat, m["iou_tot_nhat_tb"]), "latest": m,
                   "history": lich_su, "device": str(device), "batch_size": args.batch_size}, progress_path)
        logger.info("Epoch %d | loss %.4f | val IoU %.4f | IoU>0.5 %.1f%% | %.0fs",
                    epoch + 1, m["loss"], m["iou_tot_nhat_tb"], m["ty_le_iou_tren_0_5"] * 100, m["time_s"])
        if m["iou_tot_nhat_tb"] > tot_nhat:
            tot_nhat = m["iou_tot_nhat_tb"]
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({"state": mo_hinh.state_dict(), "canh_dai": args.image_size, "val": m,
                        "architecture": args.architecture, "includes_negative_images": True,
                        "training_split_sha256": split_manifest_sha256,
                        "training_scope": split_payload.get("scope"),
                        "training_image_ids": sorted(image for image, _ in train_ds.muc),
                        "initial_weight_provenance": getattr(mo_hinh, "initial_weight_provenance", None),
                        "preprocessing_version": RASTER_PREPROCESSING_VERSION}, args.out)
            logger.info("  đã lưu checkpoint tốt nhất")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stop sau %d epoch không cải thiện", args.patience)
                break

    tai = torch.load(args.out, map_location=device, weights_only=False)
    mo_hinh.load_state_dict(tai["state"])
    kq_test = danh_gia(mo_hinh, test_loader, device)
    logger.info("TEST: IoU tốt nhất TB %.4f | IoU>0.5 %.1f%% | không tìm thấy %.1f%%",
                kq_test["iou_tot_nhat_tb"], kq_test["ty_le_iou_tren_0_5"] * 100,
                kq_test["ty_le_khong_tim_thay"] * 100)
    save_json({"lich_su": lich_su, "test": kq_test, "checkpoint": args.out,
               "so_anh_train": len(train_ds), "includes_negative_images": True,
               "split_manifest": args.split_manifest,
               "preprocessing_version": RASTER_PREPROCESSING_VERSION}, args.bao_cao)
    save_json({"status": "complete", "completed_epochs": len(lich_su), "requested_epochs": args.epochs,
               "best_val_iou": tot_nhat, "test": kq_test, "checkpoint": args.out,
               "report": args.bao_cao, "device": str(device), "batch_size": args.batch_size}, progress_path)
    logger.info("Đã lưu %s", args.bao_cao)


if __name__ == "__main__":
    main()
