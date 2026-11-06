"""Bổ sung cho luận văn: kiểm tra phụ thuộc đầu vào và cơ chế của B/V4 trên checkpoint đã niêm phong.

Chỉ đọc checkpoint và boxes đã lưu, không huấn luyện lại, không chọn lại ngưỡng, không ghi vào reports đã niêm phong.
Mỗi điều kiện lưu p_yes của câu hỏi canonical theo ảnh vào reports/thesis_v4_checks_v1/fold_{k}/{arm}/{cond}.json.
Chạy: python -m experiments.thesis_v4_checks_v1 --fold 0 --arm V4 [--conds a,b] [--threads 2] [--limit N]
"""
import argparse, io, json, os, random, time
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageFilter, ImageOps
ROOT = Path('/Users/uyen/Downloads/BONEVQA'); R2 = ROOT / 'refactor_v2'; SEALED = R2 / 'reports/local_noaux_refinement_v1'; OUT = R2 / 'reports/thesis_v4_checks_v1'
CAN = 'Is there a fracture in this X-ray?'; SEED = 20261201
ARMDIR = {'B': 'B', 'V4': 'local_real_noaux'}
PRIOR = ['blank', 'shuffle']
INPUT = ['down1024', 'down512', 'down256', 'jpeg30', 'blur2', 'noise10', 'hflip', 'rot90', 'crop80', 'invert']
MECH = ['delta_off', 'qcond_off', 'noise_crops', 'placebo_boxes', 'no_proposals', 'grid_always']
QUAL = ['iou100', 'iou075', 'iou050', 'iou025', 'iou000']
LEAK = ['annotation_leak', 'annotation_all_pos_detector_neg']
CONDS = {'B': ['reference'] + PRIOR + INPUT, 'V4': ['reference'] + MECH + QUAL + LEAK + PRIOR + INPUT}

def read(p): return json.loads(Path(p).read_text())
def load_detector(fold):
    from experiments.detector640_checkpoint_v3 import load_bound_640
    from src.detector_runtime import FractureDetectorRuntime
    rec = read(SEALED / f'fold_{fold}/detector/receipt.json')
    d = object.__new__(FractureDetectorRuntime); d.model, _ = load_bound_640(rec['checkpoint'])
    d.device, d.image_size, d.score_threshold, d.max_boxes = torch.device('cpu'), 640, 0.05, 3
    return d
def transform(im, cond, rng):
    w, h = im.size
    if cond == 'blank': return Image.new('RGB', im.size, (0, 0, 0))
    if cond.startswith('down'):
        s = int(cond[4:]) / max(w, h)
        return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR) if s < 1 else im
    if cond == 'jpeg30':
        b = io.BytesIO(); im.convert('RGB').save(b, 'JPEG', quality=30); b.seek(0); return Image.open(b).convert('RGB')
    if cond == 'blur2': return im.filter(ImageFilter.GaussianBlur(2))
    if cond == 'noise10':
        a = np.asarray(im.convert('RGB'), dtype=np.float32); n = rng.normal(0, 10, a.shape[:2])[..., None]
        return Image.fromarray(np.clip(a + n, 0, 255).astype(np.uint8))
    if cond == 'hflip': return ImageOps.mirror(im)
    if cond == 'rot90': return im.rotate(90, expand=True)
    if cond == 'crop80':
        dx, dy = int(w * .1), int(h * .1); return im.crop((dx, dy, w - dx, h - dy))
    if cond == 'invert': return ImageOps.invert(im.convert('RGB'))
    return im
def shift_to_iou(box, t, size, rng):
    """Dịch box theo một trục sao cho IoU với box gốc bằng t; t=0 là ngay sát, không giao."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]; w, h = x2 - x1, y2 - y1; W, H = size
    if t >= 1: return [x1, y1, x2, y2]
    axis = rng.integers(2); side = w if axis == 0 else h
    d = side * (1 - t) / (1 + t) if t > 0 else side * 1.05
    sign = rng.choice([-1, 1])
    for s in (sign, -sign):
        nx1, ny1 = (x1 + s * d, y1) if axis == 0 else (x1, y1 + s * d)
        if nx1 >= 0 and ny1 >= 0 and nx1 + w <= W and ny1 + h <= H: return [nx1, ny1, nx1 + w, ny1 + h]
    nx1, ny1 = (max(0, min(W - w, x1 + sign * d)), y1) if axis == 0 else (x1, max(0, min(H - h, y1 + sign * d)))
    return [nx1, ny1, nx1 + w, ny1 + h]
def install_hooks(model, cond, rng_t):
    lt = getattr(model, 'local_tokens', None)
    if lt is None: return
    if cond == 'delta_off': model.cfg['vnext_force_zero'] = True
    if cond == 'qcond_off':
        f = lt.question.forward; lt.question.forward = lambda x: torch.zeros_like(f(x))
    if cond == 'noise_crops':
        f = lt.forward
        def g(global_tokens, text, text_mask, crops, valid):
            v = crops[valid.unsqueeze(-1).expand(-1, -1, crops.shape[2])] if valid.any() else None
            if v is not None and len(v) > 1:
                m, s = v.mean(0), v.std(0)
                crops = torch.where(valid[:, :, None, None], m + s * torch.randn(crops.shape, generator=rng_t), crops)
            return f(global_tokens, text, text_mask, crops, valid)
        lt.forward = g
def main():
    a = argparse.ArgumentParser(); a.add_argument('--fold', required=True); a.add_argument('--arm', required=True); a.add_argument('--conds', default=''); a.add_argument('--threads', type=int, default=2); a.add_argument('--limit', type=int); a.add_argument('--bs', type=int, default=8); a.add_argument('--device', default='cpu')
    x = a.parse_args(); torch.set_num_threads(x.threads)
    from experiments.vnext_model_v1 import VNextModel
    from src.data.fracatlas_vqa import FracAtlasVQADataset as DS
    from src.preprocessing import canonicalize_raster
    folder = SEALED / f'fold_{x.fold}'; meta = read(SEALED / 'metadata.json')[x.fold]; ds = DS(records_path=meta['roles']['test']['path'], with_masks=False, with_boxes=False)
    rows = [i for i, r in enumerate(ds.records) if r['question'] == CAN]
    if x.limit: rows = rows[:x.limit]
    saved = {r['image_id']: r['p_yes'] for r in read(folder / ARMDIR[x.arm] / 'test_predictions.json')['records'] if r['question'] == CAN}
    boxes = read(folder / 'boxes.json'); ids = [ds.records[i]['image_id'] for i in rows]
    perm = list(range(len(rows))); random.Random(SEED + int(x.fold)).shuffle(perm)
    while any(p == i for i, p in enumerate(perm)): random.shuffle(perm)
    detector = None; out_dir = OUT / f'fold_{x.fold}/{x.arm}'; out_dir.mkdir(parents=True, exist_ok=True)
    conds = x.conds.split(',') if x.conds else CONDS[x.arm]
    for cond in conds:
        dest = out_dir / f'{cond}.json'
        if dest.exists() and not x.limit: continue
        model = VNextModel.load(str(folder / ARMDIR[x.arm] / 'LAST.pt'), device=x.device).eval()
        rng = np.random.default_rng(SEED + int(x.fold)); rng_t = torch.Generator().manual_seed(SEED + int(x.fold))
        install_hooks(model, cond, rng_t)
        need_det = x.arm == 'V4' and (cond in PRIOR or cond in INPUT)
        if need_det and detector is None: detector = load_detector(x.fold)
        res = {}; proposals = {}; start = time.time()
        for off in range(0, len(rows), x.bs):
            idx = list(range(off, min(off + x.bs, len(rows)))); ims = []; bx = []
            for j in idx:
                rec = ds.records[rows[j]]; im = ds[rows[j]]['image']; iid = rec['image_id']
                if cond == 'shuffle':
                    im = ds[rows[perm[j]]]['image']
                im2 = transform(im, cond, rng)
                if x.arm == 'B': ims.append(im2); bx.append([]); continue
                pos = rec['answer'] == 'yes'; gt = [b[:4] for b in (rec.get('boxes') or [])]; det = boxes[iid]
                if need_det: b = detector.predict(canonicalize_raster(im2)) or []
                elif cond in ('no_proposals', 'delta_off') : b = [] if cond == 'no_proposals' else det
                elif cond == 'placebo_boxes': b = DS._matched_placebo_boxes(det, im2.size) or [] if det else []
                elif cond == 'grid_always':
                    W, H = im2.size; b = [[W * k / 3, H * .25, W * (k + 1) / 3, H * .75] for k in range(3)]
                elif cond in QUAL:
                    t = int(cond[3:]) / 100
                    b = [shift_to_iou(gt[k % len(gt)], t, im2.size, rng) for k in range(len(det))] if (pos and gt and det) else det
                elif cond == 'annotation_leak': b = gt[:3] if pos else []
                elif cond == 'annotation_all_pos_detector_neg': b = gt[:3] if pos and gt else det
                else: b = det
                proposals[iid] = len(b); ims.append(im2); bx.append(b)
            with torch.no_grad():
                if x.arm == 'B': out = model.predict_batch({'images': ims, 'questions': [CAN] * len(ims)})
                else: out = model.predict_batch({'images': ims, 'questions': [CAN] * len(ims), 'boxes': [b or None for b in bx], 'masks': [DS._boxes_to_masks(b, i.size) if b else [] for b, i in zip(bx, ims)]})
            for j, p in zip(idx, out): res[ids[j]] = float(p['p_yes'])
            if cond == 'reference':
                for j in idx: assert abs(res[ids[j]] - saved[ids[j]]) < 1e-4, (ids[j], res[ids[j]], saved[ids[j]])
            print(cond, off, len(rows), round(time.time() - start), flush=True)
        tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps({'cond': cond, 'fold': x.fold, 'arm': x.arm, 'seconds': time.time() - start, 'n': len(res), 'p_yes': res, 'n_boxes': proposals}))
        if not x.limit: tmp.rename(dest)
        del model
if __name__ == '__main__': main()
