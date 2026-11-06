"""Câu hỏi trơ và tráo ảnh trên mẫu 800 QA của tập test mỗi fold (mặc định fold 0), gồm cả nhánh sinh câu trả lời mở. Không huấn luyện lại."""
import json, random, sys, time
from pathlib import Path
import torch
from PIL import Image
from experiments.thesis_v4_checks_v1 import read, SEALED, OUT, ARMDIR, load_detector
from experiments.vnext_model_v1 import VNextModel
from src.data.fracatlas_vqa import FracAtlasVQADataset as DS
from src.preprocessing import canonicalize_raster
arm, cond = sys.argv[1], sys.argv[2]; fold = sys.argv[3] if len(sys.argv) > 3 else '0'; torch.set_num_threads(2)
folder = SEALED / f'fold_{fold}'; ds = DS(records_path=read(SEALED / 'metadata.json')[fold]['roles']['test']['path'], with_masks=False, with_boxes=False)
idx = sorted(random.Random(20261201).sample(range(len(ds.records)), 800))
ids = sorted({r['image_id'] for r in ds.records}); perm = ids[:]; rng = random.Random(20261202); rng.shuffle(perm)
while any(a == b for a, b in zip(ids, perm)): rng.shuffle(perm)
donor = dict(zip(ids, perm)); first = {}
for i, r in enumerate(ds.records): first.setdefault(r['image_id'], i)
m = VNextModel.load(str(folder / ARMDIR[arm] / 'LAST.pt'), device='cpu').eval(); det = load_detector(fold) if arm == 'V4' else None
rows = []; t = time.time()
for off in range(0, len(idx), 4):
    ims = []; bx = []; qs = []
    for i in idx[off:off + 4]:
        r = ds.records[i]; im = ds[i]['image'] if cond != 'shuffle' else ds[first[donor[r['image_id']]]]['image']
        if cond == 'blank': im = Image.new('RGB', im.size, (0, 0, 0))
        ims.append(im); qs.append(r['question']); bx.append((det.predict(canonicalize_raster(im)) or []) if det else [])
    with torch.no_grad():
        out = m.predict_batch({'images': ims, 'questions': qs}) if arm == 'B' else m.predict_batch({'images': ims, 'questions': qs, 'boxes': [b or None for b in bx], 'masks': [DS._boxes_to_masks(b, i.size) if b else [] for b, i in zip(bx, ims)]})
    for i, p in zip(idx[off:off + 4], out):
        r = ds.records[i]; rows.append(dict(image_id=r['image_id'], question=r['question'], gt=r['answer'], answer_type=r['answer_type'], pred=p['answer']))
    print(off, round(time.time() - t), flush=True)
suffix = '' if fold == '0' else f'_fold{fold}'
(OUT / f'qa_sample_{arm}_{cond}{suffix}.json').write_text(json.dumps(dict(fold=fold, seconds=time.time() - t, rows=rows), ensure_ascii=False))
