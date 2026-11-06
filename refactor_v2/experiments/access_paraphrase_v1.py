"""Diễn đạt lại câu hỏi gãy xương trên checkpoint niêm phong B và V4 (reports/access_v1/PROTOCOL.md). Không huấn luyện lại.
Chạy: python -m experiments.access_paraphrase_v1 <fold> <arm> <index> ; ghi reports/access_v1/paraphrase/fold_k/arm/qN.json
"""
import json, sys, time
from pathlib import Path
import torch
from experiments.thesis_v4_checks_v1 import read, SEALED, ARMDIR, CAN
from experiments.vnext_model_v1 import VNextModel
from src.data.fracatlas_vqa import FracAtlasVQADataset as DS

QUESTIONS = ['Does this X-ray show a fracture?', 'Is a fracture present in this image?', 'Is any bone broken in this radiograph?', 'Can a fracture be seen on this film?', 'Is this X-ray positive for fracture?', 'Is there no fracture in this X-ray?']
OUT = SEALED.parent / 'access_v1/paraphrase'


def main():
    fold, arm, i = sys.argv[1], sys.argv[2], int(sys.argv[3]); q = QUESTIONS[i]; torch.set_num_threads(2)
    dest = OUT / f'fold_{fold}/{arm}/q{i}.json'
    if dest.exists(): return
    dest.parent.mkdir(parents=True, exist_ok=True)
    folder = SEALED / f'fold_{fold}'; ds = DS(records_path=read(SEALED / 'metadata.json')[fold]['roles']['test']['path'], with_masks=False, with_boxes=False)
    rows = [k for k, r in enumerate(ds.records) if r['question'] == CAN]; boxes = read(folder / 'boxes.json')
    model = VNextModel.load(str(folder / ARMDIR[arm] / 'LAST.pt'), device='cpu').eval(); res = {}; start = time.time()
    for off in range(0, len(rows), 8):
        idx = rows[off:off + 8]; ims = [ds[k]['image'] for k in idx]; ids = [ds.records[k]['image_id'] for k in idx]; qs = [q] * len(idx)
        with torch.no_grad():
            if arm == 'B': out = model.predict_batch({'images': ims, 'questions': qs})
            else:
                bx = [boxes[a] for a in ids]
                out = model.predict_batch({'images': ims, 'questions': qs, 'boxes': [b or None for b in bx], 'masks': [DS._boxes_to_masks(b, im.size) if b else [] for b, im in zip(bx, ims)]})
        for a, p in zip(ids, out): res[a] = dict(p_yes=float(p['p_yes']) if 'p_yes' in p else None, answer=p['answer'])
        print(off, len(rows), round(time.time() - start), flush=True)
    tmp = dest.with_suffix('.tmp'); tmp.write_text(json.dumps(dict(question=q, fold=fold, arm=arm, seconds=time.time() - start, rows=res), ensure_ascii=False)); tmp.rename(dest)


if __name__ == '__main__': main()
