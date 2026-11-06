"""Baseline chỉ dùng ảnh trên cùng các fold (reports/access_v1/PROTOCOL.md). Học trên TRAIN của từng fold, không chọn gì trên test.
Chạy: python -m experiments.access_image_baselines_v1 clip | mobilenet <fold>
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from src.data.fracatlas_vqa import FRACATLAS_ROOT
from src.preprocessing import canonicalize_raster
R = Path('/Users/uyen/Downloads/BONEVQA/refactor_v2/reports'); SEALED = R / 'local_noaux_refinement_v1'; OUT = R / 'access_v1/image_baselines'; OUT.mkdir(parents=True, exist_ok=True)
CAN = 'Is there a fracture in this X-ray?'; meta = json.load(open(SEALED / 'metadata.json'))


def role(fold, name):
    return [r for r in json.load(open(meta[fold]['roles'][name]['path'])) if r['question'] == CAN]


def load(rel): return canonicalize_raster(Image.open(FRACATLAS_ROOT / rel)).convert('RGB')


def clip():
    """Mã hóa một lần bằng BiomedCLIP đóng băng, rồi hồi quy logistic theo từng fold."""
    from sklearn.linear_model import LogisticRegression
    torch.set_num_threads(2); cache = OUT / 'biomedclip_cls.npz'; recs = {}
    for f in meta:
        for n in ('train', 'test'):
            for r in role(f, n): recs[r['image_id']] = r['image']
    ids = sorted(recs)
    if not cache.exists():
        import open_clip
        model, _, prep = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'); model.eval(); F = []
        for k in range(0, len(ids), 16):
            with torch.no_grad(): F.append(model.encode_image(torch.stack([prep(load(recs[i])) for i in ids[k:k + 16]]), normalize=True).numpy())
            print(k, len(ids), flush=True)
        np.savez(cache, ids=np.array(ids), F=np.concatenate(F))
    z = np.load(cache); E = dict(zip(z['ids'].tolist(), z['F'])); out = {}
    for f in meta:
        tr, te = role(f, 'train'), role(f, 'test')
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(np.stack([E[r['image_id']] for r in tr]), [r['answer'] == 'yes' for r in tr])
        p = clf.predict_proba(np.stack([E[r['image_id']] for r in te]))[:, 1]
        out[f] = {r['image_id']: float(x) for r, x in zip(te, p)}
    (OUT / 'biomedclip_lr.json').write_text(json.dumps(out))


def mobilenet(fold):
    from torchvision import models, transforms as T
    dest = OUT / f'mobilenet_fold{fold}.json'
    if dest.exists(): return
    torch.set_num_threads(2); torch.manual_seed(20261103); np.random.seed(20261103)
    norm = T.Normalize([.485, .456, .406], [.229, .224, .225])
    aug = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.ColorJitter(.1, .1), T.ToTensor(), norm]); plain = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])
    m = models.mobilenet_v3_large(weights='IMAGENET1K_V1'); m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 1)
    tr, te = role(fold, 'train'), role(fold, 'test'); y = np.array([r['answer'] == 'yes' for r in tr], np.float32); pw = torch.tensor((1 - y.mean()) / y.mean())
    X = [load(r['image']).resize((256, 256)) for r in tr]; epochs = 10; bs = 16; opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=.01); start = time.time()
    for e in range(epochs):
        m.train(); perm = np.random.permutation(len(tr))
        for k in range(0, len(perm), bs):
            b = perm[k:k + bs]; loss = torch.nn.functional.binary_cross_entropy_with_logits(m(torch.stack([aug(X[i]) for i in b])).squeeze(1), torch.tensor(y[b]), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        print(fold, e, round(float(loss), 3), round(time.time() - start), flush=True)
    m.eval(); out = {}
    for k in range(0, len(te), 32):
        b = te[k:k + 32]
        with torch.no_grad(): p = torch.sigmoid(m(torch.stack([plain(load(r['image'])) for r in b]))).squeeze(1).tolist()
        out.update({r['image_id']: float(x) for r, x in zip(b, p)})
    dest.write_text(json.dumps(dict(seconds=time.time() - start, epochs=epochs, p_yes=out)))


if __name__ == '__main__':
    clip() if sys.argv[1] == 'clip' else mobilenet(sys.argv[2])
