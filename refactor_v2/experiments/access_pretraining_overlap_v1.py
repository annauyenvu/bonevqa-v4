"""R2.6 - provenance check: do FracAtlas radiographs appear in the ROCO / PMC-VQA image pools used
for the ROCO-aligned initialization?  Reference-only analysis (does not modify the project or the repo).

Checks, over the FULL local ROCO and PMC-VQA image pools (a superset of the filtered training lists):
  1. exact byte identity (SHA-256);
  2. perceptual near-duplicates: 64-bit difference hash (dHash) on grayscale, after centre-independent
     resize; candidates = Hamming distance <= THR (also reported for THR 0..10 as a sensitivity table);
  3. caption text: ROCO / PMC-VQA captions or questions that mention "FracAtlas".
Limitations: dHash is robust to resize / recompression / mild contrast change but not to strong crops,
panels inside multi-panel figures, or annotations drawn on the image.

Needs the FracAtlas, ROCO and PMC-VQA images (not in this repository); its output is stored in
reports/access_v1/review3/pretraining_overlap.json.
Usage: python experiments/access_pretraining_overlap_v1.py <project root> <out.json>
"""
import hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); DATA = ROOT / 'data'; THR = 6
PFX = 'BONE_VQA\\BaoYen_work\\data\\'


def pools():
    P = {'fracatlas': [], 'roco': [], 'pmc_vqa': []}
    for n in DATA.iterdir():
        s = n.name
        if not s.startswith(PFX) or not s.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        t = s[len(PFX):]
        if t.startswith('fracatlas\\FracAtlas\\images\\'): P['fracatlas'].append(n)
        elif t.startswith('roco\\images\\'): P['roco'].append(n)
        elif t.startswith('pmc_vqa\\images\\'): P['pmc_vqa'].append(n)
    return P


def sig(path):
    try:
        b = path.read_bytes(); h = hashlib.sha256(b).hexdigest()
        im = ImageOps.exif_transpose(Image.open(path)).convert('L')
        a = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        return str(path), h, int(np.packbits(bits).view('>u8')[0])
    except Exception as e:  # unreadable files are reported, not silently dropped
        return str(path), None, None


def popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555); x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F; return (x * 0x0101010101010101) >> 56


def main():
    P = pools(); S = {}
    with ProcessPoolExecutor(8) as ex:
        for k, v in P.items(): S[k] = list(ex.map(sig, v, chunksize=64))
    out = dict(n_images={k: len(v) for k, v in S.items()}, unreadable={k: sum(h is None for _, h, _ in v) for k, v in S.items()})
    fa = [(p, h, d) for p, h, d in S['fracatlas'] if h]
    fa_sha = {h for _, h, _ in fa}; fd = np.array([d for *_, d in fa], dtype=np.uint64)
    for pool in ('roco', 'pmc_vqa'):
        ok = [(p, h, d) for p, h, d in S[pool] if h]
        exact = [p for p, h, _ in ok if h in fa_sha]
        od = np.array([d for *_, d in ok], dtype=np.uint64); best = np.full(len(fa), 64); hist = np.zeros(65, int); cands = []
        for i in range(0, len(fa)):
            dist = popcount64(np.bitwise_xor(od, fd[i])).astype(int)
            j = int(dist.argmin()); best[i] = dist[j]
            for jj in np.flatnonzero(dist <= THR): cands.append(dict(fracatlas=Path(fa[i][0]).name.split('\\')[-1], other=Path(ok[jj][0]).name.split('\\')[-1], hamming=int(dist[jj])))
        hist = np.bincount(best, minlength=65)
        out[pool] = dict(n_checked=len(ok), exact_sha256_matches=len(exact), exact_examples=exact[:20],
                         fracatlas_images_with_nearest_hamming_le={t: int((best <= t).sum()) for t in range(0, 11)},
                         candidates_le_thr=cands, n_candidates_le_thr=len(cands), thr=THR)
    txt = {}
    for pool, files in (('roco', ['roco\\train.json', 'roco\\val.json', 'roco\\test.json']), ('pmc_vqa', ['pmc_vqa\\train.json', 'pmc_vqa\\test.json'])):
        hits = 0; n = 0
        for f in files:
            fp = DATA / (PFX + f)
            if not fp.exists(): continue
            for r in json.loads(fp.read_text()):
                n += 1; hits += 'fracatlas' in json.dumps(r, ensure_ascii=False).lower()
        txt[pool] = dict(rows=n, rows_mentioning_fracatlas=hits)
    out['caption_text_mentions'] = txt
    # stage 2: verify every dHash candidate by Pearson correlation of z-scored 64x64 grayscale thumbnails
    name2path = {Path(p).name.split('\\')[-1]: p for k in S for p, h, _ in S[k] if h}
    cache = {}
    def thumb(n):
        if n not in cache:
            im = ImageOps.exif_transpose(Image.open(name2path[n])).convert('L')
            a = np.asarray(im.resize((64, 64), Image.LANCZOS), float).ravel(); cache[n] = ((a - a.mean()) / (a.std() + 1e-6), im.size[0] / im.size[1])
        return cache[n]
    for pool in ('roco', 'pmc_vqa'):
        rs = []
        for c in out[pool]['candidates_le_thr']:
            (a, ra), (b, rb) = thumb(c['fracatlas']), thumb(c['other']); c['r64'] = float(a @ b / len(a)); c['aspect_ratio_diff'] = abs(ra - rb); rs.append(c['r64'])
        rs = np.array(rs) if rs else np.zeros(0)
        out[pool]['stage2'] = dict(n_pairs=len(rs), max_r64=float(rs.max()) if len(rs) else None,
                                   pairs_r64_ge={t: int((rs >= t).sum()) for t in (0.8, 0.9, 0.95, 0.98)},
                                   fracatlas_images_with_pair_r64_ge_0_9=len({c['fracatlas'] for c in out[pool]['candidates_le_thr'] if c['r64'] >= 0.9}),
                                   top10=sorted(out[pool]['candidates_le_thr'], key=lambda c: -c['r64'])[:10])
        out[pool]['candidates_le_thr'] = [c for c in out[pool]['candidates_le_thr'] if c['r64'] >= 0.8]
    OUT.write_text(json.dumps(out, indent=1)); print(json.dumps({k: v for k, v in out.items() if k not in ('roco', 'pmc_vqa')} | {k: {kk: vv for kk, vv in out[k].items() if kk not in ('candidates_le_thr', 'exact_examples')} for k in ('roco', 'pmc_vqa')}, indent=1))


if __name__ == '__main__':
    main()
