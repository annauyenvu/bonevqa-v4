"""BTXRD metadata -> templated clinical QA pairs, in the record format of the FracAtlas VQA files.

    venv/bin/python -I code/make_btxrd_vqa.py            # writes datasets/btxrd_vqa/
    venv/bin/python -I code/make_btxrd_vqa.py --out datasets/btxrd_vqa_v2

Mirrors refactor_v2/src/data/fracatlas_vqa.py (same record keys, same position words, same
number of sampled questions per image), with these BTXRD-specific rules:

* Split: the pilot split of prep_btxrd.py (datasets/btxrd_pilot/gt.json): centre 1 split
  train / val / test by pseudo-patient, centres 2 and 3 are ``external`` only.
* BTXRD fills the bone and joint columns ONLY for tumour images (all 1,879 normal images have
  none).  A question naming a bone would therefore reveal the tumour label, so bones and joints
  are used only in questions that already presuppose a lesion, and normal images are described
  by limb region (upper limb / lower limb / pelvis), which every image has.
* BTXRD has no fracture, hardware or fracture-count labels: those fields are None (unknown,
  not 0) and no fracture question is generated.
* Three endpoints are asked of every eligible image and tagged in ``canonical``:
    tumor       "Is there a bone tumor in this X-ray?"            every image
    malignancy  "Is the bone lesion malignant?"                   tumour images
    diagnosis   "What is the most likely diagnosis?"              tumour images with one subtype
  The last two are conditional on a lesion being present.
* Yes/no balance is kept PER QUESTION TEMPLATE, not globally as in FracAtlas.  The FracAtlas
  balancer offsets the "no"-heavy fracture question by picking "yes" variants of the optional
  questions, which makes e.g. "Is this a frontal view X-ray?" 96% yes in its TRAIN file: the
  template alone gives the answer.  Here each template that has a true and a false variant
  (limb, view, subtype, bone) alternates them; questions whose answer is fixed by the label
  (tumour, malignant, benign, multiple lesions) keep the label's real frequency.
* "Wrong" limbs, views, subtypes and bones are drawn in proportion to their TRAIN frequency, so
  that naming a frequent class is not by itself a cue for "yes".  Osteochondroma and multiple
  osteochondromas are never used as each other's wrong answer.
* Questions that put age and sex in the text carry ``clinical_context: true`` and can be
  filtered out.  Answers come only from the metadata of the same image.

Outputs: vqa_{train,val,test,external}.json, answer_vocab_train_only.json, stats.json (label
balance, question-only baselines, split checks).
"""
import json
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

ROOT = Path("/Users/uyen/Downloads/BONEVQA")
PILOT = ROOT / "yolo_pilot"
SRC = PILOT / "data_raw/btxrd/BTXRD"
GT = PILOT / "datasets/btxrd_pilot/gt.json"
KG = ROOT / "kg_rgo/out/bone_kg_draft_v1.json"
SEED = 20261116
ROLES = ("train", "val", "test", "external")
BONES = ["hand", "ulna", "radius", "humerus", "foot", "tibia", "fibula", "femur", "hip bone"]
JOINTS = ["ankle-joint", "knee-joint", "hip-joint", "wrist-joint", "elbow-joint", "shoulder-joint"]
LIMBS = ["upper limb", "lower limb", "pelvis"]
VIEWS = ["frontal", "lateral", "oblique"]
SUBTYPES = ["osteochondroma", "multiple osteochondromas", "simple bone cyst", "giant cell tumor", "osteofibroma",
            "synovial osteochondroma", "other bt", "osteosarcoma", "other mt"]
DIAGNOSIS = {s: s for s in SUBTYPES} | {"other bt": "other benign bone tumor", "other mt": "other malignant bone tumor"}
NAMED = [s for s in SUBTYPES if not s.startswith("other")]
NOT_EACH_OTHERS_NEGATIVE = {"osteochondroma", "multiple osteochondromas"}
NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

Q_TUMOR = "Is there a bone tumor in this X-ray?"
Q_MALIGNANT = "Is the bone lesion malignant?"
Q_DIAGNOSIS = "What is the most likely diagnosis?"


def position(box, width, height):  # identical to fracatlas_vqa._position
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


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def age_article(age):
    return "an" if age in (8, 11, 18) or 80 <= age < 90 else "a"


def join_words(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def normalize_answer(text):  # identical to refactor_v2/src/utils.normalize_answer
    text = (text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


class TemplateBalancer:
    """Alternates the yes / no variant separately for each question template."""

    def __init__(self):
        self.counts = defaultdict(Counter)

    def pick(self, name, variants, rng):
        if len(variants) == 1:
            answer = next(iter(variants))
        else:
            c = self.counts[name]
            answer = rng.choice(["yes", "no"]) if c["yes"] == c["no"] else ("yes" if c["yes"] < c["no"] else "no")
        self.counts[name][answer] += 1
        return variants[answer], answer


def weighted_choice(rng, weights):
    items = sorted(weights)
    return rng.choices(items, [weights[i] for i in items])[0] if items else None


def read_metadata():
    ws = openpyxl.load_workbook(SRC / "dataset.xlsx", read_only=True, data_only=True)["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    gt = json.loads(GT.read_text())
    kg_map = json.loads(KG.read_text())["label_map"]["btxrd"] if KG.exists() else {}
    infos = {}
    for r in rows[1:]:
        m = dict(zip(hdr, r))
        iid = "BT_" + str(m["image_id"]).rsplit(".", 1)[0]
        g = gt[iid]
        flagged = [s for s in SUBTYPES if m[s]]
        assert int(m["tumor"]) == g["tumor"] and sum(bool(m[k]) for k in LIMBS) == 1 and sum(bool(m[k]) for k in VIEWS) == 1, iid
        assert bool(flagged) == bool(m["tumor"]) and (m["tumor"] or not any(m[k] for k in BONES + JOINTS)), iid
        subtype = flagged[0] if len(flagged) == 1 else None
        age = str(m["age"]).strip()
        width, height = g["size"]
        ann = SRC / "Annotations" / f"{iid[3:]}.json"
        polygons = []
        if ann.exists():
            for s in json.loads(ann.read_text())["shapes"]:
                if s["shape_type"] == "polygon" and len(s["points"]) >= 3:
                    polygons.append([[float(v) for pt in s["points"] for v in pt[:2]]])
        boxes = [[float(v) for v in b[:4]] for b in g["boxes"]]
        infos[iid] = {
            "image": f"images/{m['image_id']}", "image_id": iid, "role": g["role"], "center": int(m["center"]), "group": g["group"],
            "tumor": int(m["tumor"]), "malignant": int(m["malignant"]) if m["tumor"] else None,
            "subtype": subtype, "subtypes_flagged": flagged, "diagnosis": DIAGNOSIS[subtype] if subtype else None,
            "kg_node": kg_map.get(subtype) if subtype else None,
            "limb": next(k for k in LIMBS if m[k]), "views": [k for k in VIEWS if m[k]],
            "bones": [k for k in BONES if m[k]], "joints": [k.replace("-", " ") for k in JOINTS if m[k]],
            "age": int(age) if age.isdigit() else None, "gender": {"M": "male", "F": "female"}.get(m["gender"]),
            "boxes": boxes or None, "box_labels": [b[4] for b in g["boxes"]] or None, "segmentation": polygons or None,
            "width": width, "height": height,
            "positions": list(dict.fromkeys(position(b, width, height) for b in boxes)),
        }
    assert set(infos) == set(gt)
    return infos


def site_text(info):
    """Where the lesion is, in words; falls back to the limb when no bone or joint is recorded."""
    return join_words(info["bones"] + info["joints"]) or info["limb"]


def closed_candidates(info, rng, prior):
    """Returns (fixed, slots).  A slot is (template name, {answer: question}); a slot with both a "yes" and a
    "no" question is balanced per template, a slot with one entry has its answer fixed by the label."""
    fixed = [(Q_TUMOR, "yes" if info["tumor"] else "no", "tumor")]
    wrong_limb = weighted_choice(rng, {k: w for k, w in prior["limb"].items() if k != info["limb"]})
    view = info["views"][0]
    wrong_view = weighted_choice(rng, {k: w for k, w in prior["view"].items() if k != view})
    slots = [("limb", {"yes": f"Is this an X-ray of the {info['limb']}?", "no": f"Is this an X-ray of the {wrong_limb}?"}),
             ("view", {"yes": f"Is this {article(view)} {view} view X-ray?", "no": f"Is this {article(wrong_view)} {wrong_view} view X-ray?"})]
    if info["tumor"]:
        fixed.append((Q_MALIGNANT, "yes" if info["malignant"] else "no", "malignancy"))
        slots.append(("benign", {"no" if info["malignant"] else "yes": "Is the bone lesion benign?"}))
        sub = info["subtype"]
        if sub is not None:
            wrong = {s: w for s, w in prior["subtype"].items()
                     if s != sub and not (s in NOT_EACH_OTHERS_NEGATIVE and sub in NOT_EACH_OTHERS_NEGATIVE)}
            variants = {"no": f"Is the diagnosis {weighted_choice(rng, wrong)}?"}
            if sub in NAMED:
                variants["yes"] = f"Is the diagnosis {sub}?"
            slots.append(("subtype", variants))
        if info["bones"]:
            wrong = {b: w for b, w in prior["bone"].items() if b not in info["bones"]}
            slots.append(("bone", {"yes": f"Is the lesion located in the {rng.choice(info['bones'])}?",
                                   "no": f"Is the lesion located in the {weighted_choice(rng, wrong)}?"}))
        if info["boxes"]:
            slots.append(("multiple", {"yes" if len(info["boxes"]) > 1 else "no": "Are there multiple bone lesions?"}))
    return fixed, slots


def open_candidates(info):
    """Returns (fixed, pool).  The first and the last pool items are the ones FracAtlas always samples from."""
    fixed = []
    pool = [("Which body region is shown in this X-ray?", info["limb"], False),
            ("What abnormality is seen in this X-ray?", "bone tumor" if info["tumor"] else "no abnormality", False),
            ("What type of imaging is this?", "x-ray", False)]
    if not info["tumor"]:
        pool.append(("How many bone lesions are visible?", "zero", False))
        pool.append(("Describe the finding.", f"No bone tumor is seen in the {info['limb']}.", False))
        return fixed, pool
    site = site_text(info)
    n = len(info["boxes"] or [])
    name = info["diagnosis"] or ("malignant bone tumor" if info["malignant"] else "benign bone tumor")
    if info["diagnosis"]:
        fixed.append((Q_DIAGNOSIS, info["diagnosis"], False, "diagnosis"))
        if info["age"] is not None and info["gender"]:
            pool.append((f"The patient is {age_article(info['age'])} {info['age']}-year-old {info['gender']}. {Q_DIAGNOSIS}",
                         info["diagnosis"], True))
    pool.append(("Is the bone lesion benign or malignant?", "malignant" if info["malignant"] else "benign", False))
    if info["bones"] or info["joints"]:
        pool.append(("Which bone or joint is affected by the lesion?", site, False))
    if n:
        pool.append(("How many bone lesions are visible?", NUMBER_WORDS.get(n, str(n)), False))
        pool.append(("Where is the lesion located?", f"{join_words(info['positions'])} of the {site}", False))
    plural = name.endswith("osteochondromas")
    desc = f"{name.capitalize()} are seen" if plural else f"{article(name).capitalize()} {name} is seen"
    desc += f" in the {site}"
    if info["positions"]:
        desc += f" at the {join_words(info['positions'])}"
    pool.append(("Describe the finding.", desc + ".", False))
    return fixed, pool


def generate_qa(info, rng, balancer, prior):
    fixed, slots = closed_candidates(info, rng, prior)
    out = []
    for q, a, tag in fixed:
        out.append({"question": q, "answer": a, "answer_type": "CLOSED", "canonical": tag, "clinical_context": False})
    rng.shuffle(slots)
    for name, variants in slots[: rng.randint(1, 2)]:
        q, a = balancer.pick(name, variants, rng)
        out.append({"question": q, "answer": a, "answer_type": "CLOSED", "canonical": None, "clinical_context": False})
    fixed_open, pool = open_candidates(info)
    for q, a, ctx, tag in fixed_open:
        out.append({"question": q, "answer": a, "answer_type": "OPEN", "canonical": tag, "clinical_context": ctx})
    n_open = rng.randint(1, min(3, len(pool)))
    must = [pool[0]] if rng.random() < 0.5 else [pool[-1]]
    rest = [x for x in pool if x not in must]
    rng.shuffle(rest)
    for q, a, ctx in must + rest[: max(0, n_open - 1)]:
        out.append({"question": q, "answer": a, "answer_type": "OPEN", "canonical": None, "clinical_context": ctx})
    assert len({x["question"] for x in out}) == len(out), info["image_id"]
    return out


def template(question):
    """Question with its slot values masked, for per-template statistics."""
    q = re.sub(r"^The patient is an? \d+-year-old (male|female)\. ", "[age, sex] ", question)
    for group, values in (("limb", LIMBS), ("subtype", SUBTYPES), ("bone", BONES), ("view", VIEWS)):
        for v in sorted(values, key=len, reverse=True):
            q = re.sub(rf"(?<=\s)(an? )?{re.escape(v)}(?=[ ?])", f"[{group}]", q)
    return q


def question_only_baseline(train, records):
    """Accuracy of answering from the question text alone (majority TRAIN answer for that exact string),
    next to the accuracy of always giving the TRAIN majority answer of the template."""
    by_q, by_t = defaultdict(Counter), defaultdict(Counter)
    for r in train:
        by_q[r["question"]][normalize_answer(r["answer"])] += 1
        by_t[template(r["question"])][normalize_answer(r["answer"])] += 1
    out = {}
    for t in sorted({template(r["question"]) for r in records}):
        rows = [r for r in records if template(r["question"]) == t]
        hit_q = sum(bool(by_q[r["question"]]) and by_q[r["question"]].most_common(1)[0][0] == normalize_answer(r["answer"]) for r in rows)
        hit_t = sum(bool(by_t[t]) and by_t[t].most_common(1)[0][0] == normalize_answer(r["answer"]) for r in rows)
        out[t] = {"n": len(rows), "answers": len({normalize_answer(r["answer"]) for r in rows}),
                  "acc_template_majority": round(hit_t / len(rows), 3), "acc_question_text_only": round(hit_q / len(rows), 3)}
    return out


def age_band(age):
    return None if age is None else min(age // 10, 6)


def diagnosis_without_image(train, records):
    """How far the diagnosis endpoint can be answered from metadata alone (TRAIN majority per key)."""
    rows_train = [r for r in train if r["canonical"] == "diagnosis"]
    rows = [r for r in records if r["canonical"] == "diagnosis"]
    keys = {"majority": lambda r: 0, "age_decade": lambda r: age_band(r["age"]),
            "age_decade_sex": lambda r: (age_band(r["age"]), r["gender"]),
            "age_decade_sex_limb": lambda r: (age_band(r["age"]), r["gender"], r["region"])}
    overall = Counter(r["answer"] for r in rows_train).most_common(1)[0][0]
    out = {"n": len(rows)}
    for name, key in keys.items():
        table = defaultdict(Counter)
        for r in rows_train:
            table[key(r)][r["answer"]] += 1
        hit = sum((table[key(r)].most_common(1)[0][0] if table[key(r)] else overall) == r["answer"] for r in rows)
        out[name] = round(hit / max(len(rows), 1), 3)
    return out


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=False)
    infos = read_metadata()
    by_role = {r: sorted(i for i, x in infos.items() if x["role"] == r) for r in ROLES}
    # Split checks: an image and a centre-1 pseudo-patient live in one role; external is centres 2-3 only.
    groups = defaultdict(set)
    for x in infos.values():
        groups[(x["center"], x["group"])].add(x["role"])
    assert all(len(v) == 1 for v in groups.values()), "a pseudo-patient spans two roles"
    assert all((infos[i]["center"] == 1) == (r != "external") for r in ROLES for i in by_role[r]), "centre / role mismatch"
    train = [infos[i] for i in by_role["train"]]
    prior = {"subtype": dict(Counter(x["subtype"] for x in train if x["subtype"] in NAMED)),
             "bone": dict(Counter(b for x in train for b in x["bones"])),
             "limb": dict(Counter(x["limb"] for x in train)), "view": dict(Counter(x["views"][0] for x in train))}
    meta_keys = ("image", "image_id", "role", "center", "group", "tumor", "malignant", "subtype", "diagnosis", "kg_node", "views",
                 "bones", "joints", "age", "gender", "boxes", "box_labels", "segmentation", "width", "height")
    records = {}
    for role in ROLES:
        balancer, rows = TemplateBalancer(), []
        for iid in by_role[role]:
            info = infos[iid]
            rng = random.Random(f"{SEED}:{iid}")  # per image, so a change of split never changes an image's questions
            for qa in generate_qa(info, rng, balancer, prior):
                rows.append({"image": info["image"], "image_id": iid, "dataset": "btxrd", **qa, "region": info["limb"],
                             "fractured": None, "hardware": None, "fracture_count": None,
                             **{k: info[k] for k in meta_keys if k not in ("image", "image_id")}})
        records[role] = rows
        (out_dir / f"vqa_{role}.json").write_text(json.dumps(rows, ensure_ascii=False))
    # No question may name a bone or joint for a normal image.
    words = re.compile("|".join(re.escape(w.replace("-", " ")) for w in BONES + JOINTS if w not in ("hand", "foot")))
    leaks = [r for rows in records.values() for r in rows if not r["tumor"] and words.search(r["question"] + " " + r["answer"])]
    assert not leaks, leaks[:3]
    vocab = Counter(normalize_answer(r["answer"]) for r in records["train"])
    (out_dir / "answer_vocab_train_only.json").write_text(json.dumps([a for a, _ in vocab.most_common()], ensure_ascii=False, indent=1))
    stats = {"seed": SEED, "source": str(SRC), "split": str(GT), "negative_sampling_prior_train": prior,
             "images_with_two_subtypes_no_diagnosis_question": sorted(i for i, x in infos.items() if len(x["subtypes_flagged"]) > 1),
             "roles": {}}
    for role in ROLES:
        rows = records[role]
        imgs = {r["image_id"]: r for r in rows}
        canon = {}
        for tag in ("tumor", "malignancy", "diagnosis"):
            c = Counter(r["answer"] for r in rows if r["canonical"] == tag)
            canon[tag] = dict(c.most_common())
        stats["roles"][role] = {
            "images": len(imgs), "tumor_images": sum(r["tumor"] for r in imgs.values()), "qa": len(rows),
            "qa_per_image": dict(sorted(Counter(Counter(r["image_id"] for r in rows).values()).items())),
            "by_type": dict(Counter(r["answer_type"] for r in rows)),
            "closed_yes_no": dict(Counter(r["answer"] for r in rows if r["answer_type"] == "CLOSED")),
            "clinical_context_rows": sum(r["clinical_context"] for r in rows), "canonical": canon,
            "open_answers_not_in_train_vocab": sum(normalize_answer(r["answer"]) not in vocab for r in rows if r["answer_type"] == "OPEN"),
            "diagnosis_without_image": diagnosis_without_image(records["train"], rows),
            "question_only_baseline": question_only_baseline(records["train"], rows)}
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    for role in ROLES:
        s = stats["roles"][role]
        print(f"{role:9s} images {s['images']:5d} (tumor {s['tumor_images']:4d})  QA {s['qa']:6d}  {s['by_type']}  yes/no {s['closed_yes_no']}  "
              f"context rows {s['clinical_context_rows']}  open answers unseen in train {s['open_answers_not_in_train_vocab']}")
    print("answer vocabulary (train):", len(vocab))
    for role in ROLES:
        s = stats["roles"][role]
        print(f"{role:9s} malignancy {s['canonical']['malignancy']}  diagnosis {s['canonical']['diagnosis']}")
        print(f"{'':9s} diagnosis answered without the image: {s['diagnosis_without_image']}")
    for role in ("val", "test"):
        print(f"\nquestion-only baselines on {role} (n, distinct answers, template-majority acc, exact-question-majority acc)")
        for t, v in sorted(stats["roles"][role]["question_only_baseline"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"  {v['n']:5d} {v['answers']:4d}  {v['acc_template_majority']:.3f}  {v['acc_question_text_only']:.3f}  {t}")


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else PILOT / "datasets/btxrd_vqa")
