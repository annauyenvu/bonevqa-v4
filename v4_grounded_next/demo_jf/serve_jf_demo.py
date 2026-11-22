"""Demo API of the final system JF: two-class YOLOv8 + V4 (joint FracAtlas + BTXRD) + trust layer.

    v4_grounded_next/demo_jf/run_jf_demo.sh          # http://127.0.0.1:8021

Same model, proposals and trust layer as the evaluation (scripts.eval_arm / eval_joint / answer_trust):
  * detector: outer YOLOv8s of the fold (yolo_pilot/runs/yolo/fa_btxrd_os3[_f<k>]), run in a worker process with the
    yolo venv, same predict call as yolo_pilot/code/predict_yolo.py;
  * proposals for V4: top-3 boxes of the routed class with score >= the JF level threshold of the fold (fracture class at
    level 0.02, lesion class at the locked BTXRD level), as in the proposal files;
  * displayed boxes: both classes at the detector's own operating point (max box F1@0.5 on model_val, summary_final);
  * trust layer: runs/joint/fold_<k>/trust_JF_*.json (Platt on |p - 0.5| and the top box score, ~80% coverage).
The fold is BONEVQA_DEMO_FOLD (default 0); its checkpoint was never trained on that fold's test images, which the
example images are drawn from.  Question routing: a question about a tumour / lesion / diagnosis gets lesion-class boxes,
any other question fracture-class boxes (in training FracAtlas rows carried fracture boxes, BTXRD rows lesion boxes).
"""
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.ngon_ngu import bo_dau, la_tieng_viet, sang_tieng_anh, sang_tieng_viet
from scripts.reference_score import candidate_loglik
from src.preprocessing import canonicalize_raster
from vg import kg as K
from vg import trust as T
from vg.paths import RUNS

ROOT = Path("/Users/uyen/Downloads/BONEVQA"); PILOT = ROOT / "yolo_pilot"
HERE = Path(__file__).resolve().parent; WEB = HERE / "web"
FEEDBACK = HERE / "feedback.jsonl"
FOLD = int(os.environ.get("BONEVQA_DEMO_FOLD", "0"))
NAME = "JF_v4cf_yolojoint_btxrd_r0.02"
SW = RUNS / "crossfit_yolo_joint_sweep"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
CLASS_VI = {0: "gãy xương", 1: "tổn thương xương"}
DX_VI = {"osteochondroma": "u xương sụn", "osteosarcoma": "sarcoma xương", "simple bone cyst": "nang xương đơn độc",
         "multiple osteochondromas": "đa u xương sụn", "giant cell tumor": "u tế bào khổng lồ", "osteofibroma": "u xơ xương",
         "synovial osteochondroma": "u sụn màng hoạt dịch", "other benign bone tumor": "u lành khác",
         "other malignant bone tumor": "u ác khác"}
LESION_WORDS = re.compile(r"tumou?r|lesion|malignan|benign|diagnos|osteo|cyst|giant cell", re.I)
VI_TUMOUR = [  # Vietnamese -> BTXRD templates, checked before the FracAtlas layer (bo_dau form)
    (r"chan doan|benh gi|la benh", T.DIAGNOSIS_Q),
    (r"ac tinh|lanh hay ac|lanh tinh hay ac", "Is the bone lesion malignant?"),
    (r"bao nhieu (ton thuong|u|o)", "How many bone lesions are visible?"),
    (r"(ton thuong|khoi u|u) (nam )?o (dau|vi tri nao)|vi tri (ton thuong|khoi u)", "Where is the lesion located?"),
    (r"\bu xuong\b|\bkhoi u\b|\bco u\b", "Is there a bone tumor in this X-ray?"),
]
torch.set_num_threads(int(os.environ.get("BONEVQA_DEMO_THREADS", "4")))


def load_system():
    layer = T.TrustLayer.load(RUNS / "joint" / f"fold_{FOLD}" / f"trust_{NAME}.json")
    model, _, sha = K.load_kg_arm(layer.spec["tag"], str(FOLD), "cpu")
    assert sha == layer.spec["checkpoint_sha256"], "trust layer fitted for another checkpoint"
    model.eval()
    level = json.loads((RUNS / "joint/final_btxrd_level.json").read_text())["chosen_level"]
    v4 = {0: json.loads((SW / "r0.02" / f"fold_{FOLD}" / f"proposals_fold{FOLD}.json").read_text())["thresholds"]["outer"],
          1: json.loads((SW / f"r{level}" / f"btxrd_fold{FOLD}" / "proposals_btxrd_insample.json").read_text())["thresholds"]["outer"]}
    det = json.loads((PILOT / "runs/summary_final/summary.json").read_text())["detector"][str(FOLD)]
    own = {0: det["fracatlas"]["own_best_threshold"], 1: det["btxrd"]["own_best_threshold"]}
    weights = PILOT / "runs/yolo" / ("fa_btxrd_os3" if FOLD == 0 else f"fa_btxrd_os3_f{FOLD}") / "weights/best.pt"
    worker = subprocess.Popen([str(PILOT / "venv/bin/python"), "-I", str(HERE / "yolo_worker.py"), str(weights)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert json.loads(worker.stdout.readline())["ready"]
    evaluated = set()
    for role in ("test", "external"):
        evaluated |= {r["question"] for r in json.loads((PILOT / f"datasets/btxrd_vqa/vqa_{role}.json").read_text())}
    sealed = ROOT / "refactor_v2/reports/local_noaux_refinement_v1"
    for k in range(5):
        evaluated |= {r["question"] for r in json.loads((sealed / f"fold_{k}/local_real_noaux/test_predictions.json").read_text())["records"]}
    return model, layer, sha, v4, own, level, worker, weights, evaluated


MODEL, LAYER, SHA, V4_THR, OWN_THR, BT_LEVEL, WORKER, WEIGHTS, EVALUATED = load_system()
LOCK = threading.Lock()
app = FastAPI(title="BoneVQA JF demo")
app.mount("/static", StaticFiles(directory=WEB), name="static")


def read_image(upload):
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Ảnh vượt quá 20 MB.")
    try:
        image = Image.open(io.BytesIO(data)); image.load()
        return canonicalize_raster(image)
    except Exception as exc:
        raise HTTPException(400, "Không đọc được tệp ảnh.") from exc


def detect(image):
    """All boxes per class (score >= 0.001), in canonical-image pixels."""
    with tempfile.NamedTemporaryFile(suffix=".png", dir=ROOT / "v4_grounded_next/tmp", delete=True) as f:
        image.save(f.name)
        WORKER.stdin.write(json.dumps({"path": f.name}) + "\n")
        out = json.loads(WORKER.stdout.readline())
    return {int(k): v for k, v in out["boxes"].items()}, out["ms"]


def shown(boxes):
    """Boxes displayed to the reader: both classes at the detector's own operating point."""
    return [{"box": b[:4], "score": b[4], "cls": c, "label": CLASS_VI[c]} for c in (0, 1) for b in boxes[c] if b[4] >= OWN_THR[c]]


def to_english(question):
    q = question.strip()
    if la_tieng_viet(q):
        plain = bo_dau(q).lower()
        for pat, tpl in VI_TUMOUR:
            if re.search(pat, plain):
                return tpl, True
    return sang_tieng_anh(q)


def to_vietnamese(answer):
    a = answer.strip()
    return DX_VI.get(a.lower(), None) or sang_tieng_viet(a)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"status": "ok", "model": f"JF · fold {FOLD}", "checkpoint_sha256": SHA[:12], "detector": str(WEIGHTS.relative_to(ROOT)),
            "device": f"cpu ({torch.get_num_threads()} luồng)", "v4_thresholds": V4_THR, "display_thresholds": OWN_THR,
            "btxrd_level": BT_LEVEL}


@app.get("/api/info")
def info():
    return json.loads((WEB / "info.json").read_text(encoding="utf-8"))


@app.get("/api/examples")
def examples():
    return {"examples": [{"name": p.stem, "url": f"/static/examples/{p.name}"} for p in sorted((WEB / "examples").glob("*.jpg"))]}


@app.post("/api/detect")
def propose(image: UploadFile = File(...)):
    im = read_image(image)
    start = time.perf_counter()
    with LOCK:
        boxes, _ = detect(im)
    return {"regions": shown(boxes), "width": im.width, "height": im.height, "latency_ms": round((time.perf_counter() - start) * 1000)}


@app.post("/api/ask")
def ask(image: UploadFile = File(...), question: str = Form(...), shortlist: bool = Form(True)):
    q = question.strip()
    if not q or len(q) > 500:
        raise HTTPException(400, "Câu hỏi cần dài từ 1 đến 500 ký tự.")
    im = read_image(image)
    english, translated = to_english(q)
    cls = 1 if LESION_WORDS.search(english) else 0
    start = time.perf_counter()
    with LOCK, torch.no_grad():
        boxes, det_ms = detect(im)
        props = [b for b in boxes[cls] if b[4] >= V4_THR[cls]][:3]
        batch = {"images": [im], "questions": [english], "proposals": [props], "proposal_source": ["detector_full"],
                 "image_ids": ["upload"], "component_ids": None}
        res = MODEL.infer(batch)
        pr = torch.softmax(res["logits_closed"], dim=-1)
        yes, no = MODEL.answer_to_idx["yes"], MODEL.answer_to_idx["no"]
        score = T.top_box_score(props)
        if english in T.ENDPOINT_OF:
            ep = T.ENDPOINT_OF[english]
            p = float(res["p_final"][0]) if ep == "fracture" else float(pr[0, yes] / (pr[0, yes] + pr[0, no]).clamp_min(1e-9))
            v, kind = LAYER.closed(ep, p, score, 0.8, "head_detector"), "CLOSED"
        elif english == T.DIAGNOSIS_Q and shortlist:
            sc = candidate_loglik(MODEL, res["core"], res["core"]["q_texts"], LAYER.dx["candidates"])
            gen = MODEL.generate_open(res["core"], res["core"]["q_texts"])
            v, kind = LAYER.diagnosis(sc[0].numpy(), 0.8, generated=gen[0]["text"]), "OPEN"
        elif re.match(r"^(is|are|does|do|has|have|can|was|were)\b", english, re.I) and " or " not in english.lower():
            c = int(pr[0].argmax())
            v, kind = LAYER.uncalibrated(MODEL.answer_vocab[c], float(pr[0, c])), "CLOSED"
        else:
            g = MODEL.generate_open(res["core"], res["core"]["q_texts"])[0]
            v, kind = LAYER.uncalibrated(g["text"], float(g["confidence"])), "OPEN"
    answer = str(v["answer"])
    top = [{"answer": x["diagnosis"], "answer_vi": DX_VI.get(x["diagnosis"], x["diagnosis"]), "p": x["probability"]} for x in v.get("shortlist", [])]
    msg = v.get("message_vi") or ""
    for en, vi in sorted(DX_VI.items(), key=lambda kv: -len(kv[0])):
        msg = msg.replace(en, vi)
    if translated and v.get("decision") == "uncalibrated":
        msg = msg.replace(f"Trả lời: {answer}", f"Trả lời: {to_vietnamese(answer)}", 1)
    msg = msg.replace(": yes", ": có").replace(": no", ": không").replace("Trả lời: yes", "Trả lời: có").replace("Trả lời: no", "Trả lời: không")
    return {
        "answer": to_vietnamese(answer) if translated else answer, "answer_en": answer, "question_en": english,
        "answer_type": kind, "endpoint": v.get("endpoint"), "p_yes": v.get("p_yes"),
        "trust": {"confidence": v.get("confidence"), "decision": v.get("decision"), "message_vi": msg,
                  "coverage_target": 0.8, "top3": top},
        "routed_class": CLASS_VI[cls], "proposals": [{"box": b[:4], "score": b[4]} for b in props],
        "regions": shown(boxes), "evaluated_question": english in EVALUATED,
        "detector_ms": det_ms, "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.post("/api/feedback")
def feedback(question: str = Form(...), answer: str = Form(...), verdict: str = Form(...), correct_answer: str = Form("")):
    if verdict not in ("dung", "sai", "khong_chac"):
        raise HTTPException(400, "Đánh giá không hợp lệ.")
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), model=f"JF fold {FOLD}", question=question, answer=answer,
               verdict=verdict, correct_answer=correct_answer.strip() or None)
    with open(FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"saved": True}
