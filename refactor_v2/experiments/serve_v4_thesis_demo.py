"""Local thesis demo of the V4 system: B + detector-proposed local tokens, no auxiliary loss.

Uses the sealed fold 0 LAST checkpoint and fold 0 TRAIN-only detector with their recorded hashes.
Fold 0 is the fold used by the earlier demos, not a best-fold selection. Sealed sources stay untouched.
"""
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.ngon_ngu import sang_tieng_anh, sang_tieng_viet
from experiments.detector640_checkpoint_v3 import load_bound_640
from experiments.vnext_model_v1 import VNextModel
from src.data.fracatlas_vqa import FracAtlasVQADataset
from src.detector_runtime import FractureDetectorRuntime
from src.preprocessing import canonicalize_raster

R2 = Path(__file__).resolve().parents[1]
SEALED = R2 / "reports/local_noaux_refinement_v1"
FOLD = SEALED / "fold_0"
WEB = Path(__file__).resolve().parent / "v4_thesis_demo"
FEEDBACK = R2 / "reports/v4_demo_feedback.jsonl"
CANONICAL = "Is there a fracture in this X-ray?"
MAX_IMAGE_BYTES = 20 * 1024 * 1024

torch.set_num_threads(int(os.environ.get("BONEVQA_DEMO_THREADS", "4")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_system():
    folder = FOLD / "local_real_noaux"
    checkpoint = folder / "LAST.pt"
    assert sha(checkpoint) == json.loads((folder / "completed.json").read_text())["checkpoint_sha256"]
    model = VNextModel.load(str(checkpoint), device="cpu").eval()
    receipt = json.loads((FOLD / "detector/receipt.json").read_text())
    assert sha(receipt["checkpoint"]) == receipt["sha256"]
    detector = object.__new__(FractureDetectorRuntime)
    detector.model, _ = load_bound_640(receipt["checkpoint"])
    detector.device, detector.image_size, detector.score_threshold, detector.max_boxes = torch.device("cpu"), 640, 0.05, 3
    threshold = json.loads((folder / "operating.json").read_text())["operating_point"]["threshold"]
    evaluated = set()
    for k in range(5):
        records = json.loads((SEALED / f"fold_{k}/local_real_noaux/test_predictions.json").read_text())["records"]
        evaluated.update(r["question"] for r in records)
    return model, detector, threshold, evaluated


MODEL, DETECTOR, SENSITIVE_THRESHOLD, EVALUATED_QUESTIONS = load_system()
LOCK = threading.Lock()
app = FastAPI(title="BoneVQA V4 thesis demo")
app.mount("/static", StaticFiles(directory=WEB), name="static")


def read_image(upload):
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Ảnh vượt quá 20 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return canonicalize_raster(image)
    except Exception as exc:
        raise HTTPException(400, "Không đọc được tệp ảnh.") from exc


def detect(image):
    """Same filtering as FractureDetectorRuntime.predict, keeping the score of each box."""
    canonical, tensor, scale, pad_x, pad_y = DETECTOR._preprocess(image)
    prediction = DETECTOR.model([tensor])[0]
    found = []
    for box, score in zip(prediction["boxes"].tolist(), prediction["scores"].tolist()):
        if score < DETECTOR.score_threshold or len(found) >= DETECTOR.max_boxes:
            continue
        x1 = max(0.0, min(float(canonical.width), (box[0] - pad_x) / scale))
        y1 = max(0.0, min(float(canonical.height), (box[1] - pad_y) / scale))
        x2 = max(0.0, min(float(canonical.width), (box[2] - pad_x) / scale))
        y2 = max(0.0, min(float(canonical.height), (box[3] - pad_y) / scale))
        if x2 > x1 and y2 > y1:
            found.append({"box": [x1, y1, x2, y2], "score": float(score)})
    return found


def manual_boxes(raw, image):
    try:
        boxes = json.loads(raw or "[]")
        boxes = [[float(v) for v in b] for b in boxes]
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "Danh sách vùng không hợp lệ.") from exc
    if len(boxes) > DETECTOR.max_boxes or any(len(b) != 4 for b in boxes):
        raise HTTPException(400, "Chỉ nhận tối đa 3 vùng hình chữ nhật.")
    w, h = image.size
    clipped = [[max(0.0, min(w, b[0])), max(0.0, min(h, b[1])), max(0.0, min(w, b[2])), max(0.0, min(h, b[3]))] for b in boxes]
    return [{"box": b, "score": None} for b in clipped if b[2] > b[0] and b[3] > b[1]]


@app.get("/")
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "V4 · fold 0 LAST", "device": f"cpu ({torch.get_num_threads()} luồng)", "sensitive_threshold": SENSITIVE_THRESHOLD}


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
    with LOCK, torch.no_grad():
        regions = detect(im)
    return {"regions": regions, "width": im.width, "height": im.height, "latency_ms": round((time.perf_counter() - start) * 1000)}


@app.post("/api/ask")
def ask(image: UploadFile = File(...), question: str = Form(...), source: str = Form("auto"), boxes: str = Form("[]")):
    q = question.strip()
    if not q or len(q) > 500:
        raise HTTPException(400, "Câu hỏi cần dài từ 1 đến 500 ký tự.")
    if source not in ("auto", "manual"):
        raise HTTPException(400, "Nguồn gợi ý không hợp lệ.")
    im = read_image(image)
    english, translated = sang_tieng_anh(q)
    start = time.perf_counter()
    with LOCK, torch.no_grad():
        regions = detect(im) if source == "auto" else manual_boxes(boxes, im)
        coords = [r["box"] for r in regions]
        masks = FracAtlasVQADataset._boxes_to_masks(coords, im.size) if coords else []
        out = MODEL.predict_batch({"images": [im], "questions": [english], "boxes": [coords or None], "masks": [masks]})[0]
    answer = str(out["answer"])
    return {
        "answer": sang_tieng_viet(answer) if translated else answer,
        "answer_en": answer,
        "question_en": english,
        "answer_type": str(out.get("answer_type", "OPEN")).upper(),
        "p_yes": out.get("p_yes"),
        "is_fracture_question": english == CANONICAL,
        "evaluated_question": english in EVALUATED_QUESTIONS,
        "source": source,
        "regions": regions,
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.post("/api/feedback")
def feedback(question: str = Form(...), answer: str = Form(...), verdict: str = Form(...), correct_answer: str = Form(""), source: str = Form("auto")):
    if verdict not in ("dung", "sai", "khong_chac"):
        raise HTTPException(400, "Đánh giá không hợp lệ.")
    record = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), question=question, answer=answer, verdict=verdict, correct_answer=correct_answer.strip() or None, source=source)
    with open(FEEDBACK, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"saved": True}
