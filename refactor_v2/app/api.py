import base64
import io
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from src.preprocessing import canonicalize_raster

from app.inference import build_overlay, get_engine, mask_to_bbox
from app.ngon_ngu import sang_tieng_anh, sang_tieng_viet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bonevqa.api")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
EXAMPLES_DIR = STATIC_DIR / "examples"
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

KIEU_PROMPT_HOP_LE = {"contour", "box", "circle", "mask"}
GIOI_HAN_LICH_SU = 50
DANH_GIA_HOP_LE = {"dung", "sai", "khong_chac"}
TEP_PHAN_HOI = Path(os.environ.get("BONEVQA_FEEDBACK", APP_DIR.parent / "reports" / "phan_hoi_bac_si.jsonl"))

app = FastAPI(title="BoneVQA-Prompt API", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

LICH_SU_HOI_THOAI = defaultdict(list)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def doc_anh(upload: UploadFile) -> Image.Image:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Ảnh tải lên rỗng.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {exc}") from exc
    return canonicalize_raster(img)


def parse_json_list(raw: Optional[str], ten: str):
    if not raw:
        return None
    try:
        val = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Tham số {ten} không phải JSON hợp lệ.") from exc
    if val in ([], None):
        return None
    if not isinstance(val, list):
        raise HTTPException(status_code=400, detail=f"Tham số {ten} phải là danh sách.")
    return val


def thong_tin_masks(masks: np.ndarray):
    if masks is None or len(masks) == 0:
        return []
    return [
        {"id": i + 1, "bbox": mask_to_bbox(m), "area": int(m.sum())}
        for i, m in enumerate(masks)
    ]


@app.on_event("startup")
def khoi_dong():
    get_engine()


@app.get("/")
def trang_chu():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": bool(getattr(engine, "model_loaded", False)),
        "engine": getattr(engine, "name", type(engine).__name__),
        "device": getattr(engine, "device", "cpu"),
        "detector_loaded": bool(getattr(engine, "detector", None)),
        "detector_provenance": getattr(engine, "detector_provenance", None),
        "calibration_provenance": getattr(engine, "calibration_provenance", None),
    }


@app.get("/api/examples")
def danh_sach_vi_du():
    duoi_hop_le = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    items = []
    for p in sorted(EXAMPLES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in duoi_hop_le:
            items.append({"name": p.name, "url": f"/static/examples/{p.name}"})
    return {"examples": items}


@app.post("/api/segment")
async def phan_vung(
    image: UploadFile = File(...),
    boxes: Optional[str] = Form(None),
    points: Optional[str] = Form(None),
):
    t0 = time.perf_counter()
    img = await doc_anh(image)
    boxes_list = parse_json_list(boxes, "boxes")
    points_list = parse_json_list(points, "points")
    engine = get_engine()
    masks = engine.segment(img, boxes=boxes_list, points=points_list)
    masks = np.asarray(masks).astype(np.uint8)
    if masks.ndim == 2:
        masks = masks[None]
    overlay = build_overlay(img, masks)
    return JSONResponse(
        {
            "masks_count": int(len(masks)),
            "regions": thong_tin_masks(masks),
            "overlay_image": pil_to_base64(overlay),
            "width": img.width,
            "height": img.height,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    )


@app.post("/api/ask")
async def hoi_dap(
    image: UploadFile = File(...),
    question: str = Form(...),
    boxes: Optional[str] = Form(None),
    prompt_kind: str = Form("contour"),
    session_id: Optional[str] = Form(None),
):
    t0 = time.perf_counter()
    cau_hoi = (question or "").strip()
    if not cau_hoi:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    kind = (prompt_kind or "contour").lower()
    if kind not in KIEU_PROMPT_HOP_LE:
        raise HTTPException(status_code=400, detail=f"prompt_kind phải thuộc {sorted(KIEU_PROMPT_HOP_LE)}.")
    img = await doc_anh(image)
    boxes_list = parse_json_list(boxes, "boxes")
    sid = session_id or uuid.uuid4().hex

    engine = get_engine()
    cau_hoi_anh, da_dich = sang_tieng_anh(cau_hoi)
    ket_qua = engine.answer(img, cau_hoi_anh, boxes=boxes_list, prompt_kind=kind)
    dap_an_anh = ket_qua["answer"]
    dap_an_hien = sang_tieng_viet(dap_an_anh) if da_dich else dap_an_anh
    masks = np.asarray(ket_qua.get("masks")).astype(np.uint8) if ket_qua.get("masks") is not None else np.zeros((0, img.height, img.width), np.uint8)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    ban_ghi = {
        "role": "user",
        "content": cau_hoi,
        "question_en": cau_hoi_anh,
        "prompt_kind": kind,
        "boxes": boxes_list,
        "detector_boxes": ket_qua.get("detector_boxes"),
        "time": time.time(),
    }
    tra_loi = {
        "role": "assistant",
        "content": dap_an_hien,
        "answer_en": dap_an_anh,
        "answer_type": ket_qua["answer_type"],
        "confidence": ket_qua["confidence"],
        "latency_ms": latency_ms,
        "time": time.time(),
    }
    lich_su = LICH_SU_HOI_THOAI[sid]
    lich_su.extend([ban_ghi, tra_loi])
    if len(lich_su) > GIOI_HAN_LICH_SU * 2:
        del lich_su[: len(lich_su) - GIOI_HAN_LICH_SU * 2]

    return JSONResponse(
        {
            "session_id": sid,
            "answer": dap_an_hien,
            "answer_en": dap_an_anh,
            "question_en": cau_hoi_anh,
            "answer_type": ket_qua["answer_type"],
            "confidence": ket_qua["confidence"],
            "prompt_image": pil_to_base64(ket_qua["prompt_image"]),
            "lens_image": pil_to_base64(ket_qua["lens_image"]),
            "masks_count": int(len(masks)),
            "regions": thong_tin_masks(masks),
            "detector_boxes": ket_qua.get("detector_boxes"),
            "latency_ms": latency_ms,
            "engine": getattr(engine, "name", type(engine).__name__),
            "history": lich_su,
        }
    )


@app.post("/api/phan-hoi")
async def ghi_phan_hoi(
    session_id: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    danh_gia: str = Form(...),
    answer_type: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    dap_an_dung: Optional[str] = Form(None),
    ghi_chu: Optional[str] = Form(None),
    masks_count: Optional[int] = Form(None),
    prompt_kind: Optional[str] = Form(None),
):
    if danh_gia not in DANH_GIA_HOP_LE:
        raise HTTPException(status_code=400, detail=f"danh_gia phải thuộc {sorted(DANH_GIA_HOP_LE)}.")
    ban_ghi = {
        "thoi_gian": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "confidence": confidence,
        "danh_gia": danh_gia,
        "dap_an_dung": (dap_an_dung or "").strip() or None,
        "ghi_chu": (ghi_chu or "").strip() or None,
        "masks_count": masks_count,
        "prompt_kind": prompt_kind,
    }
    TEP_PHAN_HOI.parent.mkdir(parents=True, exist_ok=True)
    with open(TEP_PHAN_HOI, "a", encoding="utf-8") as f:
        f.write(json.dumps(ban_ghi, ensure_ascii=False) + chr(10))
    logger.info("Đã ghi phản hồi %s cho câu hỏi %r", danh_gia, question[:60])
    return {"da_ghi": True, "tep": str(TEP_PHAN_HOI)}


@app.get("/api/phan-hoi/thong-ke")
def thong_ke_phan_hoi():
    if not TEP_PHAN_HOI.exists():
        return {"tong": 0, "theo_danh_gia": {}, "ty_le_dung": None, "tep": str(TEP_PHAN_HOI)}
    dem = defaultdict(int)
    tong = 0
    with open(TEP_PHAN_HOI, encoding="utf-8") as f:
        for dong in f:
            dong = dong.strip()
            if not dong:
                continue
            try:
                dem[json.loads(dong).get("danh_gia", "khong_ro")] += 1
                tong += 1
            except json.JSONDecodeError:
                continue
    co_ket_luan = dem.get("dung", 0) + dem.get("sai", 0)
    ty_le = round(dem.get("dung", 0) / co_ket_luan, 4) if co_ket_luan else None
    return {"tong": tong, "theo_danh_gia": dict(dem), "ty_le_dung": ty_le, "tep": str(TEP_PHAN_HOI)}


@app.get("/api/history/{session_id}")
def lay_lich_su(session_id: str):
    return {"session_id": session_id, "history": LICH_SU_HOI_THOAI.get(session_id, [])}


@app.delete("/api/history/{session_id}")
def xoa_lich_su(session_id: str):
    LICH_SU_HOI_THOAI.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}
