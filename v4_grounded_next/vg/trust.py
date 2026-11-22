"""Trust layer: calibrated confidence and deferral on top of an unchanged V4 checkpoint.

Nothing here touches the model or its answers.  For the three yes / no endpoints

    fracture     "Is there a fracture in this X-ray?"        P(yes) = p_final
    tumor        "Is there a bone tumor in this X-ray?"      P(yes) = yes / (yes + no) of the closed head
    malignancy   "Is the bone lesion malignant?"             idem

the layer turns what inference already produced into P(the answer is correct):

    head            logistic model on |logit P(yes)|                                  (Platt scaling of the confidence)
    head_detector   the same plus the detector's top box score and its interaction with the answer
                    (a complementary, image-grounded signal; cf. hallucination-aware calibration)

and defers the case to a clinician when that probability is below a threshold.  The threshold is the validation
quantile that keeps a chosen share of validation cases (coverage 0.9 or 0.8).  For the open diagnosis question the
optional shortlist uses the Qwen scores of the candidate diagnoses (the TRAIN answer set): temperature-scaled softmax,
top-3 list, deferral on the top probability.

Everything is fitted on calibration / model_val scores only (scripts/trust_fit.py) and stored as JSON, so the layer adds
no forward pass for the yes / no endpoints.  The shortlist costs one Qwen pass per candidate and is off by default.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .paths import CANONICAL_Q, read_json

SCHEMA = "vg-trust-v1"
ENDPOINT_OF = {CANONICAL_Q: "fracture", "Is there a bone tumor in this X-ray?": "tumor", "Is the bone lesion malignant?": "malignancy"}
DIAGNOSIS_Q = "What is the most likely diagnosis?"
COVERAGES = (0.9, 0.8)
SIGNALS = ("head", "head_detector")
TEMPERATURES = np.linspace(0.3, 5.0, 48)


def top_box_score(proposals: Sequence) -> float:
    """Highest detector score among the proposals of one image (0 when there is none)."""
    return max((float(b[4]) for b in proposals), default=0.0)


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def features(p_yes, score, signal: str) -> np.ndarray:
    p, s = np.atleast_1d(np.asarray(p_yes, dtype=float)), np.atleast_1d(np.asarray(score, dtype=float))
    margin, said_yes = np.abs(_logit(p)), (p >= 0.5).astype(float)
    return margin[:, None] if signal == "head" else np.c_[margin, said_yes, s, said_yes * s]


def _coverage_key(c: float) -> str:
    return f"{c:.2f}"


def fit_closed(p_yes, y, score) -> Dict:
    """Both confidence models and their coverage thresholds from validation scores of one endpoint."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    p, y, s = np.asarray(p_yes, dtype=float), np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    ok = ((p >= 0.5).astype(int) == y).astype(int)
    assert 0 < ok.sum() < len(ok), "validation answers are all right or all wrong: nothing to fit"
    out = {"n": int(len(y)), "positives": int(y.sum()), "accuracy": float(ok.mean())}
    for signal in SIGNALS:
        m = LogisticRegression(C=10.0, max_iter=1000).fit(features(p, s, signal), ok)
        conf = m.predict_proba(features(p, s, signal))[:, 1]
        out[signal] = {"coef": [float(v) for v in m.coef_[0]], "intercept": float(m.intercept_[0]),
                       "threshold": {_coverage_key(c): float(np.quantile(conf, 1 - c)) for c in COVERAGES},
                       "val_auroc_correct": float(roc_auc_score(ok, conf))}
    return out


def softmax_t(scores, temperature: float) -> np.ndarray:
    z = np.atleast_2d(np.asarray(scores, dtype=float)) / temperature
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_diagnosis(scores, y_index, candidates: List[str]) -> Dict:
    z, y = np.asarray(scores, dtype=float), np.asarray(y_index, dtype=int)
    nll = lambda t: float(-np.mean(np.log(softmax_t(z, t)[np.arange(len(y)), y] + 1e-12)))
    t = float(min(TEMPERATURES, key=nll))
    p = softmax_t(z, t)
    return {"n": int(len(y)), "candidates": list(candidates), "temperature": t, "accuracy": float((p.argmax(1) == y).mean()),
            "threshold": {_coverage_key(c): float(np.quantile(p.max(1), 1 - c)) for c in COVERAGES}}


def _vi(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


class TrustLayer:
    def __init__(self, spec: Dict):
        assert spec.get("schema") == SCHEMA, spec.get("schema")
        self.spec, self.endpoints, self.dx = spec, spec["endpoints"], spec.get("diagnosis")

    @classmethod
    def load(cls, path) -> "TrustLayer":
        return cls(read_json(path))

    # ------------------------------------------------------------------ yes / no endpoints
    def confidence(self, endpoint: str, p_yes, score, signal: str = "head_detector") -> np.ndarray:
        m = self.endpoints[endpoint][signal]
        z = features(p_yes, score, signal) @ np.asarray(m["coef"]) + m["intercept"]
        return 1 / (1 + np.exp(-z))

    def threshold(self, endpoint: str, coverage: float, signal: str = "head_detector") -> float:
        return self.endpoints[endpoint][signal]["threshold"][_coverage_key(coverage)]

    def closed(self, endpoint: str, p_yes: float, score: float, coverage: float = 0.8, signal: str = "head_detector") -> Dict:
        conf, tau = float(self.confidence(endpoint, p_yes, score, signal)[0]), self.threshold(endpoint, coverage, signal)
        answer, defer = ("yes" if p_yes >= 0.5 else "no"), conf < tau
        note = []
        if score == 0.0:
            note.append("detector không có hộp nào")
        msg = (f"Chưa đủ tin cậy ({_vi(conf)} < ngưỡng {_vi(tau)}): chuyển bác sĩ đọc. Câu trả lời tạm của mô hình: {answer}."
               if defer else f"Trả lời: {answer} (độ tin cậy {_vi(conf)}, ngưỡng {_vi(tau)}).")
        return {"endpoint": endpoint, "answer": answer, "p_yes": float(p_yes), "top_box_score": float(score), "confidence": conf,
                "threshold": tau, "coverage_target": coverage, "signal": signal, "decision": "defer" if defer else "answer",
                "notes": note, "message_vi": msg + (" Ghi chú: " + "; ".join(note) + "." if note else "")}

    # ------------------------------------------------------------------ diagnosis shortlist
    def diagnosis(self, scores, coverage: float = 0.8, k: int = 3, generated: Optional[str] = None) -> Dict:
        assert self.dx, "this trust file has no diagnosis block"
        p = softmax_t(scores, self.dx["temperature"])[0]
        order = np.argsort(-p)[:k]
        short = [{"diagnosis": self.dx["candidates"][i], "probability": float(p[i])} for i in order]
        conf, tau = float(p.max()), self.dx["threshold"][_coverage_key(coverage)]
        defer = conf < tau
        lst = "; ".join(f"{s['diagnosis']} ({_vi(s['probability'])})" for s in short)
        msg = (f"Chưa đủ tin cậy ({_vi(conf)} < ngưỡng {_vi(tau)}): chuyển bác sĩ đọc. Chẩn đoán phân biệt: {lst}."
               if defer else f"Chẩn đoán có khả năng nhất: {short[0]['diagnosis']} (độ tin cậy {_vi(conf)}). Chẩn đoán phân biệt: {lst}.")
        out = {"endpoint": "diagnosis", "answer": short[0]["diagnosis"], "shortlist": short, "confidence": conf, "threshold": tau,
               "coverage_target": coverage, "decision": "defer" if defer else "answer", "notes": [], "message_vi": msg}
        if generated is not None:
            out["generated_answer"] = generated
            if generated.strip().lower() != short[0]["diagnosis"].lower():
                out["notes"].append("câu trả lời sinh trực tiếp khác đáp án đứng đầu danh sách")
                out["message_vi"] += f" Ghi chú: câu trả lời sinh trực tiếp là \"{generated}\"."
        return out

    # ------------------------------------------------------------------ everything else
    @staticmethod
    def uncalibrated(answer: str, raw_confidence: Optional[float]) -> Dict:
        c = "" if raw_confidence is None else f" (điểm thô {_vi(raw_confidence)}, chưa hiệu chỉnh)"
        return {"endpoint": None, "answer": answer, "confidence": None, "raw_confidence": raw_confidence, "decision": "uncalibrated",
                "notes": ["loại câu hỏi này chưa có dữ liệu hiệu chỉnh"], "message_vi": f"Trả lời: {answer}{c}. Chưa có ngưỡng tin cậy cho loại câu hỏi này."}
