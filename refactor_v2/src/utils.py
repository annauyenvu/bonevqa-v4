import json
import logging
import os
import random
import re
import string
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from .endpoints import is_canonical_fracture_presence_question

CLOSED_PREFIXES = (
    "is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ", "has ", "have ", "had ",
    "should ", "will ", "would ", "am ", "isn't ", "aren't ", "doesn't ", "don't ",
)


def setup_logger(name: str = "bonevqa", log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon has no CUDA, but MPS is a substantial speed-up over CPU
    # for inference.  Keep CUDA as the first choice for existing deployments.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_env_dirs(work_dir: str = "D:/BaoYen_work"):
    os.environ.setdefault("HF_HOME", os.path.join(work_dir, "hf_cache"))
    os.environ.setdefault("TMP", os.path.join(work_dir, "tmp"))
    os.environ.setdefault("TEMP", os.path.join(work_dir, "tmp"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(work_dir, "pip_cache"))
    for key in ("HF_HOME", "TMP", "TEMP"):
        os.makedirs(os.environ[key], exist_ok=True)


def deep_update(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_path = cfg.pop("_base_", None)
    if base_path:
        base_full = base_path if os.path.isabs(base_path) else os.path.join(os.path.dirname(path), base_path)
        base_cfg = load_config(base_full)
        cfg = deep_update(base_cfg, cfg)
    return cfg


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "ratio": trainable / max(total, 1)}


def format_params(stats: Dict[str, int]) -> str:
    return f"tổng {stats['total'] / 1e6:.1f}M, huấn luyện {stats['trainable'] / 1e6:.2f}M ({stats['ratio'] * 100:.2f}%)"


def gpu_memory_str() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return f"VRAM hiện tại {alloc:.2f} GB, đỉnh {peak:.2f} GB"


CLOSED_PATTERNS_VI = (
    r"\bcó\b.*\bkhông\s*\??$", r"\bphải\s*không\b", r"\bđúng\s*không\b", r"\bhay\s*không\b",
)

CHOICE_PATTERNS = (r"\b(yes or no|or not)\b", r"\bhay không\b")


def is_choice_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if any(re.search(pat, q) for pat in CHOICE_PATTERNS):
        return False
    return re.search(r"\S\s+(?:or|hay|hoặc)\s+\S", q) is not None


def is_polar_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if re.search(r"\b(yes or no|or not)\b", q):
        return True
    if q.startswith(CLOSED_PREFIXES):
        return True
    return any(re.search(pat, q) for pat in CLOSED_PATTERNS_VI)


def is_closed_question(question: str) -> bool:
    if re.search(r"\b(yes or no|or not)\b", (question or "").strip().lower()):
        return True
    if is_choice_question(question):
        return False
    return is_polar_question(question)


def normalize_answer(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def normalize_yes_no(text: str) -> str:
    t = normalize_answer(text)
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return t


def open_recall(pred: str, gt: str) -> float:
    gt_tokens = normalize_answer(gt).split()
    if not gt_tokens:
        return 0.0
    pred_tokens = set(normalize_answer(pred).split())
    return sum(1 for t in gt_tokens if t in pred_tokens) / len(gt_tokens)


def bleu1(pred: str, gt: str) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref = normalize_answer(gt).split()
    hyp = normalize_answer(pred).split()
    if not ref or not hyp:
        return 0.0
    return float(sentence_bleu([ref], hyp, weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1))


def token_f1(pred: str, gt: str) -> float:
    gt_tokens = normalize_answer(gt).split()
    pred_tokens = normalize_answer(pred).split()
    if not gt_tokens or not pred_tokens:
        return float(gt_tokens == pred_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    so_khop = 0
    con_lai = dict(common)
    for t in gt_tokens:
        if con_lai.get(t, 0) > 0:
            so_khop += 1
            con_lai[t] -= 1
    if so_khop == 0:
        return 0.0
    precision = so_khop / len(pred_tokens)
    recall = so_khop / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gt: str) -> float:
    from rouge_score import rouge_scorer

    ref = normalize_answer(gt)
    hyp = normalize_answer(pred)
    if not ref or not hyp:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(ref, hyp)["rougeL"].fmeasure)


def bertscore_f1_batch(preds: List[str], gts: List[str], model_type: str = "roberta-large") -> List[float]:
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=model_type, lang="en", rescale_with_baseline=True)
    hyps = [normalize_answer(p) or "trống" for p in preds]
    refs = [normalize_answer(g) or "trống" for g in gts]
    _, _, f1 = scorer.score(hyps, refs)
    return [float(x) for x in f1]


def compute_metrics(records: List[dict], fracture_threshold: Optional[float] = None) -> Dict[str, float]:
    closed = [r for r in records if r.get("answer_type") == "CLOSED"]
    opened = [r for r in records if r.get("answer_type") != "CLOSED"]
    closed_acc = float(np.mean([normalize_yes_no(r["pred"]) == normalize_yes_no(r["gt"]) for r in closed])) if closed else 0.0
    open_acc = float(np.mean([normalize_answer(r["pred"]) == normalize_answer(r["gt"]) for r in opened])) if opened else 0.0
    open_rec = float(np.mean([open_recall(r["pred"], r["gt"]) for r in opened])) if opened else 0.0
    bleu = float(np.mean([bleu1(r["pred"], r["gt"]) for r in opened])) if opened else 0.0
    n = len(records)
    overall = (closed_acc * len(closed) + open_acc * len(opened)) / n if n else 0.0
    result = {
        "closed_acc": closed_acc, "open_acc": open_acc, "open_recall": open_rec, "bleu1": bleu,
        "overall_acc": overall, "n_closed": len(closed), "n_open": len(opened), "n": n,
    }
    fracture = [
        r for r in closed
        if is_canonical_fracture_presence_question(r.get("question"))
    ]
    if fracture:
        pairs = []
        for r in fracture:
            pred = normalize_yes_no(r["pred"])
            if fracture_threshold is not None and r.get("p_yes") is not None:
                pred = "yes" if float(r["p_yes"]) >= fracture_threshold else "no"
            pairs.append((pred, normalize_yes_no(r["gt"])))
        tp = sum(p == "yes" and g == "yes" for p, g in pairs)
        tn = sum(p == "no" and g == "no" for p, g in pairs)
        pos = sum(g == "yes" for _, g in pairs)
        neg = sum(g == "no" for _, g in pairs)
        sensitivity = tp / pos if pos else 0.0
        specificity = tn / neg if neg else 0.0
        result.update({
            "fracture_sensitivity": sensitivity,
            "fracture_specificity": specificity,
            "fracture_balanced_accuracy": 0.5 * (sensitivity + specificity),
            "n_fracture_presence": len(fracture),
        })
        scored = [r for r in fracture if r.get("p_yes") is not None]
        if len(scored) == len(fracture) and pos and neg:
            from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

            labels = np.asarray(
                [1 if normalize_yes_no(r["gt"]) == "yes" else 0 for r in scored], dtype=np.int64
            )
            scores = np.asarray([float(r["p_yes"]) for r in scored], dtype=np.float64)
            result["fracture_auroc"] = float(roc_auc_score(labels, scores))
            result["fracture_auprc"] = float(average_precision_score(labels, scores))
            result["fracture_prevalence"] = float(labels.mean())
            result["fracture_brier"] = float(brier_score_loss(labels, scores))
            edges = np.linspace(0.0, 1.0, 11)
            ece = 0.0
            for bin_idx in range(10):
                in_bin = ((scores >= edges[bin_idx])
                          & (scores < edges[bin_idx + 1] if bin_idx < 9 else scores <= edges[bin_idx + 1]))
                if np.any(in_bin):
                    ece += float(in_bin.mean()) * abs(float(scores[in_bin].mean()) - float(labels[in_bin].mean()))
            result["fracture_ece10"] = float(ece)
        if fracture_threshold is not None:
            result["fracture_yes_threshold"] = float(fracture_threshold)
    return result


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_training_curves(history: List[dict], out_dir: str, run_name: str):
    plt = _setup_matplotlib()
    os.makedirs(out_dir, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h.get("train_loss", np.nan) for h in history], marker="o", color="#1f77b4", label="Loss huấn luyện")
    if any("val_loss" in h for h in history):
        axes[0].plot(epochs, [h.get("val_loss", np.nan) for h in history], marker="s", color="#d62728", label="Loss kiểm định")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Đường cong loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for key, label, color in (("val_closed_acc", "Độ chính xác câu đóng", "#2ca02c"),
                              ("val_open_acc", "Độ chính xác câu mở", "#ff7f0e"),
                              ("val_overall_acc", "Độ chính xác chung", "#9467bd")):
        if any(key in h for h in history):
            axes[1].plot(epochs, [h.get(key, np.nan) for h in history], marker="o", color=color, label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Độ chính xác")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Độ chính xác trên tập kiểm định")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"curve_{run_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_yes_no(records: List[dict], out_path: str, title: str = ""):
    plt = _setup_matplotlib()
    labels = ["yes", "no"]
    mat = np.zeros((2, 2), dtype=int)
    for r in records:
        if r.get("answer_type") != "CLOSED":
            continue
        g, p = normalize_yes_no(r["gt"]), normalize_yes_no(r["pred"])
        if g in labels and p in labels:
            mat[labels.index(g), labels.index(p)] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black" if mat[i, j] < mat.max() / 2 else "white")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["có (yes)", "không (no)"])
    ax.set_yticklabels(["có (yes)", "không (no)"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ablation_bar(results: Dict[str, Dict[str, float]], out_path: str, title: str = "So sánh các cấu hình (ablation)"):
    plt = _setup_matplotlib()
    names = list(results.keys())
    metrics = [("closed_acc", "Chính xác đóng"), ("open_acc", "Chính xác mở"), ("open_recall", "Recall mở"),
               ("bleu1", "BLEU-1"), ("overall_acc", "Chính xác chung")]
    x = np.arange(len(names))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(names) + 3), 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
    for i, (key, label) in enumerate(metrics):
        vals = [results[n].get(key, 0.0) for n in names]
        bars = ax.bar(x + (i - 2) * width, vals, width, label=label, color=colors[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Giá trị")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
