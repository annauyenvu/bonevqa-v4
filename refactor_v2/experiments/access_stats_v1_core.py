"""Các hàm độ đo dùng chung cho phân tích bài IEEE Access."""
import numpy as np
from scipy.stats import rankdata


def ap(y, p):
    ix = np.argsort(-p, kind='stable'); y, p = y[ix], p[ix]
    e = np.r_[np.where(p[:-1] != p[1:])[0], len(p) - 1]; tp = np.cumsum(y)[e]; tot = e + 1
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / tot) / y.sum())


def auroc(y, p):
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(bins) if (b == k).any()))
