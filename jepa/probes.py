"""Closed-form probes shared by training (quick checks) and evaluation."""
from __future__ import annotations

import numpy as np


class Standardizer:
    def __init__(self, x):
        self.mean = x.mean(0)
        self.std = np.maximum(x.std(0), 1e-6)

    def __call__(self, x):
        return (x - self.mean) / self.std


def ridge_fit(x, y, ridge=1e-2):
    """Linear least squares with an L2 penalty on standardized inputs (bias unpenalized)."""
    sx = Standardizer(x)
    xs = np.concatenate([sx(x), np.ones((len(x), 1))], axis=1)
    reg = ridge * len(x) * np.eye(xs.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)
    return lambda q: np.concatenate([sx(q), np.ones((len(q), 1))], axis=1) @ w


def r2_score(y, pred):
    denom = np.sum((y - y.mean(0)) ** 2, axis=0)
    return 1.0 - np.sum((y - pred) ** 2, axis=0) / np.maximum(denom, 1e-12)


def ridge_r2(x_tr, y_tr, x_te, y_te, ridge=1e-2):
    y_tr, y_te = y_tr.reshape(len(y_tr), -1), y_te.reshape(len(y_te), -1)
    return r2_score(y_te, ridge_fit(x_tr, y_tr, ridge)(x_te))


def ridge_classify(x_tr, labels_tr, x_te, labels_te, n_classes=None, ridge=1e-2):
    """Accuracy of a one-vs-all ridge classifier (fast, closed-form linear probe)."""
    n = int(n_classes or max(labels_tr.max(), labels_te.max()) + 1)
    pred = ridge_fit(x_tr, np.eye(n)[labels_tr], ridge)(x_te).argmax(1)
    return float((pred == labels_te).mean()), pred


def knn_classify(x_tr, labels_tr, x_te, labels_te, k=20):
    """Cosine kNN on standardized features (majority vote)."""
    s = Standardizer(x_tr)
    a = s(x_tr)
    b = s(x_te)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    n = int(max(labels_tr.max(), labels_te.max()) + 1)
    pred = np.empty(len(b), dtype=np.int64)
    for i in range(0, len(b), 2048):
        sim = b[i:i + 2048] @ a.T
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        votes = np.zeros((len(idx), n))
        np.add.at(votes, (np.repeat(np.arange(len(idx)), k), labels_tr[idx].ravel()), 1)
        pred[i:i + 2048] = votes.argmax(1)
    return float((pred == labels_te).mean()), pred


def mlp_r2(x_tr, y_tr, x_te, y_te, seed=0):
    """Nonlinear probe (sklearn MLP); None if sklearn is not installed."""
    try:
        from sklearn.neural_network import MLPRegressor
    except ImportError:
        return None
    y_tr, y_te = y_tr.reshape(len(y_tr), -1), y_te.reshape(len(y_te), -1)
    sx, sy = Standardizer(x_tr), Standardizer(y_tr)
    m = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300, early_stopping=True, random_state=seed)
    target = sy(y_tr)
    m.fit(sx(x_tr), target[:, 0] if target.shape[1] == 1 else target)
    pred = m.predict(sx(x_te)).reshape(len(x_te), -1) * sy.std + sy.mean
    return r2_score(y_te, pred)


def effective_rank(x):
    """exp(entropy) of the normalized covariance eigenvalues (Roy & Vetterli 2007)."""
    ev = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0, None)
    p = ev / max(ev.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum())), ev[::-1]
