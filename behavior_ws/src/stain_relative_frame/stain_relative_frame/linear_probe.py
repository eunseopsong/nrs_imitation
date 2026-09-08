#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free multinomial logistic regression for the step [4] probes.

sklearn is not installed in this workspace and the gate must be runnable from
the ROS environment, so this is a small self-contained softmax regression:
feature standardisation (fit on train only), L2 penalty, full-batch Adam. With
2-6 features and O(100) episodes it converges in well under a second.

The probe is intentionally WEAK-ish but sufficient: the gate asks whether the
feature carries the label at all, so a linear model that can reach ~100% on
absolute coordinates and chance on relative ones is exactly the instrument
wanted.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _one_hot(y_idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y_idx.shape[0], k), dtype=np.float64)
    out[np.arange(y_idx.shape[0]), y_idx] = 1.0
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-300, None)


class LogisticProbe:
    """Multinomial logistic regression. `fit` then `score`."""

    def __init__(self, l2: float = 1e-2, iters: int = 2000, lr: float = 0.1,
                 tol: float = 1e-9):
        self.l2 = float(l2)
        self.iters = int(iters)
        self.lr = float(lr)
        self.tol = float(tol)
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        k = self.classes_.shape[0]
        idx = np.searchsorted(self.classes_, y)
        Y = _one_hot(idx, k)

        self.mu = X.mean(axis=0)
        self.sd = np.maximum(X.std(axis=0), 1e-8)
        Z = (X - self.mu) / self.sd

        n, d = Z.shape
        W = np.zeros((d, k))
        b = np.zeros(k)
        mW = vW = np.zeros_like(W)
        mb = vb = np.zeros_like(b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        prev = np.inf
        for t in range(1, self.iters + 1):
            P = _softmax(Z @ W + b)
            diff = (P - Y) / n
            gW = Z.T @ diff + self.l2 * W
            gb = diff.sum(axis=0)

            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb ** 2
            W -= self.lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            b -= self.lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)

            if t % 50 == 0:
                loss = -np.mean(np.log(np.clip(P[np.arange(n), idx], 1e-300, None)))
                loss += 0.5 * self.l2 * float((W ** 2).sum())
                if abs(prev - loss) < self.tol:
                    break
                prev = loss
        self.W, self.b = W, b
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
        Z = (X - self.mu) / self.sd
        return self.classes_[np.argmax(Z @ self.W + self.b, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y).reshape(-1)))


def episode_split(y: np.ndarray, test_fraction: float, rng: np.random.Generator,
                  stratified: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """8:2 split over EPISODES (one row = one episode here, so over rows).

    Stratified by label, so a small test set cannot end up single-class and
    make the accuracy unreadable.
    """
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]
    if not stratified:
        order = rng.permutation(n)
        n_test = max(1, int(round(test_fraction * n)))
        return order[n_test:], order[:n_test]

    test_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = max(1, int(round(test_fraction * idx.shape[0])))
        test_idx.append(idx[:n_test])
    test = np.concatenate(test_idx)
    train = np.setdiff1d(np.arange(n), test)
    return train, test


def repeated_probe(X: np.ndarray, y: np.ndarray, repeats: int, test_fraction: float,
                   seed: int, shuffle_labels: bool = False,
                   l2: float = 1e-2) -> dict:
    """Repeated stratified hold-out. Returns accuracy stats and the chance level."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    accs, majs = [], []
    for r in range(int(repeats)):
        rng = np.random.default_rng(int(seed) + r)
        y_use = rng.permutation(y) if shuffle_labels else y
        tr, te = episode_split(y_use, test_fraction, rng)
        if np.unique(y_use[tr]).size < 2:
            continue
        probe = LogisticProbe(l2=l2).fit(X[tr], y_use[tr])
        accs.append(probe.score(X[te], y_use[te]))
        _, counts = np.unique(y_use[te], return_counts=True)
        majs.append(float(counts.max() / counts.sum()))
    accs = np.asarray(accs, dtype=np.float64)
    classes, counts = np.unique(y, return_counts=True)
    return {
        "accuracy_mean": float(accs.mean()) if accs.size else float("nan"),
        "accuracy_std": float(accs.std()) if accs.size else float("nan"),
        "accuracy_min": float(accs.min()) if accs.size else float("nan"),
        "accuracy_max": float(accs.max()) if accs.size else float("nan"),
        "n_splits": int(accs.size),
        "n_classes": int(classes.size),
        "class_counts": {int(c): int(n) for c, n in zip(classes, counts)},
        "chance_uniform": 1.0 / float(classes.size),
        "chance_majority": float(counts.max() / counts.sum()),
        "chance_majority_test_mean": float(np.mean(majs)) if majs else float("nan"),
    }


def permutation_null(X: np.ndarray, y: np.ndarray, n_perm: int, test_fraction: float,
                     seed: int, l2: float = 1e-2) -> np.ndarray:
    """Accuracies obtained with the labels permuted -- the empirical null.

    At the episode counts this project works with (tens, not thousands), a
    small test split makes chance-level accuracy noisy: an 84-episode dataset
    splits into ~17 test episodes, so one extra correct episode moves the
    accuracy by 6 points. A fixed "chance + margin" line therefore fires on
    noise. This measures where chance actually lands FOR THIS dataset size, so
    the gate can ask whether a result is outside it.
    """
    accs = []
    for r in range(int(n_perm)):
        rng = np.random.default_rng(int(seed) + 100000 + r)
        y_perm = rng.permutation(np.asarray(y).reshape(-1))
        tr, te = episode_split(y_perm, test_fraction, rng)
        if np.unique(y_perm[tr]).size < 2:
            continue
        accs.append(LogisticProbe(l2=l2).fit(X[tr], y_perm[tr]).score(X[te], y_perm[te]))
    return np.asarray(accs, dtype=np.float64)


def probe_with_null(X: np.ndarray, y: np.ndarray, repeats: int, test_fraction: float,
                    seed: int, n_perm: int = 200, l2: float = 1e-2) -> dict:
    """`repeated_probe` plus the permutation p-value against its own null."""
    res = repeated_probe(X, y, repeats, test_fraction, seed, l2=l2)
    null = permutation_null(X, y, n_perm, test_fraction, seed, l2=l2)
    obs = res["accuracy_mean"]
    if null.size:
        p = float((1.0 + np.sum(null >= obs)) / (1.0 + null.size))
        res.update({
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": p,
            "n_permutations": int(null.size),
        })
    else:
        res.update({"null_mean": float("nan"), "null_std": float("nan"),
                    "null_p95": float("nan"), "p_value": float("nan"),
                    "n_permutations": 0})
    return res
