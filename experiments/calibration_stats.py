"""numpy-only statistics for the critic calibration check (experiment 009).

scipy / sklearn are not installed in the project venv and must not be added
for one experiment, so AUC (Mann-Whitney), Spearman (rank Pearson), the
game-clustered bootstrap, a small IRLS logistic regression and the reliability
table are implemented here on numpy alone. Every function is pure and tested on
known inputs in ``tests/rl/test_critic_calibration.py``.

States within one game share one outcome label and are strongly dependent, so
anything computed over many states per game is resampled by *game*
(``cluster_bootstrap``); the headline analysis instead uses one state per game
so plain row resampling is valid.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from gamekit.mc import wilson_interval
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

CI_LEVEL = (2.5, 97.5)


def rankdata(x: FloatArray) -> FloatArray:
    """Average ranks (1-based), ties share the mean rank."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    # boundaries of runs of equal values
    change = np.flatnonzero(np.diff(sorted_x)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(x)]))
    mean_rank = (starts + ends + 1) / 2.0  # mean of 1-based ranks start+1..end
    run_of = np.repeat(np.arange(len(starts)), ends - starts)
    ranks[order] = mean_rank[run_of]
    return ranks


def auc(score: FloatArray, label: NDArray[Any]) -> float:
    """P(score of a random positive > score of a random negative), ties 1/2.
    NaN when either class is empty."""
    label = np.asarray(label).astype(bool)
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(x: FloatArray, y: FloatArray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def stratified_auc(score: FloatArray, label: NDArray[Any], cell: NDArray[Any]) -> float:
    """AUC within cells of ``cell``, averaged with pair-count weights
    (n_pos * n_neg per cell). Cells with one class contribute nothing, so the
    result answers "given the same public state, does the score still rank?"."""
    label = np.asarray(label).astype(bool)
    order = np.argsort(cell, kind="mergesort")
    cell_sorted = np.asarray(cell)[order]
    bounds = np.flatnonzero(cell_sorted[1:] != cell_sorted[:-1]) + 1
    total = 0.0
    weight = 0.0
    for idx in np.split(order, bounds):
        lab = label[idx]
        pairs = int(lab.sum()) * int((~lab).sum())
        if pairs == 0:
            continue
        total += auc(score[idx], lab) * pairs
        weight += pairs
    return total / weight if weight else float("nan")


def cluster_bootstrap(
    stat: Callable[[NDArray[np.intp]], Mapping[str, float]],
    groups: NDArray[Any],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs of ``stat(row_indices)`` when whole groups (games) are
    resampled with replacement. ``groups`` gives each row's group id."""
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="mergesort")
    g_sorted = np.asarray(groups)[order]
    bounds = np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1
    members = np.split(order, bounds)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[g] for g in chosen])
        for key, value in stat(idx).items():
            draws.setdefault(key, []).append(value)
    return {
        key: (
            float(np.nanpercentile(v, CI_LEVEL[0])),
            float(np.nanpercentile(v, CI_LEVEL[1])),
        )
        for key, v in draws.items()
    }


def irls_logistic(
    x: FloatArray, y: FloatArray, *, iters: int = 30, ridge: float = 1e-6
) -> FloatArray:
    """Logistic regression weights by IRLS; ``x`` must include the intercept
    column. A tiny ridge keeps separable data finite."""
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        z = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        s = np.maximum(p * (1 - p), 1e-9)
        grad = x.T @ (y - p) - ridge * w
        hess = (x * s[:, None]).T @ x + ridge * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logistic_predict(x: FloatArray, w: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))


def standardize(train: FloatArray, *others: FloatArray) -> list[FloatArray]:
    """Standardize columns by ``train``'s mean/std, prepend an intercept."""
    mu = train.mean(axis=0)
    sd = np.where(train.std(axis=0) > 0, train.std(axis=0), 1.0)
    return [np.column_stack([np.ones(len(a)), (a - mu) / sd]) for a in (train, *others)]


def reliability_table(
    score: FloatArray, label: NDArray[Any], n_bins: int = 10
) -> list[dict[str, float]]:
    """Quantile bins of ``score`` with empirical win rate and its Wilson 95%
    interval. Dependence between rows of one game is *not* reflected in the
    interval (use one row per game for a valid one)."""
    edges = np.quantile(score, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf
    bin_of = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bin_of == b
        n = int(sel.sum())
        if n == 0:
            continue
        k = int(label[sel].sum())
        ci = wilson_interval(k, n)
        rows.append(
            {
                "bin": b,
                "n": n,
                "score_mean": float(score[sel].mean()),
                "win_rate": k / n,
                "win_ci_low": float(ci.lower),
                "win_ci_high": float(ci.upper),
            }
        )
    return rows


def quantiles(x: FloatArray) -> dict[str, float]:
    qs = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    return {f"q{int(q * 100):03d}": float(np.quantile(x, q)) for q in qs}
