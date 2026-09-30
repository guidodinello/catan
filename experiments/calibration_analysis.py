"""The pre-registered analysis of experiment 009, over the row arrays built by
``experiments.critic_calibration.stack_samples``.

Two scores are analysed everywhere: raw ``V`` and ``V+ = V + own public VP/10``.
The critic was trained on ``ShapedReward`` (``phi = public VP / 10``,
``phi(terminal) = 0``), whose shaping telescopes, so ``V(s)`` estimates roughly
``E[gamma^N * (+-1)] - phi(s)``: raw V carries a *negative* own-VP term, and
``V+`` removes it. Neither is a probability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.calibration_stats import (
    auc,
    cluster_bootstrap,
    irls_logistic,
    logistic_predict,
    quantiles,
    reliability_table,
    spearman,
    standardize,
    stratified_auc,
)

N_BOOT_HEADLINE = 2000
N_BOOT_CLUSTER = 200
BOOT_SEED = 20260929
# Own-turn-count tercile cut points for the "game progress" strata, frozen in
# log 009 from the smoke run (which read no win rate).
TURN_CUTS: tuple[float, float] = (12.0, 24.0)
# Table-leader public VP strata.
LEADER_CUTS: tuple[float, float] = (5.0, 7.0)

Rows = dict[str, NDArray[Any]]


def _ci(v: tuple[float, float]) -> list[float]:
    return [v[0], v[1]]


def _boot_rows(stat: Any, n: int, seed: int, b: int) -> dict[str, list[float]]:
    """Plain row bootstrap (independent rows)."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(b):
        for k, val in stat(rng.integers(0, n, n)).items():
            draws.setdefault(k, []).append(val)
    return {
        k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k, v in draws.items()
    }


def _scores(rows: Rows) -> dict[str, NDArray[Any]]:
    return {"v": rows["v"], "v_plus": rows["v"] + rows["vp_own"] / 10.0}


def headline(rows: Rows) -> dict[str, Any]:
    """One uniformly random decision per game: independent samples."""
    sel = rows["state_headline"]
    y = rows["state_win"][sel].astype(float)
    sc = {k: v[sel] for k, v in _scores(rows).items()}
    final_vp = rows["state_final_vp"][sel].astype(float)
    lead, own = rows["lead"][sel], rows["vp_own"][sel]

    def stat(idx: NDArray[np.intp]) -> dict[str, float]:
        return {
            "auc_v_plus": auc(sc["v_plus"][idx], y[idx]),
            "auc_v": auc(sc["v"][idx], y[idx]),
            "auc_lead": auc(lead[idx], y[idx]),
            "auc_own_vp": auc(own[idx], y[idx]),
            "spearman_v_plus_win": spearman(sc["v_plus"][idx], y[idx]),
            "spearman_v_plus_final_vp": spearman(sc["v_plus"][idx], final_vp[idx]),
            "spearman_v_plus_lead": spearman(sc["v_plus"][idx], lead[idx]),
            "spearman_v_own_vp": spearman(sc["v"][idx], own[idx]),
            "spearman_v_plus_own_vp": spearman(sc["v_plus"][idx], own[idx]),
        }

    point = stat(np.arange(len(y)))
    cis = _boot_rows(stat, len(y), BOOT_SEED, N_BOOT_HEADLINE)
    return {
        "n_games": int(len(y)),
        "win_rate": float(y.mean()),
        "metrics": {k: {"value": point[k], "ci95": cis[k]} for k in point},
        "reliability_v_plus": reliability_table(sc["v_plus"], y),
        "reliability_v": reliability_table(sc["v"], y),
        "naive_probability_v_plus": "p_hat = (V+ + 1) / 2 vs win_rate per bin",
    }


def all_states(rows: Rows) -> dict[str, Any]:
    """Every rl decision; game-clustered bootstrap (B=200)."""
    y = rows["state_win"].astype(float)
    game = rows["state_game"]
    sc = _scores(rows)
    lead, own, turn = rows["lead"], rows["vp_own"], rows["turn"]
    leader_vp = own - lead  # best opponent
    leader_vp = np.maximum(own, leader_vp)
    cell = own.astype(np.int64) * 1000 + (lead.astype(np.int64) + 500)

    def stat(idx: NDArray[np.intp]) -> dict[str, float]:
        out = {
            "auc_v_plus": auc(sc["v_plus"][idx], y[idx]),
            "auc_v": auc(sc["v"][idx], y[idx]),
            "auc_lead": auc(lead[idx], y[idx]),
            "auc_own_vp": auc(own[idx], y[idx]),
            "stratified_auc_v_plus_own_vp_x_lead": stratified_auc(
                sc["v_plus"][idx], y[idx], cell[idx]
            ),
            "stratified_auc_v_own_vp_x_lead": stratified_auc(
                sc["v"][idx], y[idx], cell[idx]
            ),
            "spearman_v_plus_lead": spearman(sc["v_plus"][idx], lead[idx]),
        }
        for name, key in (("turn", turn), ("leader_vp", leader_vp)):
            cuts = TURN_CUTS if name == "turn" else LEADER_CUTS
            band = np.digitize(key[idx], cuts)
            for b in range(3):
                m = band == b
                out[f"auc_v_plus_{name}_band{b}"] = auc(sc["v_plus"][idx][m], y[idx][m])
        return out

    point = stat(np.arange(len(y)))
    cis = cluster_bootstrap(stat, game, n_boot=N_BOOT_CLUSTER, seed=BOOT_SEED)
    return {
        "n_states": int(len(y)),
        "n_games": int(len(np.unique(game))),
        "turn_cuts": list(TURN_CUTS),
        "leader_vp_cuts": list(LEADER_CUTS),
        "metrics": {k: {"value": point[k], "ci95": _ci(cis[k])} for k in point},
        "reliability_v_plus": reliability_table(sc["v_plus"], y),
        "held_out_logistic": held_out_logistic(rows, y, game, sc["v_plus"]),
    }


def held_out_logistic(
    rows: Rows, y: NDArray[Any], game: NDArray[Any], v_plus: NDArray[Any]
) -> dict[str, Any]:
    """win ~ lead + own VP + turn, with and without V+, fitted on odd games and
    scored on even games; ΔAUC bootstrapped over the even games."""
    base = np.column_stack([rows["lead"], rows["vp_own"], rows["turn"]])
    full = np.column_stack([base, v_plus])
    train, test = game % 2 == 1, game % 2 == 0
    xb_tr, xb_te = standardize(base[train], base[test])
    xf_tr, xf_te = standardize(full[train], full[test])
    p_base = logistic_predict(xb_te, irls_logistic(xb_tr, y[train]))
    p_full = logistic_predict(xf_te, irls_logistic(xf_tr, y[train]))
    y_te, g_te = y[test], game[test]

    def stat(idx: NDArray[np.intp]) -> dict[str, float]:
        a_base, a_full = auc(p_base[idx], y_te[idx]), auc(p_full[idx], y_te[idx])
        return {
            "auc_base": a_base,
            "auc_with_v_plus": a_full,
            "delta_auc": a_full - a_base,
        }

    point = stat(np.arange(len(y_te)))
    cis = cluster_bootstrap(stat, g_te, n_boot=N_BOOT_CLUSTER, seed=BOOT_SEED + 1)
    return {"metrics": {k: {"value": point[k], "ci95": _ci(cis[k])} for k in point}}


def offer_probe(rows: Rows) -> dict[str, Any]:
    dv = rows["offer_dv"]
    if len(dv) == 0:
        return {"n_offers": 0}
    drop = rows["offer_shortfall_drop"]
    shape = rows["offer_cards_out"] > rows["offer_cards_in"]
    return {
        "n_offers": int(len(dv)),
        "n_games_with_offer": int(len(np.unique(rows["offer_game"]))),
        "dv": {
            "quantiles": quantiles(dv),
            "share_positive": float((dv > 0).mean()),
            "mean": float(dv.mean()),
        },
        "dv_gift_receive_without_paying": {
            "quantiles": quantiles(rows["offer_dv_gift"]),
            "share_positive": float((rows["offer_dv_gift"] > 0).mean()),
        },
        "dv_pay_without_receiving": {
            "quantiles": quantiles(rows["offer_dv_pay"]),
            "share_positive": float((rows["offer_dv_pay"] > 0).mean()),
        },
        "v_pending_minus_before": quantiles(rows["offer_v_pending_minus_before"]),
        "dv_by_cards_out_vs_in": {
            "pays_more_than_receives": _summ(dv[shape]),
            "pays_no_more_than_receives": _summ(dv[~shape]),
        },
        "dv_by_shortfall_drop": {
            label: _summ(dv[mask])
            for label, mask in (
                ("drop_gt_0", drop > 0),
                ("drop_eq_0", drop == 0),
                ("drop_lt_0", drop < 0),
                ("no_build_target", np.isnan(drop)),
            )
        },
    }


def _summ(x: NDArray[Any]) -> dict[str, float]:
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "share_positive": float((x > 0).mean()),
    }


def fork_analysis(rows: Rows) -> dict[str, Any]:
    """ΔV at each game's sampled offer vs the paired playout win difference.
    One offer per game, so plain resampling of offers is valid."""
    if len(rows["fork_game"]) == 0:
        return {"n_forks": 0}
    dv = rows["offer_dv"][rows["fork_offer_row"]]
    dw = rows["fork_dwin"]
    n = len(dv)

    def stat(idx: NDArray[np.intp]) -> dict[str, float]:
        pos = dv[idx] > 0
        both = bool(pos.any() and (~pos).any())
        return {
            "spearman_dv_dwin": spearman(dv[idx], dw[idx]),
            "mean_dwin_dv_pos_minus_nonpos": (
                float(dw[idx][pos].mean() - dw[idx][~pos].mean())
                if both
                else float("nan")
            ),
        }

    point = stat(np.arange(n))
    cis = _boot_rows(stat, n, BOOT_SEED + 2, N_BOOT_HEADLINE)
    edges = np.quantile(dv, np.linspace(0, 1, 6))
    quint = np.clip(np.searchsorted(edges, dv, side="right") - 1, 0, 4)
    rng = np.random.default_rng(BOOT_SEED + 3)
    quintiles = []
    for q in range(5):
        sel = quint == q
        if not sel.any():
            continue
        boot = [
            dw[sel][rng.integers(0, sel.sum(), sel.sum())].mean() for _ in range(1000)
        ]
        quintiles.append(
            {
                "quintile": q,
                "n": int(sel.sum()),
                "dv_mean": float(dv[sel].mean()),
                "mean_dwin": float(dw[sel].mean()),
                "mean_dwin_ci95": [
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                ],
            }
        )
    margins = np.quantile(dv, np.linspace(0, 0.95, 20))
    return {
        "n_forks": int(n),
        "mean_dwin_all": float(dw.mean()),
        "metrics": {k: {"value": point[k], "ci95": cis[k]} for k in point},
        "dwin_by_dv_quintile": quintiles,
        "margin_curve_exploratory": [
            {
                "margin": float(m),
                "share_accepted": float((dv > m).mean()),
                "sum_dwin_over_all_offers": float((dw * (dv > m)).mean()),
                "mean_dwin_of_accepted": float(dw[dv > m].mean())
                if (dv > m).any()
                else None,
            }
            for m in margins
        ],
    }


def verdicts(res: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable pre-registered rules (log 009, Verdict rules)."""
    out: dict[str, Any] = {}
    h = res["headline"]["metrics"]["auc_v_plus"]["ci95"][0]
    out["v_ranks_positions"] = bool(h > 0.5)
    a = res["all_states"]["metrics"]["stratified_auc_v_plus_own_vp_x_lead"]["ci95"][0]
    d = res["all_states"]["held_out_logistic"]["metrics"]["delta_auc"]["ci95"]
    out["v_more_than_vp_counter"] = bool(a > 0.5 and d[0] > 0)
    f = res.get("fork", {})
    if f.get("n_forks"):
        m = f["metrics"]
        out["go_for_38"] = bool(
            m["spearman_dv_dwin"]["ci95"][0] > 0
            and m["mean_dwin_dv_pos_minus_nonpos"]["ci95"][0] > 0
        )
    return out


def analyze(rows: Rows, *, arm: str) -> dict[str, Any]:
    res: dict[str, Any] = {
        "headline": headline(rows),
        "all_states": all_states(rows),
    }
    if arm == "vs_trading_heuristic":
        res["offer_probe"] = offer_probe(rows)
        res["fork"] = fork_analysis(rows)
    res["verdicts"] = verdicts(res)
    return res
