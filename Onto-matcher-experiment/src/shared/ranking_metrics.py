"""
Per-JD ranking metrics.

main function: compute_ranking_metrics(df, k_values)

"""

from __future__ import annotations

import numpy as np
import pandas as pd

NEUTRAL = "neutral"
MALE    = "male"
FEMALE  = "female"


# compute dcg
def _dcg_tie_aware(predicted: np.ndarray, gt: np.ndarray) -> float:

    n = len(predicted)
    if n == 0:
        return np.nan

    # get rank of predicted scores. highest is best
    order = np.argsort(-predicted, kind="stable")
    sorted_pred = predicted[order]
    sorted_gt   = gt[order]

    dcg = 0.0
    rank = 1
    i = 0
    while i < n:
        j = i
        #get size of tied block
        while j < n and sorted_pred[j] == sorted_pred[i]:
            j += 1
        block_size = j - i

        # compute discounts for tied ranks
        ranks = np.arange(rank, rank + block_size)
        discounts = 1.0 / np.log2(ranks + 1)

        # compute mean gain for tied block. 
        if len(discounts) > 0:
            mean_discount = discounts.mean()
            dcg += sorted_gt[i:j].sum() * mean_discount

        rank += block_size
        i = j

    return float(dcg)


def _ideal_dcg(gt: np.ndarray) -> float:
    gt = gt[pd.notna(gt)]
    n = len(gt)
    if n == 0:
        return np.nan

    #highest rank is best.
    sorted_gt = np.sort(gt)[::-1]
    ranks = np.arange(1, len(sorted_gt) + 1)

    # sum gains for all ranks using the log2 discounting factor
    return float(np.sum(sorted_gt / np.log2(ranks + 1)))


#compute normalized discounted cumulative gain (NDCG) for a single JD (query)
def ndcg(predicted: np.ndarray, gt: np.ndarray, k: int) -> float:
    if len(predicted) == 0:
        return np.nan
    idcg = _ideal_dcg(gt)
    if idcg == 0 or np.isnan(idcg):
        return np.nan
    return _dcg_tie_aware(predicted, gt) / idcg



def expected_inclusion_weights(scores: np.ndarray, k: int) -> np.ndarray:
    n = len(scores)
    weights = np.zeros(n, dtype=float)
    order = np.argsort(-scores, kind="stable")
    t = 0  # candidates ranked so far
    i = 0
    while i < n and t < k:
        j = i
        #get tied block
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        block_size = j - i
        t_next = t + block_size

        #compute tied block inclusion weight
        w = max(0.0, min(t_next, k) - t) / block_size
        for idx in order[i:j]:
            weights[idx] = w
        t = t_next
        i = j
    return weights

#compute precision at k for a single JD (query)
def precision_at_k(predicted: np.ndarray, gt: np.ndarray, k: int) -> float:
    if len(predicted) == 0 or k == 0:
        return np.nan
    k = min(k, len(predicted))
    w_pred = expected_inclusion_weights(predicted, k)
    w_gt   = expected_inclusion_weights(gt, k)
    return float(np.sum(w_pred * w_gt) / k)


def mae(predicted: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - gt))) if len(predicted) > 0 else np.nan

#calculate per query RMSE
def rmse(predicted: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - gt) ** 2))) if len(predicted) > 0 else np.nan


#get gt ranks withing a jd 
def _gt_rank_within_jd(gt: np.ndarray) -> np.ndarray:
    return pd.Series(gt).rank(method="average", ascending=True).to_numpy(dtype=float)

# compute all metrics for a single JD (query) and return as a dict
def _performance_row(jd_df: pd.DataFrame, perf_k: list[int]) -> dict:
    pred    = jd_df["score"].to_numpy(dtype=float)
    gt      = jd_df["gt_score"].to_numpy(dtype=float)
    gt_rank = jd_df["_gt_rank"].to_numpy(dtype=float)
    row  = {"mae": mae(pred, gt), "rmse": rmse(pred, gt), "ndcg@full": ndcg(pred, gt_rank, len(pred))}
    for k in perf_k:
        row[f"precision@{k}"] = precision_at_k(pred, gt, k)
    return row

# compute bias deltas
def _bias_row(jd_df: pd.DataFrame, bias_k: list[int],
              cf_col: str) -> dict:
    row    = {}
    #split df into groups
    groups = {g: jd_df[jd_df[cf_col] == g] for g in (NEUTRAL, MALE, FEMALE)}

    # define delta pairs (label, a_group, b_group) → metric = a − b
    comparisons = [
        (f"male_vs_neutral",   MALE,   NEUTRAL),
        (f"female_vs_neutral", FEMALE, NEUTRAL),
        (f"female_vs_male",    FEMALE, MALE),
    ]

    for tag, a_name, b_name in comparisons:
        a, b = groups[a_name], groups[b_name]
        if a.empty or b.empty:
            row[f"Δrmse_{tag}"] = np.nan
            row[f"Δndcg@full_{tag}"] = np.nan
            for k in bias_k:
                row[f"Δprecision@{k}_{tag}"] = np.nan
            continue

        # get groups scores, gt_scores and ranks
        a_pred, a_gt = a["score"].to_numpy(dtype=float), a["gt_score"].to_numpy(dtype=float)
        b_pred, b_gt = b["score"].to_numpy(dtype=float), b["gt_score"].to_numpy(dtype=float)
        a_gt_rank, b_gt_rank = a["_gt_rank"].to_numpy(dtype=float), b["_gt_rank"].to_numpy(dtype=float)

        # add performance delta rows
        row[f"Δrmse_{tag}"] = rmse(a_pred, a_gt) - rmse(b_pred, b_gt)   
        a_ndcg = ndcg(a_pred, a_gt_rank, len(a_pred))
        b_ndcg = ndcg(b_pred, b_gt_rank, len(b_pred))
        row[f"Δndcg@full_{tag}"] = (a_ndcg - b_ndcg) if not (np.isnan(a_ndcg) or np.isnan(b_ndcg)) else np.nan
        for k in bias_k:
            a_prec = precision_at_k(a_pred, a_gt, k)
            b_prec = precision_at_k(b_pred, b_gt, k)
            row[f"Δprecision@{k}_{tag}"] = (a_prec - b_prec) if not (np.isnan(a_prec) or np.isnan(b_prec)) else np.nan

    return row

#bootstrap ci intervals
def _bootstrap_ci(
    vals: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    rng = None,
) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(vals) < 2:
        m = float(vals.mean()) if len(vals) == 1 else np.nan
        return m, m
    boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


# compute aggregate of queries
def _aggregate(per_jd_rows: list[dict], n_boot: int = 10000) -> dict:
    df = pd.DataFrame(per_jd_rows).select_dtypes(include="number")
    rng = np.random.default_rng(0)
    result = {}
    # get mean and bootstrap ci for each metric across all queries
    for col in df.columns:
        vals = df[col].dropna().to_numpy(dtype=float)
        mean = float(vals.mean()) if len(vals) > 0 else np.nan
        lo, hi = _bootstrap_ci(vals, n_boot=n_boot, rng=rng)
        result[col] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(vals))}
    return result


def compute_ranking_metrics(
    df: pd.DataFrame,
    k_values: list[int],
    bias_k: list[int] | None = None,
    jd_col: str = "jd_id",
    cf_col: str = "counterfactual",
) -> dict:

    if bias_k is None:
        bias_k = k_values

    has_cf = cf_col in df.columns

    # collect per jd pair rows (each query)
    per_jd_rows = []
    for _, jd_df in df.groupby(jd_col):
        #compute gt rank aling with gt_score
        jd_df = jd_df.assign(_gt_rank=_gt_rank_within_jd(jd_df["gt_score"].to_numpy(dtype=float)))

        #get performance per jd and bias deltas per jd
        row = _performance_row(jd_df, k_values)
        if has_cf:
            row.update(_bias_row(jd_df, bias_k, cf_col))
        per_jd_rows.append(row)

    #aggreate across all queried
    return _aggregate(per_jd_rows)





# put metrics into a human-readable string for logging
def format_metrics(
    agg: dict,   #aggregated metrics dict from compute_ranking_metrics
    run_name: str,
    condition: str,
    k_values: list[int],
    n_jds: int,
    n_pairs: int,
    bias_k: list[int] | None = None,
) -> str:
    if bias_k is None:
        bias_k = k_values

    #run overview 
    lines = [
        "=" * 60,
        f"  Run       : {run_name}  (condition {condition})",
        f"  Queries   : {n_jds} JDs   |   Pairs: {n_pairs}",
        f"  Relevant  : top-k GT CVs per JD (n_relevant = k)",
        f"  perf k    : {k_values}   bias k: {bias_k}",
        "=" * 60,
    ]

    # format lines
    def _section(title: str, keys: list[str]) -> None:
        lines.append(f"\n  {title}")
        lines.append(f"  {'─' * 50}")
        for k in keys:
            if k not in agg:
                continue
            e = agg[k]
            lines.append(f"  {k:<36}  {e['mean']:>7.4f} [{e['ci_lo']:.4f}, {e['ci_hi']:.4f}]  (n={e['n']})")

    #preformance metrics
    perf_keys = ["rmse", "mae", "ndcg@full"] + [f"precision@{k}" for k in k_values]
    _section("PERFORMANCE  (mean ± 95% CI half-width across JDs)", perf_keys)

    #boas metrics
    comparisons = [
        ("male_vs_neutral",   "male − neutral"),
        ("female_vs_neutral", "female − neutral"),
        ("female_vs_male",    "female − male"),
    ]
    for tag, label in comparisons:
        bias_keys = [f"Δrmse_{tag}", f"Δndcg@full_{tag}"] + [f"Δprecision@{k}_{tag}" for k in bias_k]
        _section(f"BIAS  {label}  (mean ± 95% CI)", bias_keys)

    return "\n".join(lines)
