#compute metrics for analysis

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Threshold specification
# ============================================================================

@dataclass
class ThresholdSpec:
    """
    Specifies how to convert a continuous score column into a selection indicator.
    """

    mode: Literal["absolute", "top_k", "top_fraction"]
    value: float
    tie_method: Literal["probabilistic", "include", "exclude"] = "probabilistic"

    def describe(self) -> str:
        if self.mode == "absolute":
            return f"score >= {self.value}"
        if self.mode == "top_k":
            return f"top {int(self.value)} ({self.tie_method} ties)"
        return f"top {self.value*100:.1f}% ({self.tie_method} ties)"


def apply_threshold(scores: pd.Series, spec: ThresholdSpec, query: pd.Series | None = None) -> pd.Series:
    """
    Convert a score Series to selection probabilities in [0, 1].

    For mode="absolute" this is always 0 or 1.
    For mode="top_k" / "top_fraction" with tie_method="probabilistic",
    tied candidates at the cutoff receive a fractional value.
    """
    scores = pd.Series(scores).astype(float)

    if spec.mode == "absolute":
        return (scores >= spec.value).astype(float)

    if query is not None:
        if spec.tie_method == "probabilistic" and spec.mode in ("top_k", "top_fraction"):
            return _topk_probabilities_grouped(scores, query, spec)
        # Fallback for "include"/"exclude" tie methods (not used in this pipeline's
        # config, kept for completeness) — per-group loop.
        query = pd.Series(query).reindex(scores.index)
        result = pd.Series(0.0, index=scores.index)
        for _, idx in scores.groupby(query).groups.items():
            result.loc[idx] = apply_threshold(scores.loc[idx], spec)
        return result

    n = len(scores)
    if n == 0:
        return pd.Series(dtype=float)

    if spec.mode == "top_k":
        k = min(max(int(spec.value), 0), n)
    else:  # top_fraction
        k = min(max(round(spec.value * n), 0), n)

    if spec.tie_method == "probabilistic":
        return _topk_probabilities(scores, k)

    # Hard tie methods — sort descending, find cutoff score
    sorted_scores = scores.sort_values(ascending=False)
    cutoff_score = sorted_scores.iloc[k - 1] if k > 0 else np.inf

    if spec.tie_method == "include":
        return (scores >= cutoff_score).astype(float)
    else:  # exclude
        return (scores > cutoff_score).astype(float)


# ============================================================================
# Private helpers
# ============================================================================

def _validate_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _safe_divide(num: float, denom: float) -> float:
    if denom == 0 or pd.isna(denom):
        return np.nan
    return num / denom


def _group_scores(
    df: pd.DataFrame,
    group: str,
    score_col: str,
    group_col: str,
) -> pd.Series:
    return df.loc[df[group_col] == group, score_col].dropna().astype(float)


def _paired_wide(
    df: pd.DataFrame,
    value_col: str,
    pair_col: str,
    group_col: str,
    groups: tuple[str, str],
) -> pd.DataFrame:
    """
    Pivot df so each row is one pair, columns are the two group values.
    Drops pairs that don't have both groups present.
    """
    sub = df[df[group_col].isin(groups)][[pair_col, group_col, value_col]]
    wide = sub.pivot_table(index=pair_col, columns=group_col, values=value_col, aggfunc="mean")
    missing = [g for g in groups if g not in wide.columns]
    if missing:
        raise ValueError(f"Groups not found in data: {missing}")
    return wide[list(groups)].dropna()



def _topk_probabilities(scores: pd.Series, k: int) -> pd.Series:
    """
    Tie-aware top-k inclusion probabilities.
    Tied candidates at the cutoff boundary each receive remaining_slots / n_tied.
    """
    scores = pd.Series(scores).astype(float)
    n = len(scores)
    k = min(max(k, 0), n)

    result = pd.Series(0.0, index=scores.index)
    if k == 0:
        return result

    counts = scores.value_counts(dropna=False).sort_index(ascending=False)
    num_above = 0
    for score_val, block_size in counts.items():
        idx = scores.index[scores == score_val]
        if num_above + block_size <= k:
            result.loc[idx] = 1.0
        elif num_above < k:
            result.loc[idx] = (k - num_above) / block_size
        # else: 0.0 (default)
        num_above += block_size

    return result


def _topk_probabilities_grouped(scores: pd.Series, query: pd.Series, spec: "ThresholdSpec") -> pd.Series:

    scores = pd.Series(scores).astype(float)
    query = pd.Series(query).reindex(scores.index)
    tmp = pd.DataFrame({"q": query.to_numpy(), "s": scores.to_numpy()}, index=scores.index)

    group_size = tmp.groupby("q")["s"].transform("size")
    if spec.mode == "top_k":
        k_group = np.minimum(float(spec.value), group_size)
    else:  # top_fraction
        k_group = np.minimum((spec.value * group_size).round(), group_size)

    rank_desc = tmp.groupby("q")["s"].rank(method="min", ascending=False)
    num_above = rank_desc - 1
    block_size = tmp.groupby(["q", "s"])["s"].transform("size")

    fully_in = (num_above + block_size) <= k_group
    partially_in = (~fully_in) & (num_above < k_group)
    prob = np.where(
        fully_in, 1.0,
        np.where(partially_in, (k_group - num_above) / block_size, 0.0),
    )
    return pd.Series(prob, index=scores.index)


def gt_rank_within_query(gt: pd.Series, query: pd.Series) -> pd.Series:
    """Within-JD average rank of a GT score
    """
    return pd.Series(gt).groupby(query).rank(method="average", ascending=True)



# ============================================================================
# 3-way group delta functions (female / male / neutral)
# ============================================================================

def ndcg_delta(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    k: int = 5,
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    group_col: str = "counterfactual",
    query_col: str = "jd_id",
) -> float:
    """avg_ndcg_at_k_per_jd(group_a) - avg_ndcg_at_k_per_jd(group_b)."""
    _validate_cols(df, [score_col, gt_col, group_col, query_col])
    v_a = avg_ndcg_at_k_per_jd(df[df[group_col] == group_a], k=k, score_col=score_col,
                                gt_col=gt_col, query_col=query_col)
    v_b = avg_ndcg_at_k_per_jd(df[df[group_col] == group_b], k=k, score_col=score_col,
                                gt_col=gt_col, query_col=query_col)
    return float(v_a - v_b)


def rmse_delta(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    group_col: str = "counterfactual",
    query_col: str = "jd_id",
) -> float:
    """rmse_per_jd(group_a) - rmse_per_jd(group_b). Positive = group_a has HIGHER error."""
    _validate_cols(df, [score_col, gt_col, group_col, query_col])
    v_a = rmse_per_jd(df[df[group_col] == group_a], score_col=score_col, gt_col=gt_col, query_col=query_col)
    v_b = rmse_per_jd(df[df[group_col] == group_b], score_col=score_col, gt_col=gt_col, query_col=query_col)
    return float(v_a - v_b)


def precision_delta(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    pred_threshold: "ThresholdSpec",
    gt_threshold: "ThresholdSpec",
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    group_col: str = "counterfactual",
    query_col: str = "jd_id",
) -> float:
    """Precision(group_a) - Precision(group_b).
    """
    sub_a = df[df[group_col] == group_a]
    sub_b = df[df[group_col] == group_b]
    return (precision(sub_a, pred_threshold, gt_threshold, score_col, gt_col, query_col)
            - precision(sub_b, pred_threshold, gt_threshold, score_col, gt_col, query_col))



def rmse_per_jd(
    df: pd.DataFrame,
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    query_col: str = "jd_id",
) -> float:
    """Mean of per-JD RMSE
    """
    _validate_cols(df, [score_col, gt_col, query_col])
    sub = df[[query_col, score_col, gt_col]].dropna()
    if len(sub) == 0:
        return np.nan
    sq_err = (sub[score_col].astype(float) - sub[gt_col].astype(float)) ** 2
    per_jd_rmse = np.sqrt(sq_err.groupby(sub[query_col]).mean())
    return float(per_jd_rmse.mean()) if len(per_jd_rmse) else np.nan



def _cum_discount_table(max_rank: int) -> np.ndarray:
    """cum[r] = sum_{i=1}^{r} 1/log2(i+1) for r >= 1, cum[0] = 0."""
    ranks = np.arange(1, max_rank + 1)
    return np.concatenate([[0.0], np.cumsum(1.0 / np.log2(ranks + 1))])


def avg_ndcg_at_k_per_jd(
    df: pd.DataFrame,
    k: int = 15,
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    query_col: str = "jd_id",
    group_col: str = "occ_code", 
) -> float:
    """
    Mean NDCG@k where each JD is a query ranking its own candidates — tie-aware 
    """
    _validate_cols(df, [score_col, gt_col, query_col])
    sub = df[[query_col, score_col, gt_col]].dropna().copy()
    if len(sub) == 0:
        return np.nan
    sub[score_col] = sub[score_col].astype(float)
    sub[gt_col] = sub[gt_col].astype(float)

    max_size = int(sub.groupby(query_col).size().max())
    cum = _cum_discount_table(max(max_size, k))

    def discount_sum(start_rank: np.ndarray, end_rank: np.ndarray) -> np.ndarray:
        # Sum of 1/log2(r+1) for r in [start_rank, end_rank], clipped to k.
        end_c = np.clip(np.minimum(end_rank, k), 0, len(cum) - 1)
        start_c = np.clip(np.minimum(start_rank - 1, k), 0, len(cum) - 1)
        return cum[end_c] - cum[start_c]

    # ── DCG per JD ──
    rank_desc = sub.groupby(query_col)[score_col].rank(method="min", ascending=False)
    block_size = sub.groupby([query_col, score_col])[score_col].transform("size")
    start_rank = rank_desc.to_numpy().astype(int)
    end_rank = start_rank + block_size.to_numpy().astype(int) - 1
    disc_sum = discount_sum(start_rank, end_rank)

    rel_mean_in_block = sub.groupby([query_col, score_col])[gt_col].transform("mean")
    row_contrib = rel_mean_in_block.to_numpy() * disc_sum / block_size.to_numpy()
    dcg_per_jd = pd.Series(row_contrib, index=sub.index).groupby(sub[query_col]).sum()

    # ── Ideal DCG per JD (sort by GT descending
    gt_rank = sub.groupby(query_col)[gt_col].rank(method="first", ascending=False).to_numpy().astype(int)
    gt_rank_c = np.clip(gt_rank, 0, len(cum) - 1)
    gt_rank_prev_c = np.clip(gt_rank_c - 1, 0, len(cum) - 1)
    idcg_row = np.where(gt_rank <= k, sub[gt_col].to_numpy() * (cum[gt_rank_c] - cum[gt_rank_prev_c]), 0.0)
    idcg_per_jd = pd.Series(idcg_row, index=sub.index).groupby(sub[query_col]).sum()

    ndcg_per_jd = (dcg_per_jd / idcg_per_jd).where(idcg_per_jd > 0)
    return float(ndcg_per_jd.dropna().mean()) if ndcg_per_jd.notna().any() else np.nan


def precision(
    df: pd.DataFrame,
    pred_threshold: ThresholdSpec,
    gt_threshold: ThresholdSpec,
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    query_col: str = "jd_id",
) -> float:
    """Precision: of those predicted selected, fraction who are truly qualified.
    """
    _validate_cols(df, [score_col, gt_col, query_col])
    sub = df[[score_col, gt_col, query_col]].dropna()
    sel = apply_threshold(sub[score_col], pred_threshold, query=sub[query_col])
    qual = apply_threshold(sub[gt_col], gt_threshold, query=sub[query_col])
    denom = sel.sum()
    return float(_safe_divide((sel * qual).sum(), denom))



# ============================================================================
# Confidence intervals
# ============================================================================

def _bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable,
    metric_kwargs: dict,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    cluster_col: str | None = None,
) -> tuple[float, float]:
    """
    Bootstrap CI for a metric.
    cluster_col : if given (e.g. "jd_id"), performs a cluster bootstrap
    """
    rng = np.random.default_rng(seed)
    values = []

    if cluster_col is not None and cluster_col in df.columns:
        cluster_ids = df[cluster_col].dropna().unique()
        n_clusters = len(cluster_ids)
        idx_by_cluster = df.groupby(cluster_col).indices  # {cluster_id: positional row indices}
        for _ in range(n_bootstrap):
            chosen = rng.choice(cluster_ids, size=n_clusters, replace=True)
            row_idx = np.concatenate([idx_by_cluster[cid] for cid in chosen])
            draw_labels = np.concatenate([np.full(len(idx_by_cluster[cid]), i) for i, cid in enumerate(chosen)])
            sample = df.iloc[row_idx].copy()
            sample[cluster_col] = sample[cluster_col].astype(str).to_numpy() + "__" + draw_labels.astype(str)
            try:
                v = metric_fn(sample, **metric_kwargs)
                if v is not None and not np.isnan(v):
                    values.append(v)
            except Exception:
                continue
    else:
        n = len(df)
        for _ in range(n_bootstrap):
            sample = df.iloc[rng.choice(n, size=n, replace=True)].reset_index(drop=True)
            try:
                v = metric_fn(sample, **metric_kwargs)
                if v is not None and not np.isnan(v):
                    values.append(v)
            except Exception:
                continue

    if len(values) < max(10, n_bootstrap * 0.1):
        return np.nan, np.nan
    alpha = 1 - confidence
    return float(np.percentile(values, alpha / 2 * 100)), float(np.percentile(values, (1 - alpha / 2) * 100))


def _normal_ci(
    values: pd.Series,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Normal approximation CI for the mean of `values`. NaNs are dropped."""
    v = pd.Series(values).dropna().astype(float)
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = v.std(ddof=1) / np.sqrt(n)
    mean = v.mean()
    return float(mean - z * se), float(mean + z * se)


# ============================================================================
# Orchestrator
# ============================================================================

def compute_metrics(
    df: pd.DataFrame,
    metrics_cfg: dict[str, dict],
    pred_threshold: ThresholdSpec,
    gt_threshold: ThresholdSpec,
    pred_threshold_delta: ThresholdSpec | None = None,
    gt_threshold_delta: ThresholdSpec | None = None,
    group_a: str = "female",
    group_b: str = "male",
    group_neutral: str = "neutral",
    score_col: str = "score",
    gt_col: str = "comp_gt_score_1_10",
    group_col: str = "counterfactual",
    pair_col: str = "group_id",
    query_col: str = "jd_id",
    k: int = 50,
    n_bootstrap: int = 1000,
    ci_confidence: float = 0.95,
    ci_seed: int = 42,
) -> dict:
    """
    Returns:
        Nested dict:
        {
            "metadata": { ... },
            "metrics": {
                "impact_ratio": {"value": float, "ci_lower": float, "ci_upper": float, "ci_mode": str},
                "rmse_by_group": {
                    "female": {"value": ..., "ci_lower": ..., "ci_upper": ..., "ci_mode": ...},
                    "male":   { ... },
                    "gap":    { ... },
                },
                ...
            }
        }
    """
    pred_threshold_delta = pred_threshold_delta or pred_threshold
    gt_threshold_delta = gt_threshold_delta or gt_threshold

    # NDCG's gain is the within-JD RANK of the GT score
    GT_RANK_COL = "_gt_rank_per_jd"
    if gt_col in df.columns and query_col in df.columns:
        df = df.assign(**{GT_RANK_COL: gt_rank_within_query(df[gt_col], df[query_col])})
    else:
        df = df.assign(**{GT_RANK_COL: np.nan})

    results: dict = {
        "metadata": {
            "n_records":      len(df),
            "group_a":        group_a,
            "group_b":        group_b,
            "pred_threshold": pred_threshold.describe(),
            "gt_threshold":   gt_threshold.describe(),
            "pred_threshold_delta": pred_threshold_delta.describe(),
            "gt_threshold_delta":   gt_threshold_delta.describe(),
            "k":              k,
            "ci_confidence":  ci_confidence,
            "n_bootstrap":    n_bootstrap,
        },
        "metrics": {},
    }

    m = results["metrics"]

    def _cfg(name: str) -> dict:
        return metrics_cfg.get(name, {"enabled": False, "ci": None})

    def _ci(name: str, fn: Callable, kwargs: dict, values_for_normal: pd.Series | None = None):
        """Compute CI for a metric according to its configured ci mode."""
        cfg = _cfg(name)
        ci_mode = cfg.get("ci")
        if ci_mode == "bootstrap":
            return _bootstrap_ci(df, fn, kwargs, n_bootstrap, ci_confidence, ci_seed, cluster_col=query_col), "bootstrap"
        if ci_mode == "normal" and values_for_normal is not None:
            return _normal_ci(values_for_normal, ci_confidence), "normal"
        return (np.nan, np.nan), None

    def _entry(value: float, ci_bounds: tuple, ci_mode: str | None) -> dict:
        lo, hi = ci_bounds
        return {"value": value, "ci_lower": lo, "ci_upper": hi, "ci_mode": ci_mode}


    # ── Ranking fairness ─────────────────────────────────────────────────────


    # ── 3-way group deltas (female / male / neutral) ─────────────────────────
    # Each of ndcg/rmse/precision gets 3 signed deltas: female-neutral, male-neutral,
    # female-male. Metric key format: "{metric}_delta_{group_a}_{group_b}".
    _delta_pairs = ((group_a, group_neutral), (group_b, group_neutral), (group_a, group_b))
    _delta_specs = {
        "ndcg": (ndcg_delta, dict(k=k, score_col=score_col, gt_col=GT_RANK_COL,
                                   group_col=group_col, query_col=query_col)),
        "rmse": (rmse_delta, dict(score_col=score_col, gt_col=gt_col,
                                   group_col=group_col, query_col=query_col)),
        "precision": (precision_delta, dict(pred_threshold=pred_threshold_delta, gt_threshold=gt_threshold_delta,
                                             score_col=score_col, gt_col=gt_col,
                                             group_col=group_col, query_col=query_col)),
    }
    _delta_values: dict[str, float] = {}
    for _metric_key, (fn, _base_kw) in _delta_specs.items():
        for ga, gb in _delta_pairs:
            _name = f"{_metric_key}_delta_{ga}_{gb}"
            if not _cfg(_name)["enabled"]:
                continue
            kw = dict(_base_kw, group_a=ga, group_b=gb)
            v = _try(fn, df, kw)
            _ci_bounds, _mode = _ci(_name, fn, kw)
            m[_name] = _entry(v, _ci_bounds, _mode)
            _delta_values[_name] = v

    # Consistency guard: delta(a,b) == delta(a,c) - delta(b,c) by construction
    for _metric_key in _delta_specs:
        fm = f"{_metric_key}_delta_{group_a}_{group_b}"
        fn_ = f"{_metric_key}_delta_{group_a}_{group_neutral}"
        mn = f"{_metric_key}_delta_{group_b}_{group_neutral}"
        if fm in _delta_values and fn_ in _delta_values and mn in _delta_values:
            lhs = _delta_values[fm]
            rhs = _delta_values[fn_] - _delta_values[mn]
            if not (np.isnan(lhs) or np.isnan(rhs)) and abs(lhs - rhs) > 1e-6:
                print(f"[WARNING] consistency check failed for {_metric_key}: "
                      f"{fm}={lhs:.6f} != {fn}-{mn}={rhs:.6f} (diff {abs(lhs - rhs):.2e})")

    # ── Performance (overall) ────────────────────────────────────────────────

    if _cfg("rmse")["enabled"]:
        # Per-JD averaged (rmse_per_jd)
        fn = rmse_per_jd
        kw = dict(score_col=score_col, gt_col=gt_col, query_col=query_col)
        v = _try(fn, df, kw)
        ci, mode = _ci("rmse", fn, kw)
        m["rmse"] = _entry(v, ci, mode)


    
    if _cfg("avg_ndcg_at_k_per_jd")["enabled"]:
        fn = avg_ndcg_at_k_per_jd
        kw = dict(k=k, score_col=score_col, gt_col=GT_RANK_COL, query_col="jd_id", group_col="occ_code")
        v = _try(fn, df, kw)
        ci, mode = _ci("avg_ndcg_at_k_per_jd", fn, kw)
        m["avg_ndcg_at_k_per_jd"] = _entry(v, ci, mode)


    if _cfg("precision")["enabled"]:
        fn = precision
        kw = dict(pred_threshold=pred_threshold, gt_threshold=gt_threshold,
                  score_col=score_col, gt_col=gt_col, query_col=query_col)
        v = _try(fn, df, kw)
        ci, mode = _ci("precision", fn, kw)
        m["precision"] = _entry(v, ci, mode)


    return results


def _try(fn: Callable, df: pd.DataFrame, kwargs: dict) -> float:
    """Call fn(df, **kwargs), return nan on any error."""
    try:
        return fn(df, **kwargs)
    except Exception:
        return np.nan


# ============================================================================
# Save / load results
# ============================================================================

def save_results(results: dict, path) -> None:
    """
    Save compute_metrics() output to JSON.
    """
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(path, "w") as fh:
        json.dump(_clean(results), fh, indent=2)


def flatten_results(results: dict) -> dict:
    """
    Flatten nested compute_metrics() output into a single-level dict
    """
    flat = dict(results["metadata"])
    for name, entry in results["metrics"].items():
        if "value" in entry:
            flat[name]              = entry["value"]
            flat[f"{name}_ci_lo"]  = entry["ci_lower"]
            flat[f"{name}_ci_hi"]  = entry["ci_upper"]
        else:
            # by-group sub-dict
            for sub_name, sub_entry in entry.items():
                key = f"{name}_{sub_name}"
                flat[key]              = sub_entry["value"]
                flat[f"{key}_ci_lo"]  = sub_entry["ci_lower"]
                flat[f"{key}_ci_hi"]  = sub_entry["ci_upper"]
    return flat
