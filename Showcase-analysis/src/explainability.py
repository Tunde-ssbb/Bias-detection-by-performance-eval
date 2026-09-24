"""
explainability.py

Fits LLM scores using row-wise post-processed features to explain score variance through OLS.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Feature preparation ────────────────────────────────────────────────────────

def _prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    score_col: str,
    cluster_col: str | None = None,
    reference_categories: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """
    Build design matrix X and target vector y.
    """
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[explain] WARNING: feature columns not found and skipped: {missing}")

    # Drop only rows where the score itself is missing, then fill nan with 0
    cols_needed = [score_col] + present + ([cluster_col] if cluster_col else [])
    sub = df[cols_needed].dropna(subset=[score_col])
    y = sub[score_col].astype(float).to_numpy()
    clusters = sub[cluster_col].to_numpy() if cluster_col else None

    parts: list[pd.DataFrame] = []
    feature_names: list[str] = []

    for col in present:
        series = sub[col]
        if pd.api.types.is_numeric_dtype(series):
            s = series.fillna(0).astype(float)
            std = s.std()
            if std == 0:
                continue  # zero-variance: drop silently
            parts.append(((s - s.mean()) / std).to_frame())
            feature_names.append(col)
        else:
            dummies = pd.get_dummies(series, prefix=col, dtype=float)  # all levels, none dropped yet
            requested_ref = (reference_categories or {}).get(col)
            ref_col = f"{col}_{requested_ref}" if requested_ref is not None else None
            if ref_col is None or ref_col not in dummies.columns:
                ref_col = sorted(dummies.columns)[0]  # fallback: alphabetically-first level
            dummies = dummies.drop(columns=[ref_col])
            # Standardise dummies too so their scale matches numeric features
            for dcol in dummies.columns:
                std = dummies[dcol].std()
                if std == 0:
                    continue
                dummies[dcol] = (dummies[dcol] - dummies[dcol].mean()) / std
            parts.append(dummies.fillna(0))
            feature_names.extend(dummies.columns.tolist())

    if not parts:
        raise ValueError("No valid feature columns after preparation.")

    X_df = pd.concat(parts, axis=1)
    intercept = np.ones((len(X_df), 1))
    X = np.hstack([intercept, X_df.to_numpy()])

    return X, y, feature_names, clusters


# ── OLS linear regression ──────────────────────────────────────────────────────

def _fit_linear(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    OLS regression via statsmodels.
    """
    import statsmodels.api as sm

    n, p = X.shape  # p includes intercept column
    dof = n - p
    if dof <= 0:
        raise ValueError(f"Not enough observations ({n}) for {p} parameters.")

    model = sm.OLS(y, X)
    if clusters is not None:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
        se_type = "cluster"
    else:
        result = model.fit()
        se_type = "nonrobust"

    # Pack results — index 0 is intercept
    all_names = ["intercept"] + feature_names
    coefficients = {
        name: {
            "coef":    float(result.params[i]),
            "std_err": float(result.bse[i]),
            "t_stat":  float(result.tvalues[i]),
            "p_value": float(result.pvalues[i]),
        }
        for i, name in enumerate(all_names)
    }

    return {
        "model":        "linear",
        "n":            int(n),
        "n_features":   int(p - 1),
        "r2":           float(result.rsquared),
        "adj_r2":       float(result.rsquared_adj),
        "se_type":      se_type,
        "n_clusters":   int(len(np.unique(clusters))) if clusters is not None else None,
        "coefficients": coefficients,
    }



# ── Feature selection ─────────────────────────────────────────────────────────

def select_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    score_col: str = "score",
    corr_threshold: float = 0.85,
    min_score: float = 0.05,
    max_r2_drop: float = 0.02,
    method: str = "linear",
    rf_kwargs: dict | None = None,
    cluster_col: str | None = None,
    reference_categories: dict[str, str] | None = None,
) -> list[str]:
    """
    Systematically reduce feature_cols in two passes.

    Pass 1 — Correlation pruning 
    Pass 2 — Backward elimination
    """
    rf_kwargs = rf_kwargs or {}

    X, y, names, clusters = _prepare_features(df, feature_cols, score_col, cluster_col=cluster_col,
                                               reference_categories=reference_categories)
    X_feat = X[:, 1:]  # drop intercept column

    #correlation pruning
    feat_df = pd.DataFrame(X_feat, columns=names)
    target_corr = feat_df.corrwith(pd.Series(y)).abs()
    corr_matrix = feat_df.corr().abs()

    dropped = set()
    for i, a in enumerate(names):
        if a in dropped:
            continue
        for b in names[i + 1:]:
            if b in dropped:
                continue
            if corr_matrix.loc[a, b] > corr_threshold:
                loser = b if target_corr[a] >= target_corr[b] else a
                dropped.add(loser)
                print(f"  [corr-prune] drop '{loser}'  (|r| with '{a if loser == b else b}' "
                      f"= {corr_matrix.loc[a, b]:.2f}, lower target corr)")

    remaining = [n for n in names if n not in dropped]
    print(f"  [corr-prune] {len(dropped)} features removed → {len(remaining)} remaining")

    # Backward elimination
    def _fit(feature_names):
        idx = [names.index(n) for n in feature_names]
        X_c = np.hstack([np.ones((X_feat.shape[0], 1)), X_feat[:, idx]])
        if method == "linear":
            res = _fit_linear(X_c, y, feature_names, clusters=clusters)
            scores = {n: abs(res["coefficients"][n]["coef"]) for n in feature_names}
            r2 = res["r2"]
        return res, scores, r2, feature_names

    res, scores, r2_cur, cur_names = _fit(remaining)
    r2_label = "R²"
    score_label = "|coef|" 
    print(f"\n  [backward-elim:{method}] start  {r2_label}={r2_cur:.4f}  features={len(cur_names)}")

    while True:
        weakest = min(scores, key=scores.get)
        if scores[weakest] >= min_score:
            print(f"  [backward-elim:{method}] stop  "
                  f"smallest {score_label}={scores[weakest]:.4f} >= {min_score}")
            break

        trial_names = [n for n in cur_names if n != weakest]
        if not trial_names:
            break

        _, trial_scores, r2_trial, trial_names = _fit(trial_names)
        r2_drop = r2_cur - r2_trial

        if r2_drop > max_r2_drop:
            print(f"  [backward-elim:{method}] stop  dropping '{weakest}' would reduce "
                  f"{r2_label} by {r2_drop:.4f} > {max_r2_drop}")
            break

        print(f"  [backward-elim:{method}] drop '{weakest}'  "
              f"{score_label}={scores[weakest]:.4f}  "
              f"{r2_label} {r2_cur:.4f} → {r2_trial:.4f}")
        cur_names = trial_names
        scores = trial_scores
        r2_cur = r2_trial

    print(f"  [backward-elim:{method}] done  {r2_label}={r2_cur:.4f}  features={len(cur_names)}")
    if cur_names != _collapse_dummy_names(cur_names, df, feature_cols):
        print(f"  Final features (expanded): {cur_names}")

    # Collapse any surviving dummy back to its raw source categorical column so the
    # returned list is valid input to explain() again. 
    final = _collapse_dummy_names(cur_names, df, feature_cols)
    print(f"  Final features: {final}")
    return final


def _collapse_dummy_names(names: list[str], df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    categorical_cols = [
        c for c in feature_cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    collapsed: list[str] = []
    for n in names:
        raw = next((c for c in categorical_cols if n.startswith(f"{c}_")), n)
        if raw not in collapsed:
            collapsed.append(raw)
    return collapsed


# ── Main entry point ───────────────────────────────────────────────────────────

def explain(
    grouped: dict[str, pd.DataFrame],
    cfg: dict[str, dict],
    score_col: str = "score",
    feature_cols: list[str] | dict[str, list[str]] | None = None,
    out_dir: Path | None = None,
    cluster_col: str | None = None,
    reference_categories: dict[str, str] | None = None,
) -> dict[str, dict]:
    """
    Fit all enabled models on each group DataFrame.
    """
    if feature_cols is None:
        feature_cols = []

    def _cols_for(model_name: str) -> list[str]:
        if isinstance(feature_cols, dict):
            return feature_cols.get(model_name, [])
        return feature_cols

    all_results: dict[str, dict] = {}

    for group_name, df in grouped.items():
        print(f"\n[explain] {group_name}  ({len(df)} rows)")
        group_results: dict[str, Any] = {}

        if cfg.get("linear", {}).get("enabled"):
            try:
                X, y, feature_names, clusters = _prepare_features(
                    df, _cols_for("linear"), score_col, cluster_col=cluster_col,
                    reference_categories=reference_categories)
                print(f"  linear  features={X.shape[1]-1}  rows={len(y)}")
                res = _fit_linear(X, y, feature_names, clusters=clusters)
                group_results["linear"] = res
                se_note = f"  (SE: {res['se_type']}, {res['n_clusters']} clusters)" if res.get("n_clusters") else ""
                print(f"  linear  R²={res['r2']:.4f}  adj-R²={res['adj_r2']:.4f}  n={res['n']}{se_note}")
                _print_coefficients(res)
            except Exception as e:
                print(f"  linear  ERROR: {e}")

        all_results[group_name] = group_results

    if out_dir is not None:
        save_explain_results(all_results, Path(out_dir) / "explain_results.json")

    return all_results


def _print_coefficients(res: dict) -> None:
    print(f"  {'Feature':<30}  {'Coef':>8}  {'Std Err':>8}  {'p':>7}")
    print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*7}")
    intercept = res["coefficients"].get("intercept")
    if intercept:
        vals = intercept
        stars = "***" if vals["p_value"] < 0.001 else "**" if vals["p_value"] < 0.01 else "*" if vals["p_value"] < 0.05 else ""
        print(f"  {'intercept':<30}  {vals['coef']:>8.4f}  {vals['std_err']:>8.4f}  {vals['p_value']:>6.4f} {stars}")
    sorted_coefs = sorted(
        ((n, v) for n, v in res["coefficients"].items() if n != "intercept"),
        key=lambda x: abs(x[1]["coef"]),
        reverse=True,
    )
    for name, vals in sorted_coefs:
        stars = "***" if vals["p_value"] < 0.001 else "**" if vals["p_value"] < 0.01 else "*" if vals["p_value"] < 0.05 else ""
        print(f"  {name:<30}  {vals['coef']:>8.4f}  {vals['std_err']:>8.4f}  {vals['p_value']:>6.4f} {stars}")



# ── Save / load ────────────────────────────────────────────────────────────────

def save_explain_results(results: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, float) and (obj != obj):  # nan
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    with open(path, "w") as fh:
        json.dump(_clean(results), fh, indent=2)
    print(f"[explain] Saved → {path}")


def load_explain_results(path: Path | str) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ── Reporting ──────────────────────────────────────────────────────────────────

def plot_explain_results(
    results: dict[str, dict],
    output_dir: Path | str,
    top_n: int = 15,
    title_map: dict[str, str] | None = None,
) -> None:

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for group_name, group_res in results.items():
        if not group_res:
            continue

        for model_name, res in group_res.items():
            fig, ax = plt.subplots(
                1, 1,
                figsize=(9, max(4, top_n * 0.35)),
            )

            if model_name == "linear":
                coefs = {k: v["coef"] for k, v in res["coefficients"].items() if k != "intercept"}
                _bar_chart(
                    ax, coefs, top_n,
                    title=f"OLS coefficients\nR²={res['r2']:.3f}  adj-R²={res['adj_r2']:.3f}",
                    center_zero=True,
                    xlabel="OLS coefficient (Δ score per +1 SD of feature)",
                )

            elif model_name == "random_forest":
                _bar_chart(
                    ax, res["importances"], top_n,
                    title=f"RF feature importances\nR²_oob={res['r2_oob']:.3f}  R²_cv5={res['r2_cv5']:.3f}",
                    center_zero=False,
                    xlabel="Feature importance (mean decrease in impurity)",
                )

            else:
                plt.close(fig)
                continue

            display_title = (title_map or {}).get(group_name, group_name)
            fig.suptitle(display_title, fontsize=12, fontweight="bold")
            fig.tight_layout()
            path = fig_dir / f"explain_{group_name}_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[explain] Plot → {path}")


def _bar_chart(
    ax,
    values: dict[str, float],
    top_n: int,
    title: str,
    center_zero: bool,
    xlabel: str = "Value",
) -> None:
    import matplotlib.pyplot as plt

    # Sort by absolute value, take top_n
    sorted_items = sorted(values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [item[0] for item in sorted_items]
    vals  = [item[1] for item in sorted_items]

    colors = ["#c0392b" if v < 0 else "#2980b9" for v in vals]

    ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    if center_zero:
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="x", alpha=0.3)
