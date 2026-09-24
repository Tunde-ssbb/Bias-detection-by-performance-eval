"""
Experiment analysis entry point.

Edit the sections below to configure and run analysis.
No CLI — change parameters directly and run: python new_main.py
"""

import pandas as pd
from pathlib import Path

from src.parsing import parse_experiments, load_parsed
from src.post_processing import enrich, load_processed
from src.metrics import ThresholdSpec, compute_metrics, save_results, flatten_results
from src.explainability import explain, load_explain_results, plot_explain_results, select_features, save_explain_results


# ANSI colours — used only for step banners
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def print_step(n: int, title: str) -> None:
    """Print a clearly visible step banner."""
    print(f"\n{_BOLD}{_CYAN}{'━'*60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  STEP {n}: {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'━'*60}{_RESET}")


# ============================================================================
# EXPERIMENT REGISTRY
# Map experiment names to job IDs or absolute/relative file paths.
# Job IDs are resolved to files under DATA_DIR / "output" at parse time.
# ============================================================================

EXPERIMENTS: dict[str, str | Path] = {
    "test":                          "22843572",
    "twoitem_wjd_llama8b_gender":    "22851469",
    "twoitem_wjd_deepseek_r1_gender":"22859490",
    "twoitem_wjd_deepseek_r1_2":     "22900590",
    "twoitem_wjd_llama8b_lgbtq":     "22967298",
    "twoitem_wjd_deepseek67b_gender":"22966023",
    "twoitem_wjd_llama70b_gender":   "22966008",
    "twoitem_wjd_ministral_revision": "23140318",
    "combined_wjd_llama8b_gender":    "23140477",
}


# ============================================================================
# ANALYSIS GROUPS
# ============================================================================

# GROUP_MODE controls how experiments are grouped before analysis:
#   "each"   — one analysis group per experiment, no concatenation (default)
#   "custom" — use the GROUPS dict defined below
GROUP_MODE = "custom"

GROUPS: dict[str, list[str]] = {
    "twoitem_wjd_llama8b_gender":     ["twoitem_wjd_llama8b_gender"],
    "twoitem_wjd_llama70b_gender":    ["twoitem_wjd_llama70b_gender"],
    "twoitem_wjd_deepseek67b_gender": ["twoitem_wjd_deepseek67b_gender"],
    "twoitem_wjd_deepseek_r1_gender": ["twoitem_wjd_deepseek_r1_gender"],
}


# ============================================================================
# PARAMETERS
# ============================================================================

DATA_DIR        = Path("data")
PARSED_DIR      = DATA_DIR / "parsed_output"
PROCESSED_DIR   = DATA_DIR / "processed_output"

# Parse mode: "all" | "all_new" | "<experiment_name>"
PARSE_MODE      = "all"
VERBOSE_PARSE   = False

# Column names
SCORE_COL       = "score"
GT_COL          = "gt_score_1_10"
GROUP_COL       = "counterfactual"
PAIR_COL        = "group_id"
QUERY_COL       = "jd_id"
GROUP_A         = "female"
GROUP_B         = "male"
GROUP_NEUTRAL   = "neutral"


N_POOL          = 15  # candidates per JD across all 3 counterfactual groups
N_GROUP         = 5   # candidates per JD within a single counterfactual group
K_PRECISION     = 5   # population-level P@k — interview 5 of 15 (matches thesis "Selection" scenario)
PRED_THRESHOLD  = ThresholdSpec(mode="top_k", value=K_PRECISION)
GT_THRESHOLD    = ThresholdSpec(mode="top_k", value=K_PRECISION)


K_PRECISION_DELTA    = round(K_PRECISION * N_GROUP / N_POOL)  # = 2
PRED_THRESHOLD_DELTA = ThresholdSpec(mode="top_k", value=K_PRECISION_DELTA)
GT_THRESHOLD_DELTA   = ThresholdSpec(mode="top_k", value=K_PRECISION_DELTA)


K_NDCG          = N_POOL  # == 15 == the largest pool size in this dataset; min(K_NDCG, n) always == n

# Confidence intervals
CI_CONFIDENCE   = 0.95
CI_N_BOOTSTRAP  = 1000
CI_RANDOM_STATE = 42


METRICS: dict[str, dict] = {
    "rmse":                  {"enabled": True,  "ci": "bootstrap"}, #RMSE
     "avg_ndcg_at_k_per_jd":  {"enabled": True,  "ci": "bootstrap"}, # NDCG
    "precision":             {"enabled": True,  "ci": "bootstrap"}, #P@k
    # 3-way group deltas (headline for the final thesis table): female-neutral,
    # male-neutral, female-male, for each of ndcg / rmse / precision.
    "ndcg_delta_female_neutral":      {"enabled": True, "ci": "bootstrap"},
    "ndcg_delta_male_neutral":        {"enabled": True, "ci": "bootstrap"},
    "ndcg_delta_female_male":         {"enabled": True, "ci": "bootstrap"},
    "rmse_delta_female_neutral":      {"enabled": True, "ci": "bootstrap"},
    "rmse_delta_male_neutral":        {"enabled": True, "ci": "bootstrap"},
    "rmse_delta_female_male":         {"enabled": True, "ci": "bootstrap"},
    "precision_delta_female_neutral": {"enabled": True, "ci": "bootstrap"},
    "precision_delta_male_neutral":   {"enabled": True, "ci": "bootstrap"},
    "precision_delta_female_male":    {"enabled": True, "ci": "bootstrap"},
}

# Output
OUTPUT_DIR      = Path("results")


#Toggle steps
STEPS = {
    "parse":        False,  # parse datafiles, enable if new data
    "post_process": False,  # post_processing.py, add features and GT to data
    "analyse":      False,   # run analysis
    "explain":      False,  # Use OLS 
    "report":       True,
}


EXPLAIN_TOP_N: int = 15   # number of features shown in explain bar charts

EXPLAIN_MODELS: dict[str, dict] = {
    "linear":         {"enabled": True},
    "random_forest":  {"enabled": False,  "n_estimators": 200, "random_state": 42},
}

# ── Feature selection config──────────────────────────────────────────────────────────
# Run select_features() before explain() to prune redundant and weak predictors.
# corr_threshold: drop one of any pair with |r| above this (default 0.85)
# min_coef: backward elimination stops when smallest |coef| >= this
# max_r2_drop: never drop a feature that reduces R² by more than this
FEATURE_SELECTION: dict = {
    "enabled":        True,
    "corr_threshold": 0.85,
    "min_score":      0.05,   # min |coef| for linear, min importance for random_forest
    "max_r2_drop":    0.02,   # max R² / R²_oob drop before stopping elimination
}



TABLES = {
    # "metrics_csv": True,        # coming in new_tables.py
}

LATEX = {
    "results_table": True,    # one row per group, fairness + performance metrics
}
# LaTeX preview mode: "terminal" | "browser" | None
LATEX_PREVIEW = "browser"


# ============================================================================
# STEP 1: PARSE
# Load raw result files, extract scores from completions, and join each row
# to its matching input profile (cv_profile_text/dict, jd_profile_text/dict).
#
#   "all" — parse every experiment, overwrite existing parquet files
#   "all_new" — only parse experiments not yet saved in PARSED_DIR
#   "<name>" — parse a single experiment by name
# ============================================================================

print_step(1, "PARSE")

if STEPS["parse"]:
    parse_experiments(
        EXPERIMENTS,
        data_dir=DATA_DIR,
        parse=PARSE_MODE,
        verbose=VERBOSE_PARSE,
        out_dir=PARSED_DIR,
    )
else:
    print("[parse] Skipped — loading from existing parquet files.")

_needed = (
    set(EXPERIMENTS.keys())
    if GROUP_MODE == "each"
    else {name for members in GROUPS.values() for name in members}
)
parsed: dict[str, pd.DataFrame] = load_parsed(
    {k: v for k, v in EXPERIMENTS.items() if k in _needed},
    data_dir=DATA_DIR,
    out_dir=PARSED_DIR,
)


# ============================================================================
# STEP 2: POST-PROCESS
# Add row-wise derived features to each experiment DataFrame.
# ============================================================================

print_step(2, "POST-PROCESS")

_cols_before = set(next(iter(parsed.values())).columns) if parsed else set()

if STEPS["post_process"]:
    parsed = enrich(parsed, verbose=VERBOSE_PARSE, out_dir=PROCESSED_DIR)
else:
    print(f"[post_process] Skipped — loading from {PROCESSED_DIR}/")
    parsed = load_processed(parsed, out_dir=PROCESSED_DIR, fallback=parsed)

_cols_after = set(next(iter(parsed.values())).columns) if parsed else set()
POST_PROCESS_COLS = sorted(_cols_after - _cols_before)

# Columns added by post-processing that are identifiers or targets, not features.
EXPLAIN_EXCLUDE = {GT_COL, PAIR_COL}

_COMPETENCE_KINDS = [
    "abilities", "knowledge", "related_experience", "required_education",
    "skill", "task", "tech_skill", "work_style",
]
EXPLAIN_FEATURES_CORE = (
    ["comp_item_overlap"]
    + [f"{kind}_matched_importance" for kind in _COMPETENCE_KINDS]
    + [f"{kind}_unmatched_importance" for kind in _COMPETENCE_KINDS]
    + [f"cv_{kind}_total_importance" for kind in _COMPETENCE_KINDS]
    + [
        "cv_ling_action_verb_count",
        "cv_ling_flesch_reading_ease", "jd_ling_flesch_reading_ease",
        "cv_ling_type_token_ratio", "jd_ling_type_token_ratio",
        "cv_ling_hapax_ratio", "jd_ling_hapax_ratio",
    ]
    + ["cv_ling_style", "jd_ling_style"]
    + [GROUP_COL]  # "counterfactual" — female/male/neutral, added last
)
assert len(EXPLAIN_FEATURES_CORE) == 35, f"expected 35 core features, got {len(EXPLAIN_FEATURES_CORE)}"

EXPLAIN_FEATURES = EXPLAIN_FEATURES_CORE

print(f"\n[post_process] Added columns ({len(POST_PROCESS_COLS)}):")
for c in POST_PROCESS_COLS:
    tag = " (excluded from explain — GT)" if c == GT_COL else ""
    print(f"    {c}{tag}")
print(f"\n[post_process] Explain features ({len(EXPLAIN_FEATURES)}): {EXPLAIN_FEATURES}")


# ============================================================================
# STEP 3: GROUP
# ============================================================================

print_step(3, "GROUP")

_active_groups = (
    {name: [name] for name in parsed}   # each experiment is its own group
    if GROUP_MODE == "each"
    else GROUPS
)

grouped: dict[str, pd.DataFrame] = {}

for group_name, members in _active_groups.items():
    missing = [m for m in members if m not in parsed]
    if missing:
        print(f"[group] WARNING FOR: {group_name} missing experiments: {missing}")
        continue

    frames = []
    for member in members:
        frame = parsed[member].copy()
        frame["experiment"] = member
        frames.append(frame)

    grouped[group_name] = pd.concat(frames, ignore_index=True)
    total = sum(len(parsed[m]) for m in members)
    print(f"[group] {group_name}: {total} rows from {members}")

# drop rows without group label
for group_name, df in grouped.items():
    n_unlabeled = df[GROUP_COL].isna().sum()
    if n_unlabeled:
        print(f"[group] {group_name}: dropping {n_unlabeled} row(s) with no {GROUP_COL} label")
        grouped[group_name] = df[df[GROUP_COL].notna()].reset_index(drop=True)


SCORE_MIN = 1.0 
for group_name, df in grouped.items():
    n_missing = df[SCORE_COL].isna().sum()
    if n_missing:
        print(f"[group] {group_name}: filling {n_missing} unparseable score(s) with SCORE_MIN={SCORE_MIN}")
        df[SCORE_COL] = df[SCORE_COL].fillna(SCORE_MIN)


# ============================================================================
# STEP 4: ANALYSE
# ============================================================================

print_step(4, "ANALYSE")

analysis_results: dict[str, dict] = {}

if STEPS["analyse"]:
    for group_name, df in grouped.items():
        print(f"\n[analyse] {group_name}  ({len(df)} rows) ...")

        results = compute_metrics(
            df,
            metrics_cfg          = METRICS,
            pred_threshold       = PRED_THRESHOLD,
            gt_threshold         = GT_THRESHOLD,
            pred_threshold_delta = PRED_THRESHOLD_DELTA,
            gt_threshold_delta   = GT_THRESHOLD_DELTA,
            group_a              = GROUP_A,
            group_b              = GROUP_B,
            group_neutral        = GROUP_NEUTRAL,
            score_col            = SCORE_COL,
            gt_col               = GT_COL,
            group_col            = GROUP_COL,
            pair_col             = PAIR_COL,
            query_col            = QUERY_COL,
            k                    = K_NDCG,
            n_bootstrap          = CI_N_BOOTSTRAP,
            ci_confidence        = CI_CONFIDENCE,
            ci_seed              = CI_RANDOM_STATE,
        )

        analysis_results[group_name] = results

        # Print each metric: value and CI
        print(f"\n  {'Metric':<28}  {'Value':>9}  {'95% CI':^22}  CI mode")
        print(f"  {'─'*28}  {'─'*9}  {'─'*22}  {'─'*12}")
        for metric_name, entry in results["metrics"].items():
            if "value" not in entry:
                for sub, sub_entry in entry.items():
                    label = f"  {metric_name}.{sub}"
                    v  = sub_entry["value"]
                    lo = sub_entry["ci_lower"]
                    hi = sub_entry["ci_upper"]
                    mode = sub_entry["ci_mode"] or "—"
                    v_str  = f"{v:9.4f}"  if v  is not None and v  == v  else f"{'n/a':>9}"
                    ci_str = f"[{lo:.4f}, {hi:.4f}]" if lo == lo and hi == hi else f"{'—':^22}"
                    print(f"  {label:<30}  {v_str}  {ci_str:<22}  {mode}")
            else:
                v  = entry["value"]
                lo = entry["ci_lower"]
                hi = entry["ci_upper"]
                mode = entry["ci_mode"] or "—"
                v_str  = f"{v:9.4f}"  if v  is not None and v  == v  else f"{'n/a':>9}"
                ci_str = f"[{lo:.4f}, {hi:.4f}]" if lo == lo and hi == hi else f"{'—':^22}"
                print(f"  {metric_name:<28}  {v_str}  {ci_str:<22}  {mode}")
        print()

else:
    import json
    METRICS_DIR = OUTPUT_DIR / "metric_values"
    print(f"[analyse] Skipped — loading from {METRICS_DIR}/")
    for group_name in grouped:
        path = METRICS_DIR / f"{group_name}.json"
        if path.exists():
            with open(path) as fh:
                analysis_results[group_name] = json.load(fh)
            print(f"  Loaded {group_name}")
        else:
            print(f"  WARNING: no saved results for '{group_name}' — run with analyse=True first")


# ============================================================================
# STEP 5: EXPLAIN
# ============================================================================

print_step(5, "EXPLAIN")

EXPLAIN_DIR = OUTPUT_DIR / "explain_results"
explain_results: dict[str, dict] = {}

if STEPS["explain"]:
    _enabled_models = [m for m, c in EXPLAIN_MODELS.items() if c.get("enabled")]

    # Reference (dropped) category for each categorical explain feature
    REFERENCE_CATEGORIES = {
        "cv_ling_style": "formal",
        "jd_ling_style": "formal",
        GROUP_COL:       "neutral",  # "counterfactual"
    }

    # Feature pruning
    for _group_name, _group_df in grouped.items():
        if FEATURE_SELECTION["enabled"]:
            _explain_features: dict[str, list[str]] = {}
            for _model in _enabled_models:
                print(f"\n[feature-selection] {_group_name} / {_model} ...")
                _rf_kwargs = {k: v for k, v in EXPLAIN_MODELS.get("random_forest", {}).items()
                             if k != "enabled"} if _model == "random_forest" else {}
                _explain_features[_model] = select_features(
                    _group_df,
                    feature_cols   = EXPLAIN_FEATURES,
                    score_col      = SCORE_COL,
                    corr_threshold = FEATURE_SELECTION["corr_threshold"],
                    min_score      = FEATURE_SELECTION["min_score"],
                    max_r2_drop    = FEATURE_SELECTION["max_r2_drop"],
                    method         = _model,
                    rf_kwargs      = _rf_kwargs,
                    cluster_col    = QUERY_COL,
                    reference_categories = REFERENCE_CATEGORIES,
                )
        else:
            _explain_features = EXPLAIN_FEATURES

        explain_results.update(explain(
            {_group_name: _group_df},
            cfg          = EXPLAIN_MODELS,
            score_col    = SCORE_COL,
            feature_cols = _explain_features,
            out_dir      = None,  # saved once, combined, below
            cluster_col  = QUERY_COL,
            reference_categories = REFERENCE_CATEGORIES,
        ))

    if explain_results:
        save_explain_results(explain_results, EXPLAIN_DIR / "explain_results.json")

else:
    path = EXPLAIN_DIR / "explain_results.json"
    if path.exists():
        explain_results = load_explain_results(path)
        print(f"[explain] Skipped — loaded from {path}")
    else:
        print(f"[explain] Skipped — no saved results found at {path}. Run with explain=True first.")


# ============================================================================
# STEP 6: REPORT
# Saves results and generates enabled figures  tables / latex.
# ============================================================================

print_step(6, "REPORT")

if not STEPS["report"]:
    print("[report] Skipped.")
else:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR = OUTPUT_DIR / "metric_values"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Metrics: JSON per group + combined CSV ────────────────────────────────

    if STEPS["analyse"]:   # only save fresh results; skip if we loaded from disk
        for group_name, results in analysis_results.items():
            save_results(results, METRICS_DIR / f"{group_name}.json")
            print(f"[report] {group_name} → {METRICS_DIR}/{group_name}.json")

    rows = []
    for group_name, results in analysis_results.items():
        flat = flatten_results(results)
        flat["group"] = group_name
        rows.append(flat)
    summary_df = pd.DataFrame(rows)
    cols =  ["group"] + [c for c in summary_df.columns if c != "group"]
    summary_df[cols].to_csv(METRICS_DIR / "summary.csv", index=False)
    print(f"[report] Summary CSV → {METRICS_DIR}/summary.csv")



    # ── Explainability plots ──────────────────────────────────────────────────

    MODEL_DISPLAY_NAMES = {
        "twoitem_wjd_llama8b_gender":     "LLaMA-8B",
        "twoitem_wjd_llama70b_gender":    "LLaMA-70B",
        "twoitem_wjd_deepseek67b_gender": "DeepSeek-67B",
        "twoitem_wjd_deepseek_r1_gender": "DeepSeek-R1",
    }
    if explain_results:
        plot_explain_results(explain_results, output_dir=OUTPUT_DIR, top_n=EXPLAIN_TOP_N,
                              title_map=MODEL_DISPLAY_NAMES)



