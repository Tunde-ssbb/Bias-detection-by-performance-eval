# Experiment Analysis

Fairness and performance analysis pipeline for LLM hiring-screening (CV–JD scoring) experiments with gender counterfactuals. Produces the thesis results: tie-aware NDCG, per-JD RMSE, tie-aware Precision@k, three pairwise fairness deltas per metric, and an OLS explainability analysis.

## Structure

```
main.py                   # Entry point — edit constants at the top and run
src/
  parsing.py              # Parse raw JSONL model outputs → parquet
  post_processing.py      # Row-wise enrichment: ground-truth score, competence-overlap and linguistic features
  ground_truth_scoring.py # CV/JD matching score (importance-weighted, linear)
  metrics.py              # All metrics, CIs, threshold/tie logic
  explainability.py       # Feature selection + OLS (cluster-robust SEs) / random forest

results/
  figures/                # PNG plots (incl. explain_<group>_linear.png)
  metric_values/          # Per-group JSON + summary.csv
  explain_results/        # Explainability JSON
```

## Usage

All configuration lives in `main.py` (no CLI):


**Experiments / grouping** — `EXPERIMENTS` maps names to job IDs or file paths. `GROUP_MODE = "each"` analyses every experiment separately; `GROUP_MODE = "custom"` uses the `GROUPS` dict (currently restricted to the four thesis runs: LLaMA-8B, LLaMA-70B, DeepSeek-67B, DeepSeek-R1, gender counterfactuals).

**Step toggles** — skip steps to reuse saved outputs:
```python
STEPS = {"parse": False, "post_process": False, "analyse": True, "explain": False, "report": True}
```

**Key constants** — `N_POOL=15` candidates per JD (5 CVs × 3 counterfactuals), `N_GROUP=5`, `K_PRECISION=5`, `K_PRECISION_DELTA=2` (k scaled proportionally for per-group deltas), `K_NDCG=N_POOL` (untruncated NDCG), `CI_N_BOOTSTRAP` (1000)


## Explainability

Post-hoc OLS of the LLM score on interpretable CV/JD features (Hoffman-style post-hoc explanation), fit per model:

- Features (`EXPLAIN_FEATURES_CORE` in `main.py`, 35 raw columns): overall overlap (`comp_item_overlap`), per-category matched/unmatched importance and CV total importance for 8 competence kinds, linguistic features (`cv_ling_*`, `jd_ling_*`), styles, and the counterfactual group.
- Categorical reference levels: style → `formal`, counterfactual → `neutral` (`REFERENCE_CATEGORIES`).
- Feature selection per model (`FEATURE_SELECTION`): correlation pruning (|r| > 0.85) then backward elimination; zero-variance columns are dropped automatically.
- Standardised features (coefficient = Δ score per +1 SD); **cluster-robust standard errors** clustered on JD.
- Outputs: `results/explain_results/explain_results.json`, `results/figures/explain_<group>_linear.png`.

## Output

- `results/metric_values/<group>.json` — full nested results per group; `summary.csv` — one row per group
- `results/metric_values/summary.csv` summary table used for thesis tables
