# Run-onto-experiments

Cluster experiment runner for Onto-matcher and baseline. Two scoring conditions (A=Baseline and E=Onto-matcher), one shared dataset pipeline.



## Project layout

```
Run-onto-experiments/
├── run.py                  entry point — submit / continue runs (A and E)
├── query.py                inspect runs, pipeline stage, delete run data
├── analyse.py              compute ranking metrics, bias metrics, pipeline overview
│
├── data/
│   ├── counterfactual/         input datasets (JSONL)
│   ├── metadata/               run submission records (runs.jsonl)
│   ├── ontology_schema.json    dumped ontology schema — regenerate with:
│   │                               python src/condition_E/ontology.py
│   ├── condition_A/            Store Condition A related data
│   ├── condition_E/            Store Condition E related data
│   ├── analysed/               per-run JSONL with scores + GT + explanations
│   └── metric_results/         per-run metric text files + comparison tables
│
├── src/
│   ├── shared/
│   │   ├── pipeline_shared.py          LLM call utilities, data models, load_data()
│   │   ├── ranking_metrics.py          P@k, NDCG, RMSE, MAE, bias deltas, bootstrap CIs
│   │   ├── ground_truth_scoring.py     GT scores from O*NET profile overlap
│   │   ├── expand_counterfactuals.py   counterfactual expansion CLI
│   │   ├── pairer.py                   CV/JD pairing strategies
│   │   └── scoring_weights.py          importance level weights
│   ├── condition_A/
│   │   ├── run.py                      cluster + local runner
│   │   ├── generate_prompts.py         prompt builder
│   │   ├── parse_results.py            score parser
│   │   ├── explainer.py                explanation builder
│   │   └── status.py                   run status helpers
│   └── condition_E/
│       ├── run.py                      entry point
│       ├── extractor_e.py              two-stage NER node extraction
│       ├── cluster_pipeline_d.py       SLURM-side pipeline (extract → inst → score)
│       ├── local_pipeline_d.py         local pipeline (Claude API)
│       ├── scorer_e.py                 relation-propagation scorer
│       ├── explainer.py                explanation builder
│       ├── submit.py                   SLURM job template rendering + sbatch
│       ├── status.py                   pipeline stage reporting
│       ├── graph_utils.py              anchor node injection
│       └── ontology.py                 parse/dump/load ontology schema
│           ontology/competence_ontology.ttl
│
├── job_files/                          Not included because it contains account information

```

---

## Step 0 — Prepare counterfactual dataset

Both conditions read from `data/counterfactual/`. Run once per raw dataset:

```bash
python src/shared/expand_counterfactuals.py \
    --input  \[input file in raw data folder\] \
    --output \[output file in counterfactual folder\] \
    --counterfactual-type gender
```

---

## Condition A — Direct LLM scoring

### Local API example use

```bash
python run.py --condition A --input pilot_mini_cf.jsonl --run-local --model gpt-4o-mini
```

### Cluster (SLURM / vLLM) example use

```bash
python run.py --condition A --input pilot_mini_cf.jsonl --model llama70b --cluster snellius
```

The SLURM job generates prompts, runs vLLM inference, and auto-parses scores.

---

## Condition E — Ontology-driven scoring

### Local API

```bash
python run.py --condition E --input pilot_mini_cf.jsonl --run-local --model gpt-4o-mini
```

Runs all phases sequentially: extract → instantiate → score.

### Cluster (SLURM / vLLM)

```bash
python run.py --condition E --input pilot_mini_cf.jsonl --model llama70b --cluster snellius
```

| Step | What happens | Output |
|---|---|---|
| 1a/1b | Span detection + NER classification | `node_data/<run>.json`, `graph_data/<run>.json` |
| 1c | Importance labelling (JD nodes) | updates node_data + graph_data |
| 2 | Instantiation | `inst_data/<run>.json` |
| 3 | Ontology scoring | `scores/scores_<run>.jsonl` + `scores_<run>_detail.jsonl` |

### Step-by-step cluster reruns (condition E)

```bash
export PYTHONPATH=src:src/condition_E

python -m cluster_pipeline_d --gen-node-prompts --dataset data/counterfactual/<file>.jsonl --run <run> --data-dir data
# [submit vLLM inference on <run>_nodes_stage1.jsonl, then stage2.jsonl]
python -m cluster_pipeline_d --parse-nodes --run <run> --results <path> --data-dir data

python -m cluster_pipeline_d --gen-importance-prompts --run <run> --data-dir data
# [submit vLLM inference on <run>_importance.jsonl]
python -m cluster_pipeline_d --parse-importance --run <run> --results <path> --data-dir data

python -m cluster_pipeline_d --gen-inst-prompts --run <run> --data-dir data
# [submit vLLM inference on <run>_inst.jsonl]
python -m cluster_pipeline_d --parse-inst --run <run> --results <path> --data-dir data

python -m local_pipeline_d --score --run <run> --data-dir data
```

---

## Inspect and delete runs

```bash
python query.py                        # list all runs
python query.py --delete <run_name>    # delete run data + metadata
```

---

## Analysis

```bash
python analyse.py                      # analyse all completed runs
```

Produces per-run metric files in `data/metric_results/` and prints a pipeline overview including NER accuracy, instantiation accuracy, and P@9 relative gain (relations vs. no-relations).


---

## Regenerate ontology schema

After editing `competence_ontology.ttl`:

```bash
python src/condition_E/ontology.py
```

---

## Continue a run

```bash
python run.py --continue <run_name>
python run.py --continue <run_name> --input <new_data.jsonl>   # extend with new data
```

---


## Key configuration

| File | What to edit |
|---|---|
| `run.py` → `_CLUSTER_DEFAULTS` | SLURM parameters (GPUs, wall time, max tokens) |
| `src/condition_E/ontology.py` | ontology related config |
| `src/shared/scoring_weights.py` | Importance level weights |
| `job_files/condition_E/run_condition_E.template` | Cluster account, partition |
| `job_files/condition_A/run_vllm_serve.template` | Cluster account, partition |
