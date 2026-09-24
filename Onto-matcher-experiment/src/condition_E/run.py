"""
Condition D experiment runner.

directs local and cluster version of experiment
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


#double check API calls
def confirm_api_calls(n: int, model: str):
    print(f"\n{'='*60}")
    print(f"  >> Do you want to proceed with API calls? <<")
    print(f"  Total requests : {n}")
    print(f"  Target model   : {model}")
    print(f"{'='*60}")
    answer = input("Continue with API calls? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("Operation cancelled.")
        sys.exit(0)


# local runner
def run_local(dataset_path: Path, model_id: str, run_name: str, pairs_mode: str):
    sys.path.insert(0, str(ROOT / "src" / "condition_E"))
    from local_pipeline_d import cmd_extract, cmd_instantiate, cmd_score
    import json

    #get records
    with open(dataset_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    n_cvs = sum(1 for r in records if r.get("custom_id", "").startswith("cv::"))
    n_jds = sum(1 for r in records if r.get("custom_id", "").startswith("jd::"))
    est_calls = (n_cvs + n_jds) + n_cvs * n_jds  # nodes + instantiations (no relations)
    confirm_api_calls(est_calls, model_id)

    #run the stages
    print(f"\nPhase 1: Extract nodes  ({n_cvs} CVs, {n_jds} JDs)")
    cmd_extract(str(dataset_path), run_name)

    print(f"\nPhase 2: Instantiate  (pairs={pairs_mode})")
    cmd_instantiate(run_name, pairs_mode=pairs_mode)

    print(f"\n── Phase 3: Scoring")
    cmd_score(run_name)


#cluster runner
def _run_cluster(dataset_path: Path, model_id: str, model_shorthand: str,
                 run_name: str, defaults: dict, pairs_mode: str, cluster: str) -> str:
    from condition_E.submit import submit

    #submit job with deafult configd
    config = {
        "condition":        "E",
        "job_name":         f"cE_{run_name[:20]}",
        "model_checkpoint": model_id,
        "dataset":          str(dataset_path),
        "run_name":         run_name,
        "pairs_mode":       pairs_mode,
        "reasoning_parser": "",
        **defaults,
    }

    #main submission
    job_id = submit(config, cluster=cluster)

    # print info and return jobid
    print(f"\n{'='*60}")
    print(f"Condition E submitted  |  run: {run_name}  |  cluster: {cluster}")
    print(f"  Job ID : {job_id}  |  Model: {model_shorthand}  |  GPUs: {defaults['gpus_per_node']}")
    print(f"{'='*60}")
    print(f"Logs    : data/condition_E/logs/cE_{run_name[:20]}_{job_id}.out")
    print(f"Query   : python query.py --condition E --run {run_name}")
    return job_id


# main entry point
def run(args, model_shorthand: str, model_id: str, run_name: str,
        dataset_path: Path, cluster_defaults: dict, pairs_mode: str = "occ_match") -> "str | None":
    if args.run_local:
        run_local(dataset_path, model_id, run_name, pairs_mode)
        return None
    cluster = getattr(args, "cluster", "snellius")
    return _run_cluster(dataset_path, model_id, model_shorthand, run_name, cluster_defaults, pairs_mode, cluster)

# some fuzzy name matching for result and prompt files
def _matches_run(path: Path, run_name: str) -> bool:
    stem = path.name
    if not stem.startswith(run_name):
        return False
    rest = stem[len(run_name):]
    return rest == "" or rest[0] in ("_", ".")

# some fuzzy name matching for log files
def _log_matches_run(path: Path, run_name: str) -> bool:
    name = path.name
    return (
        f"_{run_name}_" in name or
        f"_{run_name}." in name or
        name.startswith(run_name + "_") or
        name.startswith(run_name + ".")
    )


# for run deletion helper, get all files associated with a run
def get_files_for_run(run_name: str) -> list:
    
    base = ROOT/ "data" / "condition_E"
    exact = [
        base / "node_data"  / f"{run_name}.json",
        base / "graph_data" / f"{run_name}.json",
        base / "inst_data"  / f"{run_name}.json",
        base / "scores"     / f"scores_{run_name}.jsonl",
        base / "scores"     / f"scores_{run_name}.gt.jsonl",
    ]
    globbed = [
        p for p in (base / "prompts").glob(f"{run_name}*.jsonl") if _matches_run(p, run_name)
    ] + [
        p for p in (base / "results").glob(f"{run_name}*.jsonl") if _matches_run(p, run_name)
    ] + [
        p for p in (base / "logs").glob(f"*{run_name}*") if _log_matches_run(p, run_name)
    ]
    return [p for p in exact if p.exists()] + globbed