#!/usr/bin/env python3
"""
Bias-in-HRM experiment dispatcher.

Usage:
    to run
    python run.py --condition A --input pilot_mini_cf.jsonl --model llama70b
    python run.py --condition E --input pilot_mini_cf.jsonl --model llama70b

    --condition [A or E]
    --input [filename of file in cf datafolder]
    --model [direct model name or known shorthand]
    --run-local to run with local API
    --continue to continue an existing run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

METADATA_FILE = ROOT / "data" / "metadata" / "runs.jsonl"

MODELS = {
    "llama8b":       "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama70b":      "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "qwen2.5":       "qwen/qwen-2.5-72B-Instruct",
    "gpt4omini":     "gpt-4o-mini",
    "claude-sonnet": "claude-sonnet-4-6",

}

#set cluster params
CLUSTER_SETTINGS = {
    "condition_A": dict(
        gpus_per_node=1, cpus_per_task=8, wall_time="01:00:00", num_shards=1,
        max_tokens=1024, temperature=0.0, top_p=1.0,
        max_model_len=8192, max_concurrent=16, max_chars=4000,
        stop_at="}", seed=1234,
    ),
    "condition_E": dict(
        gpus_per_node=4, cpus_per_task=18, wall_time="08:00:00",
        max_tokens=1024, max_model_len=6144,
        temperature=0.0, top_p=1.0, max_concurrent=32, seed=1234,
    ),
}


def get_model_name(model_arg: str) -> tuple:
    if model_arg in MODELS:
        return model_arg, MODELS[model_arg]
    shorthand = Path(model_arg).name.lower().replace("-instruct", "").replace("-", "")[:12]
    return shorthand, model_arg

def check_run_name_exists(name: str) -> bool:
    if METADATA_FILE.exists():
        for line in METADATA_FILE.read_text().splitlines():
            if line.strip() and json.loads(line).get("run_name") == name:
                return True
    return False

def make_run_name(input_path: Path, model_shorthand: str, condition: str) -> str:
    ts = datetime.now().strftime("%d%m_%H%M")
    base = f"{input_path.stem}_c{condition}_{model_shorthand}_{ts}"
    if not check_run_name_exists(base):
        return base
    i = 1
    while check_run_name_exists(f"{base}({i})"):
        i += 1
    return f"{base}({i})"


def _write_metadata(condition: str, run_name: str, params: dict, job_id: str = None):
    ts = datetime.now()
    if job_id is None:
        job_id = f"local_{ts.strftime('%Y%m%d_%H%M%S')}"
    record = {
        "timestamp": ts.isoformat(),
        "condition": condition,
        "run_name":  run_name,
        "job_id":    job_id,
        "params":    params,
    }
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def _continue_run(args):
    """Resume an existing run from where it left off."""
    run_name = args.continue_run

    if not METADATA_FILE.exists():
        print("Error: no metadata file found.", file=sys.stderr)
        sys.exit(1)
    entries = [json.loads(l) for l in METADATA_FILE.read_text().splitlines()
               if l.strip() and json.loads(l).get("run_name") == run_name]
    if not entries:
        print(f"Error: no metadata found for run '{run_name}'.", file=sys.stderr)
        sys.exit(1)
    entry = sorted(entries, key=lambda e: e.get("timestamp", ""))[-1]

    condition   = entry.get("condition")
    params      = entry.get("params", {})
    model_id    = params.get("model", "")
    dataset_str = params.get("input", "")
    pairs_mode  = params.get("pairs", "occ_match")
    use_strmatch       = params.get("strmatch", False)
    strmatch_threshold = params.get("strmatch_threshold", 0.5)
    was_local   = params.get("local", False)

    run_local = args.run_local or was_local
    model_shorthand, model_id = get_model_name(args.model or model_id)
    dataset_path = Path(dataset_str) if dataset_str else None

    if args.input:
        dataset_path = ROOT / "data" / "counterfactual" / args.input

    print(f"Continuing run : {run_name}")
    print(f"Condition      : {condition}")
    print(f"Model          : {model_shorthand}")
    print(f"Mode           : {'local' if run_local else 'cluster'}")

    if condition == "E":
        sys.path.insert(0, str(ROOT / "src" / "condition_E"))

        data_root   = ROOT / "data" / "condition_E"
        node_done   = (data_root / "node_data" / f"{run_name}.json").exists()
        inst_done   = (data_root / "inst_data" / f"{run_name}.json").exists()
        scores_path = data_root / "scores" / f"scores_{run_name}.jsonl"
        score_done  = scores_path.exists()

        if score_done and not args.input:
            print("Already complete (scores exist). Nothing to do.")
            return

        if score_done and args.input:
            scores_path.unlink()
            print("Removed stale scores file; will regenerate after extending.")

        if run_local:
            from src.condition_E.local_pipeline_d import cmd_extract, cmd_instantiate, cmd_instantiate_strmatch, cmd_score
            if not node_done:
                if dataset_path is None or not dataset_path.exists():
                    print("Error: dataset path required for node extraction.", file=sys.stderr)
                    sys.exit(1)
                print("Resuming from: node extraction")
                cmd_extract(str(dataset_path), run_name, str(ROOT / "data"))
            if not inst_done:
                if use_strmatch:
                    print(f"Resuming from: instantiation (string-matching, threshold={strmatch_threshold})")
                    cmd_instantiate_strmatch(run_name, str(ROOT / "data"), pairs_mode, strmatch_threshold)
                else:
                    print("Resuming from: instantiation")
                    cmd_instantiate(run_name, str(ROOT / "data"), pairs_mode)
            print("Resuming from: scoring")
            cmd_score(run_name, str(ROOT / "data"))
        else:
            if dataset_path is None or not dataset_path.exists():
                print("Error: original dataset path not found, cannot resubmit cluster job.", file=sys.stderr)
                sys.exit(1)
            from src.condition_E.submit import submit
            config = {
                "job_name":         f"cE_{run_name[:20]}",
                "run_name":         run_name,
                "dataset":          str(dataset_path),
                "model_checkpoint": model_id,
                "pairs_mode":       pairs_mode,
                "reasoning_parser": "--reasoning-parser deepseek_r1 " if "deepseek8b" in model_shorthand.lower() else "",
                **CLUSTER_SETTINGS["condition_E"],
            }
            job_id = submit(config)
            _write_metadata("E", run_name, {"model": model_id, "input": str(dataset_path),
                                            "local": False, "pairs": pairs_mode,
                                            "continued_from": entry.get("job_id")}, job_id)

    elif condition == "A":
        sys.path.insert(0, str(ROOT / "src" / "condition_A"))

        scores_path = ROOT / "data" / "condition_A" / "scores" / f"{run_name}.jsonl"
        if scores_path.exists() and not args.input:
            print("Already complete (scores exist). Nothing to do.")
            return

        if dataset_path is None or not dataset_path.exists():
            print("Error: --input is required to extend a completed condition A run.", file=sys.stderr)
            sys.exit(1)

        from src.condition_A.run import run as run_A
        job_id = run_A(args, model_shorthand, model_id, run_name, dataset_path,
                       CLUSTER_SETTINGS["condition_A"], pairs_mode)
        _write_metadata("A", run_name, {"model": model_id, "input": str(dataset_path),
                                        "local": False, "pairs": pairs_mode,
                                        "continued_from": entry.get("job_id")}, job_id)

    else:
        print(f"Continuation not supported for condition {condition}.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Bias-in-HRM experiment dispatcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--condition", required=False, choices=["A", "E"],
                        help="Experiment condition (A or E)")
    parser.add_argument("--input", default=None,
                        help="Filename in data/counterfactual/")
    parser.add_argument("--model", default=None,
                        help="Model shorthand or HuggingFace ID")
    parser.add_argument("--run-local", action="store_true",
                        help="Run locally via Claude API instead of cluster")
    parser.add_argument("--pairs", default="occ_match", choices=["occ_match", "all"],
                        help="Pairing strategy: occ_match (default) or all CV×JD combinations")
    parser.add_argument("--continue", dest="continue_run", metavar="RUN_NAME",
                        help="Resume an existing run from where it left off")
    args = parser.parse_args()

    if args.continue_run:
        _continue_run(args)
        return

    if not args.condition:
        parser.error("--condition is required")

    if args.input is None:
        print("Error: --input is required.", file=sys.stderr)
        sys.exit(1)

    dataset_path = ROOT / "data" / "counterfactual" / args.input
    if not dataset_path.exists():
        print(f"Error: file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    if args.model is None:
        print("Error: --model is required for submission.", file=sys.stderr)
        sys.exit(1)

    model_shorthand, model_id = get_model_name(args.model)
    run_name = make_run_name(dataset_path, model_shorthand, args.condition)
    print(f"Dataset : {dataset_path}")
    print(f"Run     : {run_name}")

    if args.condition == "A":
        sys.path.insert(0, str(ROOT / "src" / "condition_A"))
        from src.condition_A.run import run as run_A
        job_id = run_A(args, model_shorthand, model_id, run_name, dataset_path,
                       CLUSTER_SETTINGS["condition_A"], args.pairs)
        _write_metadata("A", run_name, {"model": model_id, "input": str(dataset_path),
                                        "local": args.run_local, "pairs": args.pairs}, job_id)

    elif args.condition == "E":
        sys.path.insert(0, str(ROOT / "src" / "condition_E"))
        from src.condition_E.run import run as run_E
        job_id = run_E(args, model_shorthand, model_id, run_name, dataset_path,
                       CLUSTER_SETTINGS["condition_E"], args.pairs)
        _write_metadata("E", run_name, {"model": model_id, "input": str(dataset_path),
                                        "local": args.run_local, "pairs": args.pairs}, job_id)


if __name__ == "__main__":
    main()
