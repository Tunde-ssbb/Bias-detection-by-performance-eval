"""
Condition A experiment runner.

Called by the root run.py. Handles prompt generation, local API inference,
and cluster (SLURM/vLLM) submission. Returns job_id for cluster runs, None for local.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _confirm_api_calls(n: int, model: str):
    print(f"\n{'='*60}")
    print(f"  >> API Request Confirmation <<")
    print(f"  Total requests : {n}")
    print(f"  Target model   : {model}")
    print(f"{'='*60}")
    answer = input("Continue with API calls? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        print("cancelled.")
        sys.exit(0)


def generate_prompts(dataset_path: Path, run_name: str, pairs_mode: str = "occ_match") -> Path:
    from condition_A.generate_prompts import format_prompt
    from shared.pipeline_shared import load_data, CVRecord, JDRecord
    from shared.pairer import get_pairs

    print(f"\n── Generating condition A prompts from {dataset_path.name}  (pairs={pairs_mode})")
    records = load_data(dataset_path)

    records_by_id = {r["custom_id"]: r for r in records}
    pairs = get_pairs(records, pairs_mode)

    n_cvs = sum(1 for r in records if r.get("custom_id", "").startswith("cv::"))
    n_jds = sum(1 for r in records if r.get("custom_id", "").startswith("jd::"))
    print(f"   {n_cvs} CV variants × {n_jds} JDs → {len(pairs)} pairs")

    prompt_path = ROOT / "data" / "condition_A" / "prompts" / f"{run_name}.jsonl"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(prompt_path, "w") as f:
        for cv_id, jd_id in pairs:
            r   = records_by_id[cv_id]
            r_jd = records_by_id[jd_id]
            cf_key = r.get("counterfactual_key", r.get("cf", "original"))
            cv = CVRecord(
                custom_id=r["custom_id"], occ_code=r["occ_code"],
                profile_text=r["profile_text"], batch_name=r.get("batch_name", ""),
                style=r.get("style", ""), profile_dict=r.get("profile_dict", {}),
                occ_title=r.get("occ_title", ""),
            )
            jd = JDRecord(
                custom_id=r_jd["custom_id"], occ_code=r_jd["occ_code"],
                profile_text=r_jd["profile_text"], batch_name=r_jd.get("batch_name", ""),
                style=r_jd.get("style", ""), profile_dict=r_jd.get("profile_dict", {}),
                occ_title=r_jd.get("occ_title", r_jd.get("occ_code", "")),
            )
            row = {
                "id":                  f"{cv_id}__{jd_id}",
                "variant_id":          "base",
                "cv_id":               cv_id,
                "jd_id":               jd_id,
                "occ_code":            r["occ_code"],
                "counterfactual":      cf_key,
                "counterfactual_info": r.get("counterfactual_info", ""),
                "cv_style":            cv.style,
                "jd_style":            jd.style,
                "prompt":              format_prompt(cv, jd),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"   {written} prompts written → {prompt_path}")
    return prompt_path


def _is_openai(model_id: str) -> bool:
    return model_id.startswith(("gpt-", "o1", "o3"))


def _submit_and_poll_openai(client, model_id: str, prompts: list[dict]) -> dict[str, str]:
    """Submit OpenAI batch and poll until done. Returns {custom_id: completion_text}."""
    import tempfile, os, time

    lines = [json.dumps({
        "custom_id": row["id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": model_id, "max_tokens": 512, "temperature": 0,
                 "messages": [{"role": "user", "content": row["prompt"]}]},
    }) for row in prompts]

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        tmp = f.name
    try:
        with open(tmp, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(input_file_id=file_obj.id,
                                      endpoint="/v1/chat/completions",
                                      completion_window="24h")
    finally:
        os.unlink(tmp)

    print(f"  [batch] submitted {len(prompts)} requests → {batch.id}")
    start = time.time()
    try:
        while True:
            batch = client.batches.retrieve(batch.id)
            counts = batch.request_counts
            elapsed = int(time.time() - start)
            m, s = divmod(elapsed, 60)
            msg = (f"  [batch] status={batch.status}  "
                   f"completed={counts.completed}/{counts.total}  "
                   f"elapsed={m:02d}:{s:02d}")
            print(f"\r{msg:<90}", end="", flush=True)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                print()
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n  [batch] cancelling {batch.id} ...")
        try:
            client.batches.cancel(batch.id)
            batch = client.batches.retrieve(batch.id)
            if batch.status == "completed":
                print("  [batch] already completed — collecting results.")
            else:
                print("  [batch] cancelled.")
                raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  [batch] cancel failed: {e}")
            raise KeyboardInterrupt

    if batch.status != "completed":
        raise RuntimeError(f"Batch ended with status={batch.status}")

    results = {}
    for line in client.files.content(batch.output_file_id).text.splitlines():
        if not line.strip():
            continue
        rec  = json.loads(line)
        cid  = rec["custom_id"]
        body = rec.get("response", {}).get("body", {})
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        results[cid] = text
    return results


def _run_local(dataset_path: Path, model_id: str, run_name: str, pairs_mode: str):
    from condition_A.parse_results import parse_results
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT.parent / "onto-matcher" / ".env", override=False)

    if _is_openai(model_id):
        import openai
        client = openai.OpenAI()
    else:
        import anthropic
        client = anthropic.Anthropic()

    prompt_path = generate_prompts(dataset_path, run_name, pairs_mode)
    with open(prompt_path) as f:
        prompts = [json.loads(l) for l in f if l.strip()]

    # Skip already-done rows
    out_path = ROOT / "data" / "condition_A" / "results" / f"{run_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for l in f:
                if l.strip():
                    done_ids.add(json.loads(l).get("id"))
    todo = [r for r in prompts if r.get("id") not in done_ids]
    print(f"  {len(todo)} to process, {len(done_ids)} already done")

    if not todo:
        print("Nothing to do.")
        return

    _confirm_api_calls(len(todo), model_id)

    if _is_openai(model_id):
        # Batch mode
        completions = _submit_and_poll_openai(client, model_id, todo)
        with open(out_path, "a") as out_f:
            for row in todo:
                text = completions.get(row["id"], "")
                result = {
                    "id":                  row.get("id"),
                    "completion":          text,
                    "cv_id":               row.get("cv_id"),
                    "jd_id":               row.get("jd_id"),
                    "occ_code":            row.get("occ_code"),
                    "counterfactual":      row.get("counterfactual"),
                    "counterfactual_info": row.get("counterfactual_info"),
                    "variant_id":          row.get("variant_id"),
                    "cv_style":            row.get("cv_style"),
                    "jd_style":            row.get("jd_style"),
                }
                out_f.write(json.dumps(result) + "\n")
        print(f"Done. {len(todo)} results → {out_path}")
    else:
        # Sequential mode (Anthropic)
        with open(out_path, "a") as out_f:
            for i, row in enumerate(todo, 1):
                print(f"[{i}/{len(todo)}] {row.get('id', '?')}")
                try:
                    msg  = client.messages.create(model=model_id, max_tokens=512,
                                                  messages=[{"role": "user", "content": row["prompt"]}])
                    text = msg.content[0].text
                    usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
                    result = {
                        "id":                  row.get("id"),
                        "completion":          text,
                        "cv_id":               row.get("cv_id"),
                        "jd_id":               row.get("jd_id"),
                        "occ_code":            row.get("occ_code"),
                        "counterfactual":      row.get("counterfactual"),
                        "counterfactual_info": row.get("counterfactual_info"),
                        "variant_id":          row.get("variant_id"),
                        "cv_style":            row.get("cv_style"),
                        "jd_style":            row.get("jd_style"),
                        "usage": usage,
                    }
                except Exception as e:
                    result = {"id": row.get("id"), "error": str(e)}
                    print(f"  ERROR: {e}")
                out_f.write(json.dumps(result) + "\n")
        print(f"\nDone. {len(todo)} results → {out_path}")

    print(f"\n── Parsing scores")
    stats = parse_results(out_path, run_name)
    print(f"   Valid JSON : {stats['valid']}/{stats['total']}  ({stats['parse_rate']*100:.1f}%)")
    print(f"   Scores     → {stats['output']}")


def _submit_cluster(config: dict) -> str:
    import subprocess
    from datetime import datetime

    template_dir = ROOT / "job_files" / "condition_A"
    if "DeepSeek-R1-Distill-Llama-8B" in config.get("model_checkpoint", ""):
        template_file = template_dir / "run_vllm_serve_ds_reas.template"
        config = {**config, "max_tokens": 1024}
        print(f"Using DeepSeek-R1 template")
    else:
        template_file = template_dir / "run_vllm_serve.template"

    if not template_file.exists():
        print(f"✗ Template not found: {template_file}", file=sys.stderr)
        sys.exit(1)

    job_content = template_file.read_text().format(**config)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_file    = template_dir / f"generated_cA_{config['run_name']}_{timestamp}.job"
    job_file.write_text(job_content)

    sbatch_cmd = ["sbatch"]
    if config["num_shards"] > 1:
        sbatch_cmd.extend(["--array", f"0-{config['num_shards']-1}"])
    sbatch_cmd.extend(["--export", f"ALL,NUM_SHARDS={config['num_shards']}"])
    sbatch_cmd.append(str(job_file))

    print(f"Submitting: {' '.join(sbatch_cmd)}")
    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"✓ Submitted: job {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ sbatch failed:\n  stdout: {e.stdout}\n  stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)


def _run_cluster(dataset_path: Path, model_id: str, model_shorthand: str,
                 run_name: str, defaults: dict, pairs_mode: str) -> str:
    prompt_path = generate_prompts(dataset_path, run_name, pairs_mode)

    config = {
        "condition":        "A",
        "run_name":         run_name,
        "job_name":         f"vllm_{model_shorthand}",
        "model_checkpoint": model_id,
        "infile":           str(prompt_path),
        "prev_jsonl_file":  "nonexist.jsonl",
        **defaults,
    }
    job_id = _submit_cluster(config)

    print(f"\n{'='*60}")
    print(f"Condition A submitted  |  run: {run_name}")
    print(f"  Job ID : {job_id}  |  Model: {model_shorthand}  |  GPUs: {defaults['gpus_per_node']}")
    print(f"{'='*60}")
    print(f"Monitor : squeue -j {job_id}")
    print(f"Query   : python query.py --condition A")
    return job_id


def run(args, model_shorthand: str, model_id: str, run_name: str,
        dataset_path: Path, cluster_defaults: dict, pairs_mode: str = "occ_match") -> "str | None":
    """Entry point called by root run.py. Returns job_id or None for local runs."""
    if args.run_local:
        _run_local(dataset_path, model_id, run_name, pairs_mode)
        return None
    return _run_cluster(dataset_path, model_id, model_shorthand, run_name, cluster_defaults, pairs_mode)


def get_files_for_run(run_name: str) -> list:
    """Return all data files that belong to this run."""
    base = ROOT / "data" / "condition_A"
    exact = [
        base / "prompts" / f"{run_name}.jsonl",
        base / "scores"  / f"{run_name}.jsonl",
        base / "scores"  / f"{run_name}.gt.jsonl",
    ]
    globbed = [
        p for p in (base / "results").glob(f"{run_name}*.jsonl") if _matches_run(p, run_name)
    ] + [
        p for p in (base / "logs").glob(f"*{run_name}*") if _log_matches_run(p, run_name)
    ]
    return [p for p in exact if p.exists()] + globbed

def _matches_run(path: Path, run_name: str) -> bool:
    stem = path.name
    if not stem.startswith(run_name):
        return False
    rest = stem[len(run_name):]
    return rest == "" or rest[0] in ("_", ".")


def _log_matches_run(path: Path, run_name: str) -> bool:
    name = path.name
    return (
        f"_{run_name}_" in name or
        f"_{run_name}." in name or
        name.startswith(run_name + "_") or
        name.startswith(run_name + ".")
    )
