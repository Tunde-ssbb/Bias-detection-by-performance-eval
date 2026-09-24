#!/usr/bin/env python3
"""
Parse raw vLLM output for condition A and save extracted scores.

Reads the completion field from each result row, attempts to parse it as JSON,
and writes the extracted score + justification to data/condition_A/scores/.

Usage:
    python src/condition_A/parse_results.py --results data/condition_A/results/my_run_llama70b_24205995_0.jsonl
    python src/condition_A/parse_results.py --results data/condition_A/results/my_run_llama70b_24205995_0.jsonl --run my_run_name
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _repair_truncated(text: str):
    """Try to parse truncated JSON by closing unterminated strings and braces."""
    start = text.find("{")
    if start == -1:
        return None
    fragment = text[start:].rstrip().rstrip(",")

    in_string = False
    escape_next = False
    depth = 0
    for ch in fragment:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

    if depth <= 0:
        return None

    # Close unterminated string then close open braces
    suffix = ('"' if in_string else "") + "}" * depth
    try:
        return json.loads(fragment + suffix)
    except json.JSONDecodeError:
        return None


def _extract_json(text: str):
    """Try to extract a JSON object from an LLM completion. for truncated or malformed JSON, attempt to repair it."""
    if not text or not text.strip():
        return None
    # Strip markdown code fences
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    # Find first {...} block (complete)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        cleaned = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    # Fallback: try to repair truncated JSON
    return _repair_truncated(text)


def parse_results(results_path: Path, run_name: str) -> dict:
    """Parse a vLLM results JSONL and save scores. Returns summary stats."""
    out_path = ROOT / "data" / "condition_A" / "scores" / f"{run_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = valid = errors = 0
    with open(results_path) as f_in, open(out_path, "a") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)

            if "error" in row:
                errors += 1
                f_out.write(json.dumps({
                    "id":    row.get("id"),
                    "error": row["error"],
                    "valid": False,
                }) + "\n")
                continue

            parsed = _extract_json(row.get("completion", ""))
            score   = parsed.get("score")     if parsed else None
            justif  = parsed.get("justification") if parsed else None
            is_valid = parsed is not None and score is not None

            if is_valid:
                valid += 1

            out_row = {
                "id":                  row.get("id"),
                "cv_id":               row.get("cv_id"),
                "jd_id":               row.get("jd_id"),
                "occ_code":            row.get("occ_code"),
                "counterfactual":      row.get("counterfactual"),
                "counterfactual_info": row.get("counterfactual_info"),
                "variant_id":          row.get("variant_id"),
                "cv_style":            row.get("cv_style"),
                "jd_style":            row.get("jd_style"),
                "score":               score,
                "justification":       justif,
                "valid":               is_valid,
                "raw_completion":      row.get("completion"),
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    return {"total": total, "valid": valid, "errors": errors,
            "parse_rate": round(valid / total, 3) if total else 0,
            "output": str(out_path)}


def main():
    parser = argparse.ArgumentParser(description="Parse condition A vLLM results")
    parser.add_argument("--results", required=True,
                        help="Path to raw vLLM output JSONL (data/condition_A/results/...)")
    parser.add_argument("--run",     default=None,
                        help="Run name for output filename (defaults to results file stem)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    run_name = args.run or results_path.stem
    print(f"Parsing {results_path.name}  →  run: {run_name}")

    stats = parse_results(results_path, run_name)

    print(f"\n{'='*50}")
    print(f"Total results : {stats['total']}")
    print(f"Valid JSON    : {stats['valid']}  ({stats['parse_rate']*100:.1f}%)")
    print(f"Errors        : {stats['errors']}")
    print(f"Saved to      : {stats['output']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
