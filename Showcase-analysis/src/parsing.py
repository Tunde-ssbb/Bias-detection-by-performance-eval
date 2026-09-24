"""
Load raw LLM output files, extracts scores from completions (handling
incomplete/truncated outputs), and joins each row to its matching input
profile (CV text, CV dict, JD text, JD dict).

Parsed results are saved to data/parsed_output/<experiment_name>.parquet.
"""

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Parse-status labels (attached per row so callers can filter/inspect)
# ---------------------------------------------------------------------------
STATUS_OK        = "ok"          # clean JSON parse, score + justification present
STATUS_RECOVERED = "recovered"   # score extracted via regex from malformed/truncated output
STATUS_NO_SCORE  = "no_score"    # output present but score could not be extracted
STATUS_EMPTY     = "empty"       # completion field was missing or blank
STATUS_NO_INPUT  = "no_input"    # cv_id or jd_id had no matching profile in input file


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_batch_name(filename: str) -> str:
    """First underscore-delimited token of the filename is the batch name."""
    return filename.split("_")[0]


def _load_input_profiles(
    batch_name: str,
    input_dir: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Load CV and JD profiles from the input file for a batch.
    Returns (cv_profiles, jd_profiles) keyed by custom_id.
    """
    input_file = input_dir / f"final_{batch_name}.jsonl"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    cv_profiles: dict[str, dict] = {}
    jd_profiles: dict[str, dict] = {}

    for record in _load_jsonl(input_file):
        cid = record.get("custom_id", "")
        if cid.startswith("cv::"):
            cv_profiles[cid] = record
        elif cid.startswith("jd::"):
            jd_profiles[cid] = record

    return cv_profiles, jd_profiles


def _parse_completion(raw: str) -> tuple[float | None, str, str]:
    """
    Extract (score, justification, status) from a raw completion string.

    Tries in order:
      1. Strip markdown fences, parse as JSON  → STATUS_OK
      2. Regex for "score": <number>           → STATUS_RECOVERED
      3. Give up                               → STATUS_NO_SCORE / STATUS_EMPTY
    """
    if not raw or not raw.strip():
        return None, "", STATUS_EMPTY

    text = raw.strip()

    # Strip complete markdown fences: ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Strip opening fence without closing (truncated output)
    elif text.startswith("```"):
        text = text[text.find("\n") + 1:].rstrip("`\n ")

    # try Full JSON parse
    try:
        data = json.loads(text)
        score = data.get("score")
        justification = str(data.get("justification", ""))
        if score is not None:
            return float(score), justification, STATUS_OK
    except (json.JSONDecodeError, ValueError):
        pass

    # try regex fallback — score is the priority even if JSON is truncated
    score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if score_match:
        score = float(score_match.group(1))
        just_match = re.search(r'"justification"\s*:\s*"([^"]*)', text)
        justification = (just_match.group(1) + "…[TRUNCATED]") if just_match else "[TRUNCATED]"
        return score, justification, STATUS_RECOVERED

    return None, "[PARSE_FAILED]", STATUS_NO_SCORE


def _resolve_files(name: str, ref: str | Path, output_dir: Path) -> list[Path]:
    """Return list of .jsonl output files for a registry entry."""
    if isinstance(ref, Path):
        return [ref]
    matches = sorted(output_dir.glob(f"*{ref}*.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"No .jsonl files containing job ID '{ref}' found in {output_dir} "
            f"(experiment: '{name}')"
        )
    return matches


# ---------------------------------------------------------------------------
# Core: parse one experiment
# ---------------------------------------------------------------------------

def parse_experiment(
    name: str,
    ref: str | Path,
    data_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse all output files for one experiment and return a joined DataFrame.
    """
    output_dir = data_dir / "output"
    input_dir  = data_dir / "input"

    files = _resolve_files(name, ref, output_dir)


    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Experiment : {name}")
        print(f"  Files      : {[f.name for f in files]}")

    all_frames: list[pd.DataFrame] = []

    for filepath in files:
        raw_rows = _load_jsonl(filepath)
        n_total  = len(raw_rows)

        # Load matching input profiles (keyed by batch name)
        batch_name = _extract_batch_name(filepath.name)
        try:
            cv_profiles, jd_profiles = _load_input_profiles(batch_name, input_dir)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            cv_profiles, jd_profiles = {}, {}

        # Counters for verbose report
        n_ok        = 0
        n_recovered = 0
        n_no_score  = 0
        n_empty     = 0
        n_no_input  = 0

        records: list[dict[str, Any]] = []

        for row in raw_rows:
            completion_raw = row.get("completion", "")
            score, justification, status = _parse_completion(completion_raw)

            cv_id = row.get("cv_id", "")
            jd_id = row.get("jd_id", "")

            cv_profile = cv_profiles.get(cv_id, {})
            jd_profile = jd_profiles.get(jd_id, {})

            # Downgrade status if profile lookup failed
            if (cv_id and not cv_profile) or (jd_id and not jd_profile):
                status = STATUS_NO_INPUT
                n_no_input += 1
            elif status == STATUS_OK:
                n_ok += 1
            elif status == STATUS_RECOVERED:
                n_recovered += 1
            elif status == STATUS_NO_SCORE:
                n_no_score += 1
            elif status == STATUS_EMPTY:
                n_empty += 1

            records.append({
                # Output metadata
                "id":                row.get("id"),
                "model":             row.get("model"),
                "backend":           row.get("backend"),
                "seed":              row.get("seed"),
                "max_new_tokens":    row.get("max_new_tokens"),
                "max_chars":         row.get("max_chars"),
                "n":                 row.get("n"),
                "variant_id":        row.get("variant_id"),
                "cv_id":             cv_id,
                "jd_id":             jd_id,
                "occ_code":          row.get("occ_code"),
                "occ_title":         row.get("occ_title"),
                "counterfactual":    row.get("counterfactual"),
                "counterfactual_info": row.get("counterfactual_info"),
                "cv_batch":          row.get("cv_batch"),
                "cv_style":          row.get("cv_style"),
                "jd_batch":          row.get("jd_batch"),
                "jd_style":          row.get("jd_style"),
                # Extracted from completion
                "score":             score,
                "justification":     justification,
                "parse_status":      status,
                # Input profile data (joined)
                "cv_profile_text":   cv_profile.get("profile_text", ""),
                "cv_profile_dict":   json.dumps(cv_profile.get("profile_dict", {})),
                "jd_profile_text":   jd_profile.get("profile_text", ""),
                "jd_profile_dict":   json.dumps(jd_profile.get("profile_dict", {})),
                # Raw completion for debugging
                "completion_raw":    completion_raw,
            })

        if verbose:
            n_problem = n_recovered + n_no_score + n_empty + n_no_input
            print(f"\n  File: {filepath.name}")
            print(f"Total rows      : {n_total}")
            print(f"Clean parses    : {n_ok}  ({100*n_ok/n_total:.1f}%)")
            print(f"Recovered       : {n_recovered}  — score via regex, truncated output")
            print(f" No score        : {n_no_score}  — completion present but unparseable")
            print(f"Empty           : {n_empty}  — completion field missing/blank")
            print(f"Missing profile : {n_no_input}  — cv_id or jd_id not in input file")
            print(f"total no. of issues   : {n_problem}")

        all_frames.append(pd.DataFrame(records))

    df = pd.concat(all_frames, ignore_index=True)

    if verbose:
        n_total_all = len(df)
        n_with_score = df["score"].notna().sum()
        print(f"\n  Total across all files : {n_total_all} rows")
        print(f"  Rows with a score      : {n_with_score} ({100*n_with_score/n_total_all:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# Public entry point called from main.py
# ---------------------------------------------------------------------------

def parse_experiments(
    experiments: dict[str, str | Path],
    data_dir: Path,
    parse: str = "all_new",
    verbose: bool = True,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:

    if out_dir is None:
        out_dir = data_dir / "parsed_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which experiments to parse
    if parse == "all":
        targets = dict(experiments)
    elif parse == "all_new":
        existing = {p.stem for p in out_dir.glob("*.parquet")}
        targets = {k: v for k, v in experiments.items() if k not in existing}
        if verbose and not targets:
            print("[parse] all_new: no new experiments to parse (all already in parsed_output/)")
        elif verbose:
            print(f"[parse] all_new: {len(targets)} new experiment(s) to parse")
    else:
        if parse not in experiments:
            raise ValueError(
                f"Unknown experiment '{parse}'. "
                f"Available: {list(experiments.keys())}"
            )
        targets = {parse: experiments[parse]}

    parsed: dict[str, pd.DataFrame] = {}

    print(f"Parsing experiments: {[name for name in targets]}")
    for name, ref in targets.items():
        
        try:
            df = parse_experiment(name, ref, data_dir, verbose=verbose)
            save_path = out_dir / f"{name}.parquet"
            df.to_parquet(save_path, index=False)
            if verbose:
                print(f"  Saved → {save_path}")
            parsed[name] = df
        except Exception as exc:
            print(f"\n[parse] ERROR — {name}: {exc}")

    if verbose and parsed:
        print(f"\n[parse] Done. {len(parsed)} experiment(s) saved to {out_dir}/")

    return parsed


# ---------------------------------------------------------------------------
# Load already-parsed experiments from parquet (no re-parsing)
# ---------------------------------------------------------------------------

def load_parsed(
    experiments: dict[str, str | Path],
    data_dir: Path,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    if out_dir is None:
        out_dir = data_dir / "parsed_output"

    loaded: dict[str, pd.DataFrame] = {}

    for name in experiments:
        path = out_dir / f"{name}.parquet"
        if path.exists():
            loaded[name] = pd.read_parquet(path)
        else:
            print(f"[load_parsed] WARNING: no parquet for '{name}' maybe it is not parsed yet")

    return loaded
