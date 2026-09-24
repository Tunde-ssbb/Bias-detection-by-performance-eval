#!/usr/bin/env python3
"""
Query experiment metadata and pipeline status.

Usage:
    python query.py                          # list all recorded runs
    python query.py --condition A            # condition A runs + job details
    python query.py --condition E            # condition E runs + pipeline stage
    python query.py --run <run_name>         # details on one specific run
    python query.py --delete <run_or_job_id> # delete run data + metadata
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

METADATA_FILE = ROOT / "data" / "metadata" / "runs.jsonl"
DATA_DIR      = ROOT / "data"


# ── Metadata helpers ──────────────────────────────────────────────────────────

def load_metadata(condition: str = None) -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    entries = []
    with open(METADATA_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if condition:
                cond = e.get("condition")
                if cond != condition:
                    continue
            entries.append(e)
    entries.sort(key=lambda x: x.get("timestamp") or x.get("submission_time", ""), reverse=True)
    return entries


def _find_by_run_or_job(key: str) -> list[dict]:
    """Return all metadata records matching run_name or job_id."""
    all_entries = load_metadata()
    by_run = [e for e in all_entries if e.get("run_name") == key]
    if by_run:
        return by_run
    by_job = [e for e in all_entries if str(e.get("job_id", "")) == key]
    return by_job


def _get_condition(entry: dict) -> str:
    return entry.get("condition", "?")


def _get_files_for_entry(entry: dict, job_id_only: bool = False) -> list[Path]:

    cond     = _get_condition(entry)
    run_name = entry.get("run_name", "")
    job_id   = str(entry.get("job_id", ""))
    if cond == "A":
        from src.condition_A.run import get_files_for_run
    elif cond == "E":
        sys.path.insert(0, str(ROOT / "src" / "condition_E"))
        from src.condition_E.run import get_files_for_run
    else:
        return []
    all_files = get_files_for_run(run_name)
    if job_id_only and job_id:
        return [f for f in all_files if job_id in f.name]
    return all_files



# ── Delete ────────────────────────────────────────────────────────────────────

def delete_run(key: str):
    """Delete all data files and metadata entries for a run name or job id."""
    matches = _find_by_run_or_job(key)

    if not matches:
        print(f"No metadata records found for '{key}'.")
        return

    if len(matches) > 1:
        run_names = {e.get("run_name") for e in matches}
        if len(run_names) > 1:
            print(f"\n⚠  WARNING: '{key}' matches {len(matches)} metadata records across "
                  f"{len(run_names)} different run names. This indicates a metadata issue.")
        else:
            print(f"\n⚠  WARNING: '{key}' matches {len(matches)} metadata records for the "
                  f"same run name. This likely means the run was double-recorded.")
        print("\nMatching records:")
        for e in matches:
            ts = (e.get("timestamp") or e.get("submission_time", "?"))[:19]
            print(f"  run={e.get('run_name')}  job={e.get('job_id')}  cond={_get_condition(e)}  ts={ts}")
        print()

    # When deleting by job_id and multiple records share the same run_name,
    # restrict file deletion to job-specific files (logs, run_meta) so that
    # shared data files (node_data, inst_data, scores…) are not wiped.
    all_run_names = {e.get("run_name") for e in load_metadata()}
    matched_run_names = {e.get("run_name") for e in matches}
    job_id_only = (
        not any(e.get("run_name") == key for e in matches) and  # keyed by job_id, not run_name
        any(                                                      # run_name has other surviving records
            rn in all_run_names and
            sum(1 for e in load_metadata() if e.get("run_name") == rn) > len(matches)
            for rn in matched_run_names
        )
    )

    # Collect all files across all matching records (deduplicated)
    all_files: list[Path] = []
    seen_paths: set[Path] = set()
    for entry in matches:
        for f in _get_files_for_entry(entry, job_id_only=job_id_only):
            if f not in seen_paths:
                all_files.append(f)
                seen_paths.add(f)

    if job_id_only:
        print("  (Deleting by job_id with other records sharing this run name — "
              "only job-specific log/meta files will be removed, not shared data files.)")

    if not all_files and not matches:
        print("Nothing to delete.")
        return

    print(f"Metadata records to remove: {len(matches)}")
    for e in matches:
        ts = (e.get("timestamp") or e.get("submission_time", "?"))[:19]
        print(f"  [{_get_condition(e)}] run={e.get('run_name')}  job={e.get('job_id')}  ts={ts}")

    if all_files:
        print(f"\nFiles to delete: {len(all_files)}")
        for p in sorted(all_files):
            print(f"  {p.relative_to(ROOT)}")
    else:
        print("\nNo data files found on disk (metadata only).")

    confirm = input(f"\nProceed with deletion? [y/N] ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Cancelled.")
        return

    for p in all_files:
        p.unlink()

    if METADATA_FILE.exists():
        run_names_to_remove = {e.get("run_name") for e in matches}
        job_ids_to_remove   = {str(e.get("job_id")) for e in matches}
        lines = METADATA_FILE.read_text().splitlines()
        kept  = []
        for line in lines:
            if not line.strip():
                continue
            e = json.loads(line)
            if (e.get("run_name") in run_names_to_remove and
                    str(e.get("job_id")) in job_ids_to_remove):
                continue
            kept.append(line)
        METADATA_FILE.write_text("\n".join(kept) + ("\n" if kept else ""))

    print(f"\nDeleted {len(all_files)} file(s) and {len(matches)} metadata record(s).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query experiment metadata and pipeline status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run", default=None,
                        help="Filter to a specific run name")
    parser.add_argument("--delete", metavar="RUN_OR_JOB_ID",
                        help="Delete all data and metadata for a run (by run name or job id)")
    args = parser.parse_args()

    if args.delete:
        delete_run(args.delete)
        return


    all_entries = load_metadata()
    if not all_entries:
        print("No runs recorded yet.")
        return

    print(f"\n── All recorded runs ({len(all_entries)} total) {'─' * 30}")
    print(f"\n{'Cond':<6} {'Job ID':<14} {'Timestamp':<20} {'Run name':<50} Status")
    print("─" * 110)

    for e in all_entries:
        p     = e.get("params") or e.get("config", {})
        cond  = _get_condition(e)
        job_id = e.get("job_id", "?")
        ts    = (e.get("timestamp") or e.get("submission_time", "?"))[:19]
        rn    = e.get("run_name") or p.get("run_name") or p.get("run") or \
                Path(p.get("infile", "")).stem or "?"

        if cond == "E":
            detail = "✓ scored" if (DATA_DIR / "condition_E" / "scores" / f"{rn}.jsonl").exists() else "submitted"
        
        else:
            detail = "✓ scored" if (DATA_DIR / "condition_A" / "scores" / f"{rn}.jsonl").exists() else "submitted"

        print(f"  {cond:<4} {job_id:<14} {ts:<20} {rn:<50} {detail}")


    print(f"Delete: python query.py --delete <run_name_or_job_id>")


if __name__ == "__main__":
    main()
