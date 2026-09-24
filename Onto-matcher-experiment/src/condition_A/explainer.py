import json
from pathlib import Path


def build_explanations(run_name: str, data_dir: Path) -> dict[tuple, str]:
    scores_path = data_dir / "condition_A" / "scores" / f"{run_name}.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores not found: {scores_path}")

    explanations = {}
    for line in scores_path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        cv_id = row.get("cv_id")
        jd_id = row.get("jd_id")
        if not cv_id or not jd_id:
            continue
        justification = row.get("justification") or row.get("raw_completion") or ""
        explanations[(cv_id, jd_id)] = justification.strip()

    return explanations
