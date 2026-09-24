
import json
from typing import Dict, Any
from pathlib import Path

from scoring_weights import ED_EXP_IMPORTANCE  # noqa: F401


def _w(importance: float) -> float:
    """Round importance to nearest integer (matches IMPORTANCE_WEIGHTS linear scale)."""
    return float(round(importance))

def compute_max_score(jd_dict: Dict[str, Any]) -> float:
        # Calculate theoretical min and max scores based on JD importances
    max_score = 0.0
    min_score = 0.0

    for item in jd_dict.values():
        if item.get('kind', '') in ['required education', 'related experience']:
            # Special handling for education/experience items: importance is fixed
            importance = ED_EXP_IMPORTANCE
        else:
            importance = item.get('importance', 1.0)

        max_score += _w(importance)  # Matched
        min_score -= _w(importance)  # Unmatched
 
    
    return max_score, min_score

def normalize_score(raw_score: float, max_score: float, min_score: float) -> float:

    # Handle edge case where all importances are 0
    if max_score == min_score:
        return 5.5  # Middle of 1-10 range

    
    # Min-max normalization to [1, 10] range
    normalized = 1 + (raw_score - min_score) * (10 - 1) / (max_score - min_score)
    
    return max(1.0, min(10.0, normalized))




def compute_ground_truth_score(cv_dict: Dict[str, Any], jd_dict: Dict[str, Any], normalize: bool = True) -> float:
    """
    Compute ground truth score by comparing CV and JD profile dictionaries.

    """
    raw_score = 0.0
    max_score, min_score = compute_max_score(jd_dict)
    
    for jd_code, jd_item in jd_dict.items():
        importance = jd_item.get('importance', 1.0)
        kind = jd_item.get('kind', '')
        
        # Check if this is a special case (education or experience)
        if kind in ['required education', 'related experience']:
            matched = check_education_experience_match(jd_code, cv_dict, kind)
            importance = ED_EXP_IMPORTANCE  # Override importance for these items
        else:
            # Regular match - just check if code exists in cv_dict
            matched = jd_code in cv_dict
        
        if matched:
            raw_score += _w(importance)
        else:
            raw_score -= _w(importance)
    
    if normalize:
        return normalize_score(raw_score, max_score, min_score)
    else:
        return raw_score


def check_education_experience_match(jd_code: str, cv_dict: Dict[str, Any], kind: str) -> bool:

    # Extract prefix and integer from JD code
    parts = jd_code.rsplit('.', 1)
    if len(parts) != 2:
        # No final dot or can't split - fall back to exact match
        return jd_code in cv_dict
    
    jd_prefix, jd_level_str = parts
    try:
        jd_level = int(jd_level_str)
    except ValueError:
        # Not an integer after the last dot - fall back to exact match
        return jd_code in cv_dict
    
    # Look for matching prefix in cv_dict
    for cv_code, cv_item in cv_dict.items():
        # Check if this item has the same kind
        if cv_item.get('kind', '') != kind:
            continue
        
        # Extract prefix and integer from CV code
        cv_parts = cv_code.rsplit('.', 1)
        if len(cv_parts) != 2:
            continue
        
        cv_prefix, cv_level_str = cv_parts
        try:
            cv_level = int(cv_level_str)
        except ValueError:
            continue
        
        # Match if prefix is same and CV level >= JD level
        if cv_prefix == jd_prefix and cv_level >= jd_level:
            return True
    
    return False


def add_ground_truth_scores(
    scores_path: str,
    dataset_path: str,
    output_path: str = None,
) -> None:
    records = {
        r["custom_id"]: r
        for r in (json.loads(l) for l in open(dataset_path))
    }

    rows = [json.loads(l) for l in open(scores_path)]

    ok = errors = 0
    out_rows = []
    for row in rows:
        cv_rec = records.get(row.get("cv_id"))
        jd_rec = records.get(row.get("jd_id"))
        if cv_rec is None or jd_rec is None:
            print(f"  [warn] missing record for cv={row.get('cv_id')} jd={row.get('jd_id')}")
            row["gt_score"] = None
            errors += 1
        else:
            row["gt_score"] = compute_ground_truth_score(
                cv_rec["profile_dict"], jd_rec["profile_dict"]
            )
            ok += 1
        out_rows.append(row)

    out = Path(output_path or scores_path)
    out.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")

    valid = [r["gt_score"] for r in out_rows if r.get("gt_score") is not None]
    print(f"Ground truth scores: {ok} ok, {errors} missing → {out}")
    if valid:
        print(f"  mean={sum(valid)/len(valid):.2f}  min={min(valid):.2f}  max={max(valid):.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python ground_truth_scoring.py <scores.jsonl> <dataset.jsonl> [output.jsonl]")
        sys.exit(1)
    add_ground_truth_scores(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
