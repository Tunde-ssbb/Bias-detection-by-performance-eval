"""
Condition D explainer — same direct-match breakdown as condition B,
but reads from condition D's own graph_data and inst_data.
"""

import json, sys, os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from scoring_weights import IMPORTANCE_WEIGHTS

DEFAULT_WEIGHT = IMPORTANCE_WEIGHTS["important"]


def _explain_pair(cv_record: dict, jd_record: dict, instantiations: dict) -> str:
    jd_nodes = {
        n["id"]: n for n in jd_record.get("nodes", [])
        if not n.get("id", "").endswith("_anchor")
    }
    cv_nodes = {n["id"]: n for n in cv_record.get("nodes", [])}

    inst_by_jd: dict[str, list[tuple]] = {jd_id: [] for jd_id in jd_nodes}
    for (cv_id, jd_id), conf in instantiations.items():
        if jd_id in inst_by_jd:
            inst_by_jd[jd_id].append((cv_id, conf))

    lines = ["INSTANTIATION BREAKDOWN — condition D (type-grounded extraction, direct match)"]
    lines.append("=" * 60)

    matched = unmatched = 0
    for jd_id, jd_node in sorted(jd_nodes.items()):
        importance = jd_node.get("importance", "important")
        w = IMPORTANCE_WEIGHTS.get(importance, DEFAULT_WEIGHT)
        matches = sorted(inst_by_jd.get(jd_id, []), key=lambda x: -x[1])

        if matches:
            matched += 1
            best_cv_id, best_conf = matches[0]
            cv_label = cv_nodes.get(best_cv_id, {}).get("label", best_cv_id)
            cv_type  = cv_nodes.get(best_cv_id, {}).get("type", "")
            lines.append(
                f"\n✓ [{jd_node.get('type', '')}] [{importance}] {jd_node.get('label', jd_id)}"
                f"  →  +{w * best_conf:.2f}"
            )
            lines.append(f"    [{cv_type}] {cv_label}  (conf={best_conf:.2f})")
            for extra_cv_id, extra_conf in matches[1:]:
                cv_l = cv_nodes.get(extra_cv_id, {}).get("label", extra_cv_id)
                cv_t = cv_nodes.get(extra_cv_id, {}).get("type", "")
                lines.append(f"    [{cv_t}] {cv_l}  (conf={extra_conf:.2f})")
        else:
            unmatched += 1
            lines.append(
                f"\n✗ [{jd_node.get('type', '')}] [{importance}] {jd_node.get('label', jd_id)}"
                f"  →  -{w:.2f}  (no match)"
            )

    lines.append(f"\nSummary: {matched} matched, {unmatched} unmatched out of {len(jd_nodes)} JD nodes")
    return "\n".join(lines)


def build_explanations(run_name: str, data_dir: Path) -> dict[tuple, str]:
    graph_path = data_dir / "condition_E" / "graph_data" / f"{run_name}.json"
    inst_path  = data_dir / "condition_E" / "inst_data"  / f"{run_name}.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"graph_data not found: {graph_path}")
    if not inst_path.exists():
        raise FileNotFoundError(f"inst_data not found: {inst_path}")

    graph_data = json.loads(graph_path.read_text())
    inst_data  = json.loads(inst_path.read_text())
    records    = graph_data["records"]

    explanations = {}
    for pair in inst_data["pairs"].values():
        cv_id = pair["cv_id"]
        jd_id = pair["jd_id"]
        cv = records.get(cv_id)
        jd = records.get(jd_id)
        if cv is None or jd is None:
            continue
        instantiations = {
            tuple(k.split("|||")): v
            for k, v in pair["instantiations"].items()
        }
        explanations[(cv_id, jd_id)] = _explain_pair(cv, jd, instantiations)

    return explanations
