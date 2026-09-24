"""
Analysis pipeline for bias-in-HRM experiments.

For each completed run (scores file exists), produces:
  data/analysed/<run_name>_analysed.jsonl   — cv/jd pairs with scores + explanation
  data/metric_results/<run_name>_metrics.txt — aggregated ranking metrics

Usage:
    python analyse.py                        # analyse all completed runs
    python analyse.py <run_name> [...]       # analyse specific runs
    python analyse.py --ks 5 10
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "shared"))

from src.shared.ground_truth_scoring import add_ground_truth_scores
from src.shared.ranking_metrics import compute_ranking_metrics, format_metrics
from src.shared.scoring_weights import ED_EXP_IMPORTANCE

METADATA_FILE = ROOT / "data" / "metadata" / "runs.jsonl"
ANALYSED_DIR  = ROOT / "data" / "analysed"
METRICS_DIR   = ROOT / "data" / "metric_results"

K_VALUES = [9]
BIAS_K   = [3]


# ── Scores paths ──────────────────────────────────────────────────────────────

def _scores_path(condition: str, run_name: str) -> Path:
    if condition == "A":
        return ROOT / "data" / "condition_A" / "scores" / f"{run_name}.jsonl"
    if condition == "E":
        return ROOT / "data" / "condition_E" / "scores" / f"scores_{run_name}.jsonl"
    raise ValueError(f"Unknown condition: {condition}")



def _load_all_meta() -> list[dict]:
    if not METADATA_FILE.exists():
        return []
    return [json.loads(l) for l in METADATA_FILE.open() if l.strip()]


# retrieve path for dataset used in a run.
def _find_dataset_path(condition: str, run_name: str, all_meta: list[dict]) -> str:
    meta = next((e for e in reversed(all_meta)
                 if e.get("run_name") == run_name and e.get("condition") == condition), None)
    if meta is None:
        raise ValueError(f"No metadata for run '{run_name}' condition {condition}")

    params = meta.get("params", {})
    if "input" in params:
        dataset_path = params["input"]
    else:
        src = params.get("copied_from")
        src_meta = next((e for e in reversed(all_meta)
                         if e.get("run_name") == src), None) if src else None
        if src_meta and "input" in src_meta.get("params", {}):
            dataset_path = src_meta["params"]["input"]
        else:
            raise ValueError(f"No 'input' param in metadata for run '{run_name}'")

    local = ROOT / "data" / "counterfactual" / Path(dataset_path).name
    if not Path(dataset_path).exists() and local.exists():
        dataset_path = str(local)
    return dataset_path


# find all completed runs from metadata (scores exist)
def completed_runs(all_meta: list[dict]) -> list[tuple[str, str]]:
    seen = set()
    runs = []
    for e in reversed(all_meta):
        cond = e.get("condition")
        name = e.get("run_name")
        if not cond or not name or (cond, name) in seen:
            continue
        seen.add((cond, name))
        try:
            if _scores_path(cond, name).exists():
                runs.append((cond, name))
        except ValueError:
            pass
    return runs


# ── Score loading ─────────────────────────────────────────────────────────────

def _gt_breakdown(cv_dict: dict, jd_dict: dict) -> dict:
    from src.shared.ground_truth_scoring import check_education_experience_match, ED_EXP_IMPORTANCE
    matched, unmatched = [], []
    for code, item in jd_dict.items():
        kind = item.get("kind", "")
        imp  = ED_EXP_IMPORTANCE if kind in ("required education", "related experience") else item.get("importance", 1.0)
        hit  = (check_education_experience_match(code, cv_dict, kind)
                if kind in ("required education", "related experience")
                else code in cv_dict)
        entry = {"code": code, "name": item.get("name", code), "kind": kind, "importance": round(imp, 2)}
        (matched if hit else unmatched).append(entry)
    matched.sort(key=lambda x: -x["importance"])
    unmatched.sort(key=lambda x: -x["importance"])
    return {"matched": matched, "unmatched": unmatched}


def _load_scores_with_gt(condition: str, run_name: str, dataset_path: str) -> pd.DataFrame:
    # get score rows
    rows = [json.loads(l) for l in _scores_path(condition, run_name).open() if l.strip()]
    # get original data rows.
    records = {r["custom_id"]: r for r in (json.loads(l) for l in open(dataset_path) if l.strip())}

    from src.shared.ground_truth_scoring import compute_ground_truth_score
    for row in rows:
        cv_rec = records.get(row.get("cv_id"))
        jd_rec = records.get(row.get("jd_id"))
        if cv_rec and jd_rec:
            #compute gt score and score breakdown (set of matched and unmatched items)
            row["gt_score"]    = compute_ground_truth_score(cv_rec["profile_dict"], jd_rec["profile_dict"])
            row["gt_breakdown"] = _gt_breakdown(cv_rec["profile_dict"], jd_rec["profile_dict"])
        else:
            row["gt_score"] = row["gt_breakdown"] = None

    df = pd.DataFrame(rows)
    if "cv_id" in df.columns:
        df["group_id"] = df["cv_id"].str.replace(r"::cf_\w+$", "", regex=True)
    return df




import re as _re

# partial Onet -> coreo mapping label: ({onet kinds}, {cv types}, {jd types})
_CATEGORIES: dict[str, tuple[set, set, set]] = {
    "skill":      ({"skill"},         {"Skill"},                          {"SkillType"}),
    "tech_skill": ({"tech_skill"},    {"Resource"},                       {"ResourceType"}),
    "knowledge":  ({"knowledge"},     {"Knowledge"},                      {"KnowledgeType"}),
    "attitude":   ({"abilities"}, {"Attitude"}, {"AttitudeType"}),
    "task":       ({"task"},          {"HumanTask"},                      {"TaskType"}),
    "required_education": ({"required_education"}, {}, {}),
    "related_experience": ({"related_experience"}, {}, {}),
    "work_style": ({"work_style"}, {}, {}),
}

_INST_TYPE_PAIRS: set[tuple[str, str]] = {
    ("Skill", "SkillType"), ("Resource", "ResourceType"), ("Knowledge", "KnowledgeType"),
    ("Attitude", "AttitudeType"), ("HumanTrait", "AttitudeType"),
    ("HumanQuality", "AttitudeType"), ("HumanTask", "TaskType"),
}

_IMP_THRESHOLDS = [
    (4.0, "extremely important"), (3.0, "very important"), (2.0, "important"),
    (1.0, "somewhat important"),  (0.0, "not important"),
]

def _float_to_level(v: float) -> str:
    for threshold, label in _IMP_THRESHOLDS:
        if v > threshold:
            return label
    return "not important"

def _tokens(s: str) -> set[str]:
    return set(_re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())

def _is_match(onet_name: str, node_label: str, threshold: float = 0.8) -> bool:
    # check f o*net item name and extracted label overlap
    a, b = _tokens(onet_name), _tokens(node_label)
    if not a or not b:
        return False
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter <= longer and len(shorter) / len(longer) >= 0.5:
        return True
    return len(a & b) / len(a | b) >= threshold

def _graph_data_path(condition: str, run_name: str) -> "Path | None":
    if condition != "E":
        return None
    p = ROOT / "data" / "condition_E" / "graph_data" / f"{run_name}.json"
    return p if p.exists() else None


# get per document failure analyses
def compute_competence_extraction_overlap(condition: str, run_name: str) -> "dict | None":

    # collect the graph data
    graph_path = _graph_data_path(condition, run_name)
    if graph_path is None:
        return None

    data    = json.loads(graph_path.read_text())
    records = data.get("records", {})
    _IMP_LABELS = ["extremely important", "very important", "important", "somewhat important", "not important"]
    zero = lambda: {"onet_total": 0, "span_matched": 0, "type_matched": 0,
                    "imp_by_label": {lbl: {"correct": 0, "any_lbl_extracted": 0, "total": 0} for lbl in _IMP_LABELS}}
    result: dict[str, dict] = {cat: {"cv": zero(), "jd": zero()} for cat in _CATEGORIES}

    # for each document
    for rec in records.values():
        # get onet items
        pd_items = rec.get("profile_dict", {})

        #get extracted nodes
        nodes    = rec.get("nodes", [])
        side     = "cv" if rec.get("custom_id", "").startswith("cv::") else "jd"


        for cat, (onet_kinds, cv_types, jd_types) in _CATEGORIES.items():
            correct_types = cv_types if side == "cv" else jd_types
            # get onet items of this category

            if cat not in ("required_education", "related_experience"):
                onet_items = [
                    (v.get("name", "") or v.get("description", ""), v.get("importance" , 1.0))
                    for v in pd_items.values()
                    if v.get("kind") in onet_kinds
                ]
            else:
                onet_items = [
                    (v.get("name", "") or v.get("description", ""), ED_EXP_IMPORTANCE)
                    for v in pd_items.values()
                    if v.get("kind") in onet_kinds
                ]
            if not onet_items:
                continue

            
            all_labels = [(n["label"], n.get("type", ""), n.get("importance", "")) for n in nodes if n.get("label")]

            bucket = result[cat][side]
            bucket["onet_total"] += len(onet_items)

            for onet_name, onet_imp in onet_items:
                # find corresponding node for each onet item
                best_lbl, best_type, best_importance = next(
                    ((lbl, ntype, nimp) for lbl, ntype, nimp in all_labels if _is_match(onet_name, lbl)),
                    (None, None, None)
                )
                if best_lbl is None:
                    continue


                bucket["span_matched"] += 1
                #check type match 
                if best_type in correct_types:
                    bucket["type_matched"] += 1

                #check importance match of best matched node
          
                if side == "jd" and onet_imp is not None:
    
                    expected = _float_to_level(onet_imp)
             
                    b = bucket["imp_by_label"][expected]
                    b["total"] += 1
                    if best_importance != "" or best_importance is not None:
                        b["any_lbl_extracted"] += 1
                    if best_importance == expected:
                        b["correct"] += 1

    return result


# instantiation accuracy (for overview

def compute_inst_accuracy(condition: str, run_name: str) -> "dict | None":
    if condition != "E":
        return None

    inst_path  = ROOT / "data" / "condition_E" / "inst_data"  / f"{run_name}.json"
    graph_path = ROOT / "data" / "condition_E" / "graph_data" / f"{run_name}.json"
    if not inst_path.exists() or not graph_path.exists():
        return None

    inst_data  = json.loads(inst_path.read_text())
    graph_data = json.loads(graph_path.read_text())
    records    = graph_data["records"]

    node_index = {n["id"]: n for rec in records.values() for n in rec.get("nodes", [])}

    totals = {"gt_total": 0, "span_both": 0, "type_compat": 0,
              "inst_found": 0, "inst_typed": 0, "inst_total": 0, "inst_correct": 0}

    for pair in inst_data["pairs"].values():
        cv_rec  = records.get(pair["cv_id"], {})
        jd_rec  = records.get(pair["jd_id"], {})
        cv_dict = cv_rec.get("profile_dict", {})
        jd_dict = jd_rec.get("profile_dict", {})
        if not cv_dict or not jd_dict:
            continue

        def _extract_nodes(rec):
            return [n for n in rec.get("nodes", [])
                    if n.get("label") and n.get("source") not in ("inferred", "anchor")]

        cv_nodes = _extract_nodes(cv_rec)
        jd_nodes = _extract_nodes(jd_rec)

        inst_pairs = []
        for key in pair.get("instantiations", {}):
            cv_nid, jd_nid = key.split("|||")
            cv_n, jd_n = node_index.get(cv_nid, {}), node_index.get(jd_nid, {})
            if cv_n.get("label") and jd_n.get("label"):
                inst_pairs.append((cv_n, jd_n))
        totals["inst_total"] += len(inst_pairs)

        all_cv = [(n["label"], n.get("type", ""), n["id"]) for n in cv_nodes]
        all_jd = [(n["label"], n.get("type", ""), n["id"]) for n in jd_nodes]
        gt_valid: set[tuple[str, str]] = set()

        for code, jd_val in jd_dict.items():
            if code not in cv_dict:
                continue
            gt_name = jd_val.get("name") or jd_val.get("description", "")
            if not gt_name:
                continue
            kind = jd_val.get("kind")

            cat_match = next(
                ((cv_types, jd_types)
                 for _, (onet_kinds, cv_types, jd_types) in _CATEGORIES.items()
                 if kind in onet_kinds),
                None,
            )
            if cat_match is None:
                totals["gt_total"] += 1
                cv_s = [nid for _, _, nid in all_cv if _is_match(gt_name, _)]
                # simpler: just check if both sides have a span match
                cv_s = [nid for lbl, _, nid in all_cv if _is_match(gt_name, lbl)]
                jd_s = [nid for lbl, _, nid in all_jd if _is_match(gt_name, lbl)]
                if cv_s and jd_s:
                    totals["span_both"] += 1
                    if any(cv_n["id"] in cv_s and jd_n["id"] in jd_s for cv_n, jd_n in inst_pairs):
                        totals["inst_found"] += 1
                continue

            cv_types, jd_types = cat_match
            totals["gt_total"] += 1

            cv_s = [(lbl, t, nid) for lbl, t, nid in all_cv if _is_match(gt_name, lbl)]
            jd_s = [(lbl, t, nid) for lbl, t, nid in all_jd if _is_match(gt_name, lbl)]
            if not cv_s or not jd_s:
                continue
            totals["span_both"] += 1

            if any((ct, jt) in _INST_TYPE_PAIRS for _, ct, _ in cv_s for _, jt, _ in jd_s):
                totals["type_compat"] += 1

            cv_ids = {nid for _, _, nid in cv_s}
            jd_ids = {nid for _, _, nid in jd_s}
            for cv_nid in cv_ids:
                for jd_nid in jd_ids:
                    gt_valid.add((cv_nid, jd_nid))

            found = next(
                ((cv_n, jd_n) for cv_n, jd_n in inst_pairs
                 if cv_n["id"] in cv_ids and jd_n["id"] in jd_ids),
                None,
            )
            if found:
                totals["inst_found"] += 1
                if found[0].get("type") in cv_types and found[1].get("type") in jd_types:
                    totals["inst_typed"] += 1

        totals["inst_correct"] += sum(
            1 for cv_n, jd_n in inst_pairs if (cv_n["id"], jd_n["id"]) in gt_valid
        )

    return {"totals": totals}


# ── Pipeline overview ───

def format_overview(run_name: str, condition: str,
                    skill_acc: "dict | None",
                    inst_acc:  "dict | None",
                    rel_gain_pct: "float | None" = None) -> str:
    def pct(n, d): return f"{n/d:.1%}" if d else "—"

    lines = ["", "=" * 76, f"  PIPELINE OVERVIEW  —  {run_name}  (condition {condition})", "=" * 76]

    ner_span_n = ner_span_d = ner_type_n = ner_type_d = imp_correct = imp_total = imp_any_lbl_extracted = 0
    _IMP_LABELS = ["extremely important", "very important", "important", "somewhat important", "not important"]

    if skill_acc:
        for cat_data in skill_acc.values():
            if not isinstance(cat_data, dict):
                continue
            for side, sd in cat_data.items():
                if not isinstance(sd, dict):
                    continue
                ner_span_d += sd.get("onet_total",   0)
                ner_span_n += sd.get("span_matched", 0)
                ner_type_d += sd.get("span_matched", 0)
                ner_type_n += sd.get("type_matched", 0)
                if side == "jd":
                    for lbl in _IMP_LABELS:
                        b = sd.get("imp_by_label", {}).get(lbl, {})
                        imp_correct += b.get("correct", 0)
                        imp_any_lbl_extracted += b.get("any_lbl_extracted", 0)
                        imp_total   += b.get("total",   0)

    inst_n = inst_d = inst_prec_n = inst_prec_d = 0
    if inst_acc:
        t = inst_acc.get("totals", {})
        inst_d      = t.get("type_compat",  0)
        inst_n      = t.get("inst_found",   0)
        inst_prec_d = t.get("inst_total",   0)
        inst_prec_n = t.get("inst_correct", 0)

    lines.append(f"  NER span      {pct(ner_span_n, ner_span_d):>7}  ({ner_span_n}/{ner_span_d} O*NET items detected as any span)")
    lines.append(f"  NER type      {pct(ner_type_n, ner_type_d):>7}  ({ner_type_n}/{ner_type_d} detected spans with correct type)")
    lines.append(f"  IMP label     {pct(imp_correct, imp_total):>7}  ({imp_correct}/{imp_total}  importance labels correct ({imp_any_lbl_extracted}/{imp_total} items for which a. label was extracted))")
    if inst_acc:
        lines.append(f"  INST recall   {pct(inst_n, inst_d):>7}  ({inst_n}/{inst_d} type-compatible pairs instantiated)")
        lines.append(f"  INST prec     {pct(inst_prec_n, inst_prec_d):>7}  ({inst_prec_n}/{inst_prec_d} extracted pairs that are correct)")
    else:
        lines.append(f"  INST recall        —  (condition {condition}: no inst)")
        lines.append(f"  INST prec          —")
    if rel_gain_pct is not None:
        sign = "+" if rel_gain_pct >= 0 else ""
        lines.append(f"  P@9 rel gain  {sign}{rel_gain_pct:.1f}%  (relations vs no-relations, discount=0)")
    lines.append("=" * 76)
    return "\n".join(lines)



# recompute P@k with relation discount = 0
def _rescore_norel(condition: str, run_name: str) -> "dict | None":
    if condition != "E":
        return None
    scores_p = _scores_path(condition, run_name)
    detail_p = scores_p.parent / (scores_p.stem + "_detail.jsonl")
    if not detail_p.exists():
        print(f"   [warn] no detail file for norel ablation: {detail_p.name}")
        return None

    norel: dict[tuple[str, str], float] = {}
    for line in detail_p.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        detail = rec.get("detail", {})
        total_w = raw = 0.0
        for node_info in detail.values():
            w = node_info.get("weight", 3.0)
            total_w += w
            node_score = 1.0 if node_info.get("hop_distance") == 1 else 0.0
            raw += w * (2 * node_score - 1)
        score = 5.5 if total_w == 0 else round(max(1.0, min(10.0, 1 + (raw + total_w) * 9 / (2 * total_w))), 3)
        norel[(rec["cv_id"], rec["jd_id"])] = score
    return norel


def _load_explainer(condition: str):
    import importlib.util
    paths = {
        "A": ROOT / "src" / "condition_A" / "explainer.py",
        "E": ROOT / "src" / "condition_E" / "explainer.py",
    }
    path = paths.get(condition)
    if path is None:
        raise ValueError(f"Unknown condition: {condition}")
    spec = importlib.util.spec_from_file_location(f"explainer_{condition}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_explanations



def analyse_run(condition: str, run_name: str, all_meta: list[dict],
                k_values: list[int] = K_VALUES, bias_k: list[int] = BIAS_K):
    print(f"\n── {run_name}  (condition {condition})")

    dataset_path = _find_dataset_path(condition, run_name, all_meta)
    print(f"   dataset  : {Path(dataset_path).name}")

    df = _load_scores_with_gt(condition, run_name, dataset_path)
    df = df.dropna(subset=["score", "gt_score"])
    print(f"   pairs    : {len(df)}")

    # print(f"   building explanations ...")
    # try:
    #     explanations = _load_explainer(condition)(run_name, ROOT / "data")
    # except Exception as e:
    #     import traceback; traceback.print_exc()
    #     print(f"   [warn] explanations unavailable: {e}")
    #     explanations = {}

    ANALYSED_DIR.mkdir(parents=True, exist_ok=True)
    analysed_path = ANALYSED_DIR / f"{run_name}_analysed.jsonl"
    with analysed_path.open("w") as f:
        for _, row in df.iterrows():
            out = {
                "run_name":       run_name,       "condition":      condition,
                "cv_id":          row.get("cv_id"),  "jd_id":       row.get("jd_id"),
                "occ_code":       row.get("occ_code"), "cv_style":  row.get("cv_style"),
                "jd_style":       row.get("jd_style"), "counterfactual": row.get("counterfactual"),
                "variant_id":     row.get("variant_id"), "score":   row.get("score"),
                "gt_score":       row.get("gt_score"), "gt_breakdown": row.get("gt_breakdown"),
                # "explanation":    explanations.get((row["cv_id"], row["jd_id"]), ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"   analysed → {analysed_path.relative_to(ROOT)}")

    agg   = compute_ranking_metrics(df, k_values, bias_k=bias_k)
    n_jds = df["jd_id"].nunique() if "jd_id" in df.columns else 0

    # rescore with relation discount =0 to see effect of relations on P@k
    norel_scores = _rescore_norel(condition, run_name)
    rel_gain = None
    if norel_scores is not None:
        df_norel = df.copy()
        df_norel["score"] = df_norel.apply(
            lambda r: norel_scores.get((r["cv_id"], r["jd_id"]), r["score"]), axis=1
        )
        agg_norel = compute_ranking_metrics(df_norel, k_values, bias_k=None)
        for k in k_values:
            m          = f"precision@{k}"
            full_mean  = agg.get(m, {}).get("mean")
            norel_mean = agg_norel.get(m, {}).get("mean")
            if full_mean is not None and norel_mean is not None and norel_mean != 0:
                gain = (full_mean - norel_mean) / abs(norel_mean) * 100
                agg[f"{m}_rel_gain_pct"] = {"mean": gain, "ci_lo": float("nan"),
                                             "ci_hi": float("nan"), "n": agg.get(m, {}).get("n", 0)}
        rel_gain = agg.get("precision@9_rel_gain_pct", {}).get("mean")

    # failure analysis metrics
    skill_acc = compute_competence_extraction_overlap(condition, run_name)
    inst_acc  = compute_inst_accuracy(condition, run_name)

    overview_text = format_overview(run_name, condition, skill_acc, inst_acc, rel_gain_pct=rel_gain)
    metrics_text  = format_metrics(agg, run_name, condition, k_values, n_jds, len(df), bias_k=bias_k)
    full_text     = overview_text + "\n" + metrics_text
    print(full_text)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{run_name}_metrics.txt"
    metrics_path.write_text(full_text + "\n")
    print(f"   metrics  → {metrics_path.relative_to(ROOT)}")

    dataset_name = Path(dataset_path).stem
    (METRICS_DIR / f"{run_name}_metrics.json").write_text(json.dumps({
        "condition": condition, "run_name": run_name,
        "model": _infer_model(run_name), "dataset": dataset_name, "agg": agg,
    }, indent=2))

    return agg, dataset_name


def _infer_model(run_name: str) -> str:
    for shorthand in ("llama70b", "llama8b", "qwen2.5", "deepseek8b",
                      "gpt4omini", "claude-sonnet"):
        if shorthand in run_name.lower():
            return {"gpt4omini": "gpt-4o-mini"}.get(shorthand, shorthand)
    return run_name


def load_all_agg_records(metrics_dir: Path) -> list[tuple[str, str, str, str, dict]]:
    records = []
    for p in sorted(metrics_dir.glob("*_metrics.json")):
        try:
            d = json.loads(p.read_text())
            records.append((d["condition"], d["run_name"], d["model"], d["dataset"], d["agg"]))
        except Exception:
            continue
    return records


def write_comparison_csv(records: list, path: Path, dataset_filter: str | None = None) -> None:
    """Write a paired A-vs-E CSV with per-metric mean/CI/delta columns."""
    if not records:
        return
    if dataset_filter:
        records = [r for r in records if r[3] == dataset_filter]

    from collections import defaultdict
    grouped: dict[tuple, dict] = defaultdict(dict)
    for condition, _, model, dataset, agg in records:
        grouped[(dataset, model)][condition] = agg

    pairs = [(key, v) for key, v in grouped.items() if "A" in v and "E" in v]
    if not pairs:
        print("\n   [warn] no A+E pairs found — CSV not written")
        return

    all_metrics: list[str] = []
    seen: set[str] = set()
    for _, cond_map in pairs:
        for agg in cond_map.values():
            for k in agg:
                if k not in seen:
                    all_metrics.append(k); seen.add(k)

    fieldnames = ["dataset", "model"]
    for m in all_metrics:
        fieldnames += [f"{m}_A_mean", f"{m}_A_ci_lo", f"{m}_A_ci_hi",
                       f"{m}_E_mean", f"{m}_E_ci_lo", f"{m}_E_ci_hi", f"{m}_delta"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (dataset, model), cond_map in sorted(pairs):
            agg_a, agg_e = cond_map["A"], cond_map["E"]
            row: dict = {"dataset": dataset, "model": model}
            for m in all_metrics:
                ea, ee = agg_a.get(m, {}), agg_e.get(m, {})
                ma, me = ea.get("mean", float("nan")), ee.get("mean", float("nan"))
                delta  = (me - ma) if not (math.isnan(ma) or math.isnan(me)) else float("nan")
                row[f"{m}_A_mean"]  = round(ma,                            6)
                row[f"{m}_A_ci_lo"] = round(ea.get("ci_lo", float("nan")), 6)
                row[f"{m}_A_ci_hi"] = round(ea.get("ci_hi", float("nan")), 6)
                row[f"{m}_E_mean"]  = round(me,                            6)
                row[f"{m}_E_ci_lo"] = round(ee.get("ci_lo", float("nan")), 6)
                row[f"{m}_E_ci_hi"] = round(ee.get("ci_hi", float("nan")), 6)
                row[f"{m}_delta"]   = round(delta,                         6)
            w.writerow(row)
    print(f"\n   comparison CSV → {path}")



# main entry
def main():

    # find all cpmpleted runs
    all_meta = _load_all_meta()

    all_completed_runs = completed_runs(all_meta)

    #analyse all runs
    print(f"Analysing {len(all_completed_runs)} run(s)")
    for condition, run_name in all_completed_runs:
        try:
            analyse_run(condition, run_name, all_meta)
        except Exception as e:
            print(f"[error] {run_name}: {e}", file=sys.stderr)
            raise

    #aggregate metirc into comparison CSVs
    all_records = load_all_agg_records(METRICS_DIR)
    for dataset in dict.fromkeys(r[3] for r in all_records):
        write_comparison_csv(all_records, METRICS_DIR / f"comparison_AE_{dataset}.csv",
                             dataset_filter=dataset)


if __name__ == "__main__":
    main()
