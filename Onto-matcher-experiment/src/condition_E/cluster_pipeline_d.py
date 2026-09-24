# main function of the pipeline for condition E, called by the. job template
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from condition_E.ontology import load_schema as _load_ontology_schema  

_dr_map  = None
_ancestors_fn = None


# read relations from schema
def load_rel_schema():
    global _dr_map, _ancestors_fn
    if _dr_map is not None:
        return
    schema = _load_ontology_schema()
    _dr_map = {}
    for bucket in schema["relations"].values():
        for r in bucket:
            _dr_map[r["label"]] = (r.get("domain", ""), r.get("range", ""))

    super_map = schema.get("superclass_map", {})
    cache: dict = {}

    # set up generic ancestory function 
    def ancestors(t: str) -> set:
        if t in cache:
            return cache[t]
        visited, queue = set(), [t]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            parents = super_map.get(cur, [])
            if isinstance(parents, str):
                parents = [parents]
            queue.extend(parents)
        cache[t] = visited
        return visited

    _ancestors_fn = ancestors

# validate edges
def _fix_edges(edges: list, nodes: list) -> list:
    load_rel_schema()
    node_map = {n["id"]: n for n in nodes}
    out = []
    for e in edges:
        rel = e.get("relation", "")

        # skip isA relations and onds without domain and range definitions
        if rel == "isA" or rel not in _dr_map:
            out.append(e)
            continue
        domain, range_ = _dr_map[rel]
        if not domain and not range_:
            out.append(e)
            continue

        # check src and domain types of relatiion
        src_type = node_map.get(e["source"], {}).get("type", "")
        tgt_type = node_map.get(e["target"], {}).get("type", "")
        src_ok = not domain or domain in _ancestors_fn(src_type)
        tgt_ok = not range_  or range_  in _ancestors_fn(tgt_type)
        if src_ok and tgt_ok:
            out.append(e)
        # also allow if reverse direction is valid (flip edge)
        elif (not domain or domain in _ancestors_fn(tgt_type)) and \
             (not range_  or range_  in _ancestors_fn(src_type)):
            out.append({**e, "source": e["target"], "target": e["source"]})
    return out


# ── Completion parsing (shared with condition C) ──────────────────────────────

# def _detect_truncation(raw: str) -> bool:
#     stripped = raw.strip()
#     return bool(stripped) and not stripped.endswith("}")


# def _parse_jsonl_completion(raw: str) -> tuple:
#     if not raw or not raw.strip():
#         return [], "empty_completion"

#     lines = raw.strip().splitlines()
#     if lines and lines[0].startswith("```"):
#         lines = lines[1:]
#     if lines and lines[-1].strip() == "```":
#         lines = lines[:-1]

#     truncated = _detect_truncation(raw)
#     items = []
#     for i, line in enumerate(lines):
#         line = line.strip()
#         if not line:
#             continue
#         if truncated and i == len(lines) - 1:
#             continue
#         try:
#             items.append(json.loads(line))
#         except json.JSONDecodeError:
#             pass

#     if not items:
#         return [], "truncated" if truncated else "no_json"
#     return items, None


def _print_summary(step: str, ok: list, failed: dict):
    total = len(ok) + sum(len(v) for v in failed.values())
    print(f"\n── {step} summary ({'─' * (50 - len(step))})")
    print(f"  ok      : {len(ok)}/{total}")
    if failed:
        for mode, ids in sorted(failed.items()):
            print(f"  {mode:<20}: {len(ids)} record(s)")
            for rid in ids:
                print(f"    {rid}")
    else:
        print("  no failures")


# json and path helpers
def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _node_data_path(run_name: str, data_root: Path) -> Path:
    return data_root / "node_data" / f"{run_name}.json"

def _graph_data_path(run_name: str, data_root: Path) -> Path:
    return data_root / "graph_data" / f"{run_name}.json"

def _inst_data_path(run_name: str, data_root: Path) -> Path:
    return data_root / "inst_data" / f"{run_name}.json"


def _iter_results(results_path: str):
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Phase 1a: Generate NER prompts ─────────────────────────

def cmd_gen_node_prompts(dataset_path: str, run_name: str, data_dir: str = "data"):
    from extractor_e import build_ner_prompt

    data_root = Path(data_dir) / "condition_E"
    out_path  = data_root / "prompts" / f"{run_name}_nodes.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    #load exsiting node_data 
    node_data = _load_json(_node_data_path(run_name, data_root),
                           {"run_name": run_name, "records": {}})

    with open(dataset_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # build prompts for each record, skip if already processed
    skipped = written = 0
    with open(out_path, "w") as f:
        for rec in records:
            rid  = rec["custom_id"]
            side = rec["profile_type"]

            if node_data["records"].get(rid, {}).get("node_status") == "ok":
                skipped += 1
                continue

            #actual prompt build  and dump records to file
            prompt = build_ner_prompt(side, rec["profile_text"])
            f.write(json.dumps({
                "id":                  f"node::{rid}",
                "prompt":              prompt,
                "record_id":           rid,
                "profile_type":        side,
                "occ_code":            rec["occ_code"],
                "style":               rec["style"],
                "variant_id":          rec.get("variant_id", "base"),
                "counterfactual":      rec.get("counterfactual_key") or rec.get("cf"),
                "counterfactual_info": rec.get("counterfactual_info", ""),
                "profile_dict":        rec.get("profile_dict", {}),
                "profile_text":        rec["profile_text"],
            }) + "\n")
            written += 1

    print(f"NER prompts: {written} written, {skipped} skipped (already ok) → {out_path}")


# ── Step 1b: Parse NER results → node_data ──────────────────────

def cmd_parse_nodes(run_name: str, results_path: str, data_dir: str = "data"):
    from condition_E.graph_utils import inject_anchor_nodes
    from extractor_e import parse_ner_response

    data_root = Path(data_dir) / "condition_E"
    node_data = _load_json(_node_data_path(run_name, data_root),
                           {"run_name": run_name, "records": {}})

    meta_by_record: dict = {}
    succeeded: dict = defaultdict(list)   # record_id → [node dicts]
    failed: dict = defaultdict(list)

    for item in _iter_results(results_path):
        pid = item.get("id", "")
        if not pid.startswith("node::"):
            continue

        # truncate node prefix from id and get meta
        record_id = pid[len("node::"):]

        meta_by_record[record_id] = {
            "profile_type":        item["profile_type"],
            "occ_code":            item["occ_code"],
            "style":               item["style"],
            "variant_id":          item.get("variant_id", "base"),
            "counterfactual":      item.get("counterfactual"),
            "counterfactual_info": item.get("counterfactual_info", ""),
            "profile_dict":        item.get("profile_dict", {}),
            "profile_text":        item.get("profile_text", ""),
        }

        if "error" in item:
            failed["vllm_error"].append(record_id)
            continue

        raw = item.get("completion", "")
        if not raw.strip():
            failed["empty_response"].append(record_id)
            continue

        # parse nodes and append to to 
        side  = item.get("profile_type", "cv")
        nodes = parse_ner_response(raw, side, item.get("profile_text", ""))
        succeeded[record_id].extend(nodes)

    ok_ids: list = []
    for record_id, nodes in succeeded.items():
        if node_data["records"].get(record_id, {}).get("node_status") == "ok":
            continue

        meta = meta_by_record[record_id]
        # add achor node
        nodes = inject_anchor_nodes(nodes, meta["profile_type"])

        node_data["records"][record_id] = {
            "custom_id":   record_id,
            **meta,
            "nodes":       nodes,
            "node_status": "ok",
            "node_error":  None,
        }
        ok_ids.append(record_id)

    #save parsed nodes
    _save_json(node_data, _node_data_path(run_name, data_root))
    _print_summary("parse-nodes", ok_ids, failed)
    print(f"\nnode_data → {_node_data_path(run_name, data_root)}")

    if ok_ids:
        _build_graph_data(run_name, data_root, node_data, new_ids=set(ok_ids))


def _build_graph_data(run_name: str, data_root: Path, node_data: dict,
                      new_ids: "set | None" = None):
    # Load existing graph_data to preserve rel_status, edges, and other fields
    # written by later steps (cmd_parse_rel, cmd_parse_importance).
    graph_data = _load_json(_graph_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})
    included = skipped = 0

    for record_id, rec in node_data["records"].items():
        if rec.get("node_status") != "ok":
            skipped += 1
            continue
        # Only insert/update records that are new, preserve existing ones so
        # rel_status, edges, and importance already set by downstream steps survive.
        if record_id in graph_data["records"] and (new_ids is None or record_id not in new_ids):
            included += 1
            continue
        graph_data["records"][record_id] = {
            "custom_id":           record_id,
            "profile_type":        rec["profile_type"],
            "occ_code":            rec["occ_code"],
            "style":               rec["style"],
            "variant_id":          rec.get("variant_id", "base"),
            "counterfactual":      rec.get("counterfactual"),
            "counterfactual_info": rec.get("counterfactual_info", ""),
            "profile_dict":        rec.get("profile_dict", {}),
            "profile_text":        rec.get("profile_text", ""),
            "nodes":               rec["nodes"],
            # edges absent until cmd_parse_rel sets it (used as done-sentinel)
        }
        included += 1

    _save_json(graph_data, _graph_data_path(run_name, data_root))
    print(f"graph_data: {included} records, {skipped} excluded → {_graph_data_path(run_name, data_root)}")


# ── Step 1c: Generate importance qualification prompts (JD only) ────────────

def cmd_gen_importance_prompts(run_name: str, data_dir: str = "data"):
    from extractor_e import build_importance_prompt

    data_root   = Path(data_dir) / "condition_E"
    node_data   = _load_json(_node_data_path(run_name, data_root),
                             {"run_name": run_name, "records": {}})
    prompts_dir = data_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / f"{run_name}_importance.jsonl"

    n_written = n_skip_done = n_skip_empty = 0
    with out_path.open("w") as f:
        for record_id, rec in node_data["records"].items():
            if rec.get("node_status") != "ok":
                continue
            if rec.get("profile_type") != "jd":
                continue
            nodes = [n for n in rec["nodes"] if not n.get("id", "").endswith("_anchor")]
            if not nodes:
                n_skip_empty += 1
                continue
            # Skip if already processed (explicit flag, or all nodes already have importance)
            if rec.get("importance_status") == "ok" or all(n.get("importance") for n in nodes):
                n_skip_done += 1
                continue

            #prompt build and record meta
            prompt = build_importance_prompt(nodes, rec.get("profile_text", ""))
            entry  = {
                "id":           f"imp::{record_id}",
                "record_id":    record_id,
                "profile_type": rec["profile_type"],
                "occ_code":     rec.get("occ_code", ""),
                "style":        rec.get("style", ""),
                "variant_id":   rec.get("variant_id", "base"),
                "prompt":       prompt,
            }
            f.write(json.dumps(entry) + "\n")
            n_written += 1

    print(f"gen-importance-prompts: {n_written} written, "
          f"{n_skip_done} already done, {n_skip_empty} empty → {out_path}")


# ── Step 1c.6: Parse importance results → fill in jd side node data  and save ─────

def cmd_parse_importance(run_name: str, results_path: str, data_dir: str = "data"):
    from extractor_e import parse_importance_response

    data_root  = Path(data_dir) / "condition_E"
    node_data  = _load_json(_node_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})
    graph_data = _load_json(_graph_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})

    n_ok = n_fail = 0
    for item in _iter_results(results_path):
        pid = item.get("id", "")
        if not pid.startswith("imp::"):
            continue
        record_id = pid[len("imp::"):]

        if "error" in item:
            n_fail += 1
            continue
        raw = item.get("completion", "")
        if not raw.strip():
            n_fail += 1
            continue

        rec = node_data["records"].get(record_id)
        if not rec:
            n_fail += 1
            continue

        if rec.get("importance_status") == "ok":
            n_ok += 1
            continue

        nodes = rec["nodes"]
        non_anchor = [n for n in nodes if not n.get("id", "").endswith("_anchor")]
        # parse responses
        parse_importance_response(raw, non_anchor)
        rec["importance_status"] = "ok"

        if record_id in graph_data["records"]:
            graph_data["records"][record_id]["nodes"] = nodes

        n_ok += 1

    _save_json(node_data, _node_data_path(run_name, data_root))
    _save_json(graph_data, _graph_data_path(run_name, data_root))
    print(f"parse-importance: {n_ok} records updated, {n_fail} failed")


# ── Step 1d: Generate relation extraction prompts ─────────────────────────────

def cmd_gen_rel_prompts(run_name: str, data_dir: str = "data"):
    from extractor_e import build_rel_prompt

    data_root  = Path(data_dir) / "condition_E"
    graph_data = _load_json(_graph_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})

    out_path = data_root / "prompts" / f"{run_name}_rel.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w") as f:
        for record_id, rec in graph_data["records"].items():
            # rel_status flag (new runs) or edges key present (old runs that ran parse-rel)
            if rec.get("rel_status") == "ok" or "edges" in rec:
                skipped += 1
                continue
            side  = rec["profile_type"]
            nodes = rec.get("nodes", [])
            if not nodes:
                skipped += 1
                continue
            prompt = build_rel_prompt(side, nodes, original_text=rec.get("profile_text", ""))
            entry = {
                "id":           f"rel::{record_id}",
                "prompt":       prompt,
                "profile_type": side,
                "occ_code":     rec.get("occ_code", ""),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    print(f"Relation prompts: {written} written, {skipped} skipped → {out_path}")


# ── Step 1e: Parse relation results → graph_data edges ────────────────────────

def cmd_parse_rel(run_name: str, results_path: str, data_dir: str = "data"):
    from extractor_e import parse_rel_response

    data_root  = Path(data_dir) / "condition_E"
    graph_data = _load_json(_graph_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})

    ok_ids: list = []
    failed: dict = defaultdict(list)

    for item in _iter_results(results_path):
        pid = item.get("id", "")
        if not pid.startswith("rel::"):
            continue

        record_id = pid[len("rel::"):]
        rec = graph_data["records"].get(record_id)
        if rec is None:
            failed["unknown_record"].append(record_id)
            continue

        if rec.get("rel_status") == "ok":
            ok_ids.append(record_id)
            continue

        if "error" in item:
            failed["vllm_error"].append(record_id)
            continue

        raw = item.get("completion", "")
        if not raw.strip():
            failed["empty_response"].append(record_id)
            continue

        edges, shadow_nodes, isa_edges = parse_rel_response(raw, rec.get("nodes", []))
        existing_ids = {n["id"] for n in rec.get("nodes", [])}
        for sn in shadow_nodes:
            if sn["id"] not in existing_ids:
                rec.setdefault("nodes", []).append(sn)
                existing_ids.add(sn["id"])
        rec["edges"] = _fix_edges(edges + isa_edges, rec.get("nodes", []))
        rec["rel_status"] = "ok"
        ok_ids.append(record_id)

    _save_json(graph_data, _graph_data_path(run_name, data_root))
    _print_summary("parse-rel", ok_ids, failed)
    print(f"\ngraph_data → {_graph_data_path(run_name, data_root)}")


# ── Step 2a: Generate instantiation prompts ───────────────────────────────────

def cmd_gen_inst_prompts(run_name: str, data_dir: str = "data", pairs_mode: str = "occ_match"):
    from extractor_e import build_inst_prompt
    from shared.pairer import get_pairs

    data_root  = Path(data_dir) / "condition_E"
    graph_data = _load_json(_graph_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})
    inst_data  = _load_json(_inst_data_path(run_name, data_root),
                            {"run_name": run_name, "pairs": {}})
    done_pairs = set(inst_data["pairs"].keys())

    records  = graph_data["records"]
    out_path = data_root / "prompts" / f"{run_name}_inst.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pair_ids  = get_pairs(list(records.values()), pairs_mode)
    all_pairs = [(records[cv_id], records[jd_id]) for cv_id, jd_id in pair_ids
                 if cv_id in records and jd_id in records]

    written = skipped_done = 0
    with open(out_path, "w") as f:
        for cv, jd in all_pairs:
            pair_key = f"{cv['custom_id']}|||{jd['custom_id']}"
            if pair_key in done_pairs:
                skipped_done += 1
                continue

            cv_nodes = cv["nodes"]
            jd_nodes = jd["nodes"]
            prompt   = build_inst_prompt(cv_nodes, jd_nodes)

            f.write(json.dumps({
                "id":       f"inst::{pair_key}",
                "prompt":   prompt,
                "cv_id":    cv["custom_id"],
                "jd_id":    jd["custom_id"],
                "occ_code": cv["occ_code"],
                "cv_nodes": cv_nodes,
                "jd_nodes": jd_nodes,
            }) + "\n")
            written += 1

    print(f"Instantiation prompts: {written} written, "
          f"{skipped_done} pairs skipped (already done) → {out_path}")


# ── Step 2b: Parse instantiation results ─────────────────────────────────────

def cmd_parse_inst(run_name: str, results_path: str, data_dir: str = "data"):
    from extractor_e import parse_inst_response

    data_root  = Path(data_dir) / "condition_E"
    inst_data  = _load_json(_inst_data_path(run_name, data_root),
                            {"run_name": run_name, "pairs": {}})
    graph_data = _load_json(_graph_data_path(run_name, data_root),
                            {"run_name": run_name, "records": {}})
    records = graph_data["records"]

    ok_ids: list = []
    failed: dict = defaultdict(list)

    def _merge_shadows(record_id, shadow_nodes, isa_edges):
        rec = records.get(record_id)
        if rec is None:
            return
        existing_ids = {n["id"] for n in rec.get("nodes", [])}
        for sn in shadow_nodes:
            if sn["id"] not in existing_ids:
                rec.setdefault("nodes", []).append(sn)
                existing_ids.add(sn["id"])
        rec.setdefault("edges", [])
        existing_edges = {(e["source"], e["target"], e["relation"]) for e in rec["edges"]}
        for ie in isa_edges:
            key = (ie["source"], ie["target"], ie["relation"])
            if key not in existing_edges:
                rec["edges"].append(ie)
                existing_edges.add(key)

    for item in _iter_results(results_path):
        pid = item.get("id", "")
        if not pid.startswith("inst::"):
            continue

        pair_key = pid[len("inst::"):]

        if pair_key in inst_data["pairs"]:
            continue

        if "error" in item:
            failed["vllm_error"].append(pid)
            continue

        raw = item.get("completion", "")
        if not raw.strip():
            failed["empty_response"].append(pid)
            continue

        cv_id    = item.get("cv_id", "")
        jd_id    = item.get("jd_id", "")
        cv_nodes = item.get("cv_nodes", [])
        jd_nodes = item.get("jd_nodes", [])
        # Filter to the same node set used in build_inst_prompt so ENT indices align
        cv_nodes_prompt = [n for n in cv_nodes if n.get("source") not in ("inferred", "anchor")]
        jd_nodes_prompt = [n for n in jd_nodes if n.get("source") not in ("inferred", "anchor")]
        matches, cv_shadows, cv_isa_edges, jd_shadows, jd_isa_edges = \
            parse_inst_response(raw, cv_nodes_prompt, jd_nodes_prompt)

        _merge_shadows(cv_id, cv_shadows, cv_isa_edges)
        _merge_shadows(jd_id, jd_shadows, jd_isa_edges)

        insts = {f"{m['cv_id']}|||{m['jd_id']}": 1.0 for m in matches}

        inst_data["pairs"][pair_key] = {
            "cv_id":          cv_id,
            "jd_id":          jd_id,
            "occ_code":       item.get("occ_code", ""),
            "instantiations": insts,
        }
        ok_ids.append(pair_key)

    _save_json(inst_data, _inst_data_path(run_name, data_root))
    _save_json(graph_data, _graph_data_path(run_name, data_root))
    _print_summary("parse-inst", ok_ids, failed)
    print(f"\ninst_data → {_inst_data_path(run_name, data_root)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condition D batch pipeline — generate/parse prompts for vLLM inference"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gen-node-prompts",        action="store_true", help="Step 1a: NER prompts (one per doc, single-stage)")
    group.add_argument("--parse-nodes",             action="store_true", help="Step 1b: parse NER results → node_data/ + graph_data/")
    group.add_argument("--gen-importance-prompts",  action="store_true", help="Step 1c.5: generate importance qualification prompts (JD only)")
    group.add_argument("--parse-importance",        action="store_true", help="Step 1c.6: parse importance results → annotate node_data + graph_data")
    group.add_argument("--gen-rel-prompts",         action="store_true", help="Step 1d: generate relation extraction prompts from graph_data/")
    group.add_argument("--parse-rel",               action="store_true", help="Step 1e: parse relation results → graph_data/ edges")
    group.add_argument("--gen-inst-prompts",        action="store_true", help="Step 2a: generate instantiation prompts")
    group.add_argument("--parse-inst",              action="store_true", help="Step 2b: parse instantiation results → inst_data/")

    parser.add_argument("--run",        required=True)
    parser.add_argument("--dataset",    help="Dataset JSONL (--gen-node-prompts only)")
    parser.add_argument("--results",    help="vLLM results JSONL (--parse-* steps)")
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--pairs-mode", default="occ_match", choices=["occ_match", "all"])

    args = parser.parse_args()

    if args.gen_node_prompts:
        if not args.dataset:
            parser.error("--gen-node-prompts requires --dataset")
        cmd_gen_node_prompts(args.dataset, args.run, args.data_dir)

    elif args.parse_nodes:
        if not args.results:
            parser.error("--parse-nodes requires --results")
        cmd_parse_nodes(args.run, args.results, args.data_dir)

    elif args.gen_importance_prompts:
        cmd_gen_importance_prompts(args.run, args.data_dir)

    elif args.parse_importance:
        if not args.results:
            parser.error("--parse-importance requires --results")
        cmd_parse_importance(args.run, args.results, args.data_dir)

    elif args.gen_rel_prompts:
        cmd_gen_rel_prompts(args.run, args.data_dir)

    elif args.parse_rel:
        if not args.results:
            parser.error("--parse-rel requires --results")
        cmd_parse_rel(args.run, args.results, args.data_dir)

    elif args.gen_inst_prompts:
        cmd_gen_inst_prompts(args.run, args.data_dir, args.pairs_mode)

    elif args.parse_inst:
        if not args.results:
            parser.error("--parse-inst requires --results")
        cmd_parse_inst(args.run, args.results, args.data_dir)
