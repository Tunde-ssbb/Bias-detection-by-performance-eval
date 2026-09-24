"""
Condition E pipeline — local execution with batch LLM calls.

Phases:
  --extract      NER batch → importance batch (JD only) → REL batch
  --instantiate  INST batch
  --score        score pairs using graph-propagation formula

Each phase submits all prompts at once, waits for the batch to complete,
then parses all results before moving to the next phase.
"""

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from condition_E.ontology import load_schema as _load_ontology_schema  # noqa: E402

_dr_map:      "dict | None" = None
_ancestors_fn = None


def _load_rel_schema():
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


def _fix_edges(edges: list, nodes: list) -> list:
    _load_rel_schema()
    node_map = {n["id"]: n for n in nodes}
    out = []
    for e in edges:
        rel = e.get("relation", "")
        if rel == "isA" or rel not in _dr_map:
            out.append(e)
            continue
        domain, range_ = _dr_map[rel]
        if not domain and not range_:
            out.append(e)
            continue
        src_type = node_map.get(e["source"], {}).get("type", "")
        tgt_type = node_map.get(e["target"], {}).get("type", "")
        src_ok = not domain or domain in _ancestors_fn(src_type)
        tgt_ok = not range_  or range_  in _ancestors_fn(tgt_type)
        if src_ok and tgt_ok:
            out.append(e)
        elif (not domain or domain in _ancestors_fn(tgt_type)) and \
             (not range_  or range_  in _ancestors_fn(src_type)):
            out.append({**e, "source": e["target"], "target": e["source"]})
    return out


_client = None
_MODEL  = "claude-opus-4-6"


def set_model(model: str) -> None:
    global _MODEL
    _MODEL = model


def _get_client():
    global _client
    if _client is None:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env", override=False)
        load_dotenv(_ROOT.parent / "onto-matcher" / ".env", override=False)
        if _MODEL.startswith("gpt-") or _MODEL.startswith("o1") or _MODEL.startswith("o3"):
            import openai
            _client = openai.OpenAI()
        else:
            import anthropic
            _client = anthropic.Anthropic()
    return _client


def _is_openai() -> bool:
    return _MODEL.startswith("gpt-") or _MODEL.startswith("o1") or _MODEL.startswith("o3")


# ── Batch helpers ─────────────────────────────────────────────────────────────

def _submit_batch_openai(requests: list[dict]) -> str:
    """requests: list of {custom_id, messages}. Returns batch_id."""
    import tempfile, os
    client = _get_client()
    lines = []
    for req in requests:
        lines.append(json.dumps({
            "custom_id": req["custom_id"],
            "method":    "POST",
            "url":       "/v1/chat/completions",
            "body": {
                "model":       _MODEL,
                "messages":    req["messages"],
                "temperature": 0,
                "max_tokens":  8000,
            },
        }))
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        tmp = f.name
    try:
        with open(tmp, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    finally:
        os.unlink(tmp)
    print(f"  [batch] submitted {len(requests)} requests → batch_id={batch.id}")
    return batch.id


def _poll_batch_openai(batch_id: str) -> dict[str, str]:
    """Poll until done. Returns {custom_id: completion_text}."""
    client = _get_client()
    start = time.time()
    try:
        while True:
            batch = client.batches.retrieve(batch_id)
            counts = batch.request_counts
            elapsed = int(time.time() - start)
            mins, secs = divmod(elapsed, 60)
            msg = (f"  [batch] status={batch.status}  "
                   f"completed={counts.completed}/{counts.total}  failed={counts.failed}  "
                   f"elapsed={mins:02d}:{secs:02d}")
            print(f"\r{msg:<90}", end="", flush=True)
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                print()
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n  [batch] cancelling {batch_id} ...", flush=True)
        try:
            client.batches.cancel(batch_id)
            print(f"  [batch] cancelled.")
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # 409 means batch already completed — fall through to collect results
            batch = client.batches.retrieve(batch_id)
            if batch.status == "completed":
                print(f"  [batch] already completed — collecting results.")
            else:
                print(f"  [batch] cancel request failed: {e}")
                raise KeyboardInterrupt
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} ended with status={batch.status}")
    content = client.files.content(batch.output_file_id).text
    results = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid  = rec["custom_id"]
        body = rec.get("response", {}).get("body", {})
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        finish = body.get("choices", [{}])[0].get("finish_reason", "")
        if finish == "length":
            print(f"  [warn] '{cid}' truncated at max_tokens")
        results[cid] = text
    return results


def _submit_batch_anthropic(requests: list[dict]) -> str:
    """requests: list of {custom_id, messages}. Returns batch_id."""
    client = _get_client()
    batch_requests = [
        {
            "custom_id": req["custom_id"],
            "params": {
                "model":       _MODEL,
                "max_tokens":  8000,
                "messages":    req["messages"],
            },
        }
        for req in requests
    ]
    batch = client.messages.batches.create(requests=batch_requests)
    print(f"  [batch] submitted {len(requests)} requests → batch_id={batch.id}")
    return batch.id


def _poll_batch_anthropic(batch_id: str) -> dict[str, str]:
    """Poll until done. Returns {custom_id: completion_text}."""
    client = _get_client()
    start = time.time()
    try:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
            elapsed = int(time.time() - start)
            mins, secs = divmod(elapsed, 60)
            msg = (f"  [batch] status={batch.processing_status}  "
                   f"succeeded={counts.succeeded}/{total}  errored={counts.errored}  "
                   f"elapsed={mins:02d}:{secs:02d}")
            print(f"\r{msg:<90}", end="", flush=True)
            if batch.processing_status == "ended":
                print()
                break
            time.sleep(300)
    except KeyboardInterrupt:
        print(f"\n  [batch] cancelling {batch_id} ...", flush=True)
        try:
            client.messages.batches.cancel(batch_id)
            print(f"  [batch] cancelled.")
        except Exception as e:
            print(f"  [batch] cancel request failed: {e}")
        raise
    results = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = msg.content[0].text.strip()
            if msg.stop_reason == "max_tokens":
                print(f"  [warn] '{cid}' truncated at max_tokens")
            results[cid] = text
        else:
            print(f"  [warn] '{cid}' failed: {result.result.type}")
    return results


def _run_batch(requests: list[dict]) -> dict[str, str]:
    """Submit a batch and block until results are ready. Returns {custom_id: text}."""
    if not requests:
        return {}
    if _is_openai():
        batch_id = _submit_batch_openai(requests)
        return _poll_batch_openai(batch_id)
    else:
        batch_id = _submit_batch_anthropic(requests)
        return _poll_batch_anthropic(batch_id)


def _save_step(data_root: Path, run_name: str, step: str,
               requests: list[dict], results: dict[str, str],
               extra_meta: dict[str, dict] | None = None) -> None:
    """Persist prompts and raw completions for a pipeline step."""
    prompt_dir = data_root / "prompts"
    result_dir = data_root / "results"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompt_dir / f"{run_name}_{step}.jsonl"
    result_path = result_dir / f"{run_name}_{step}_local.jsonl"

    with prompt_path.open("a") as pf, result_path.open("a") as rf:
        for req in requests:
            cid  = req["custom_id"]
            meta = (extra_meta or {}).get(cid, {})
            pf.write(json.dumps({"id": cid, "prompt": req["messages"], **meta},
                                ensure_ascii=False) + "\n")
            rf.write(json.dumps({"id": cid, "model": _MODEL,
                                 "completion": results.get(cid, ""), **meta},
                                ensure_ascii=False) + "\n")


def _to_messages(prompt: list[dict] | str) -> list[dict]:
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": prompt}]


def _parse_jsonl(raw: str) -> list[dict]:
    lines = raw.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    non_empty = [l for l in lines if l.strip()]
    truncated = bool(non_empty) and not non_empty[-1].strip().endswith("}")
    items = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if truncated and i == len(non_empty) - 1:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return items


# ── Phase 1: Extract nodes ────────────────────────────────────────────────────

def cmd_extract(dataset_path: str, run_name: str, data_dir: str = "data"):
    from condition_E.graph_utils import inject_anchor_nodes
    from extractor_e import (
        build_ner_prompt,
        parse_ner_response,
        build_importance_prompt,
        parse_importance_response,
        build_rel_prompt,
        parse_rel_response,
    )

    data_root  = Path(data_dir) / "condition_E"
    node_path  = data_root / "node_data"  / f"{run_name}.json"
    graph_path = data_root / "graph_data" / f"{run_name}.json"
    for p in (node_path, graph_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    node_data  = json.loads(node_path.read_text())  if node_path.exists()  else {"run_name": run_name, "records": {}}
    graph_data = json.loads(graph_path.read_text()) if graph_path.exists() else {"run_name": run_name, "records": {}}

    with open(dataset_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    already_done = set(node_data["records"].keys())
    todo = [rec for rec in records if rec["custom_id"] not in already_done]
    print(f"NER: {len(todo)} to process, {len(already_done)} already done")

    # ── NER batch ────────────────────────────────────────────────────────────
    if todo:
        ner_requests = []
        for rec in todo:
            side = "cv" if rec["custom_id"].startswith("cv::") else "jd"
            ner_requests.append({
                "custom_id": rec["custom_id"],
                "messages":  _to_messages(build_ner_prompt(side, rec["profile_text"])),
            })
        ner_results = _run_batch(ner_requests)
        _save_step(data_root, run_name, "nodes", ner_requests, ner_results,
                   {rec["custom_id"]: {"occ_code": rec.get("occ_code",""),
                                       "profile_type": "cv" if rec["custom_id"].startswith("cv::") else "jd"}
                    for rec in todo})

        # Parse NER and store nodes (without relations yet)
        ner_nodes: dict[str, list] = {}
        for rec in todo:
            rid  = rec["custom_id"]
            raw  = ner_results.get(rid, "")
            side = "cv" if rid.startswith("cv::") else "jd"
            nodes = parse_ner_response(raw, side, rec["profile_text"])
            nodes = inject_anchor_nodes(nodes, side)
            print(f"  NER {rid}: {len(nodes)} nodes")
            ner_nodes[rid] = nodes

        # ── Importance batch (JD only) ────────────────────────────────────────
        jd_todo = [rec for rec in todo if not rec["custom_id"].startswith("cv::")]
        if jd_todo:
            imp_requests = [
                {
                    "custom_id": rec["custom_id"],
                    "messages":  _to_messages(
                        build_importance_prompt(ner_nodes[rec["custom_id"]], rec["profile_text"])
                    ),
                }
                for rec in jd_todo if ner_nodes.get(rec["custom_id"])
            ]
            imp_results = _run_batch(imp_requests)
            _save_step(data_root, run_name, "importance", imp_requests, imp_results)
            for rec in jd_todo:
                rid = rec["custom_id"]
                raw = imp_results.get(rid, "")
                ner_nodes[rid] = parse_importance_response(raw, ner_nodes[rid])
                n_imp = sum(1 for n in ner_nodes[rid] if n.get("importance"))
                print(f"  importance {rid}: {n_imp}/{len(ner_nodes[rid])} labelled")

        # ── REL batch ─────────────────────────────────────────────────────────
        rel_requests = []
        for rec in todo:
            rid  = rec["custom_id"]
            side = "cv" if rid.startswith("cv::") else "jd"
            nodes = ner_nodes[rid]
            rel_requests.append({
                "custom_id": rid,
                "messages":  _to_messages(
                    build_rel_prompt(side, nodes, original_text=rec["profile_text"])
                ),
            })
        rel_results = _run_batch(rel_requests)
        _save_step(data_root, run_name, "rel", rel_requests, rel_results)

        # Parse REL and write to node/graph data
        for rec in todo:
            rid   = rec["custom_id"]
            side  = "cv" if rid.startswith("cv::") else "jd"
            nodes = ner_nodes[rid]
            raw   = rel_results.get(rid, "")
            edges, shadow_nodes, isa_edges = parse_rel_response(raw, nodes)

            existing_ids = {n["id"] for n in nodes}
            for sn in shadow_nodes:
                if sn["id"] not in existing_ids:
                    nodes.append(sn)
                    existing_ids.add(sn["id"])
            all_edges = _fix_edges(edges + isa_edges, nodes)
            print(f"  REL {rid}: {len(edges)} relations, {len(shadow_nodes)} shadow nodes")

            meta = {
                "custom_id":          rid,
                "profile_type":       "cv" if rid.startswith("cv::") else "jd",
                "occ_code":           rec.get("occ_code", ""),
                "style":              rec.get("style", ""),
                "variant_id":         rec.get("variant_id", "base"),
                "counterfactual":     rec.get("counterfactual_key") or rec.get("counterfactual", ""),
                "counterfactual_info":rec.get("counterfactual_info", ""),
                "profile_dict":       rec.get("profile_dict", {}),
                "profile_text":       rec.get("profile_text", ""),
            }
            node_data["records"][rid]  = {"nodes": nodes, **meta}
            graph_data["records"][rid] = {"nodes": nodes, "edges": all_edges, **meta}

        node_path.write_text(json.dumps(node_data,  ensure_ascii=False))
        graph_path.write_text(json.dumps(graph_data, ensure_ascii=False))

    print(f"Extraction done → {node_path}")


# ── Phase 2: Instantiate ──────────────────────────────────────────────────────

def cmd_instantiate(run_name: str, data_dir: str = "data", pairs_mode: str = "occ_match"):
    from extractor_e import build_inst_prompt, parse_inst_response

    data_root  = Path(data_dir) / "condition_E"
    graph_path = data_root / "graph_data" / f"{run_name}.json"
    inst_path  = data_root / "inst_data"  / f"{run_name}.json"
    inst_path.parent.mkdir(parents=True, exist_ok=True)

    if not graph_path.exists():
        raise FileNotFoundError(f"graph_data not found — run --extract first: {graph_path}")

    graph_data = json.loads(graph_path.read_text())
    records    = graph_data["records"]
    inst_data  = json.loads(inst_path.read_text()) if inst_path.exists() else {"run_name": run_name, "pairs": {}}

    cv_ids = [rid for rid in records if rid.startswith("cv::")]
    jd_ids = [rid for rid in records if rid.startswith("jd::")]

    def _occ(rid: str) -> str:
        return rid.split("::")[2] if rid.count("::") >= 2 else ""

    all_pairs = [
        (cv, jd) for cv in cv_ids for jd in jd_ids
        if pairs_mode != "occ_match" or _occ(cv) == _occ(jd)
    ]
    todo = [(cv, jd) for cv, jd in all_pairs if f"{cv}|||{jd}" not in inst_data["pairs"]]
    print(f"INST: {len(todo)} to process, {len(all_pairs) - len(todo)} already done")

    if todo:
        inst_requests = []
        for cv_id, jd_id in todo:
            cv_nodes = records[cv_id]["nodes"]
            jd_nodes = records[jd_id]["nodes"]
            # Filter to the same node set used in build_inst_prompt so ENT indices align
            cv_nodes_prompt = [n for n in cv_nodes if n.get("source") not in ("inferred", "anchor")]
            jd_nodes_prompt = [n for n in jd_nodes if n.get("source") not in ("inferred", "anchor")]
            inst_requests.append({
                "custom_id": f"{cv_id}|||{jd_id}",
                "messages":  _to_messages(build_inst_prompt(cv_nodes_prompt, jd_nodes_prompt)),
                "_cv_nodes_prompt": cv_nodes_prompt,
                "_jd_nodes_prompt": jd_nodes_prompt,
            })

        # Stash node lists before submitting (they won't survive serialisation)
        node_lookup = {
            req["custom_id"]: (req.pop("_cv_nodes_prompt"), req.pop("_jd_nodes_prompt"))
            for req in inst_requests
        }

        inst_results = _run_batch(inst_requests)
        _save_step(data_root, run_name, "inst", inst_requests, inst_results)

        for cv_id, jd_id in todo:
            pair_key = f"{cv_id}|||{jd_id}"
            raw = inst_results.get(pair_key, "")
            cv_nodes_prompt, jd_nodes_prompt = node_lookup[pair_key]

            matches, cv_shadows, cv_isa_edges, jd_shadows, jd_isa_edges = \
                parse_inst_response(raw, cv_nodes_prompt, jd_nodes_prompt)
            print(f"  {pair_key[:60]}: {len(matches)} inst, "
                  f"{len(cv_shadows)} CV shadows, {len(jd_shadows)} JD shadows")

            def _merge(record_id, shadows, isa_edges):
                rec = records[record_id]
                existing_ids = {n["id"] for n in rec.get("nodes", [])}
                for sn in shadows:
                    if sn["id"] not in existing_ids:
                        rec.setdefault("nodes", []).append(sn)
                        existing_ids.add(sn["id"])
                existing_edges = {(e["source"], e["target"], e["relation"]) for e in rec.get("edges", [])}
                for ie in isa_edges:
                    key = (ie["source"], ie["target"], ie["relation"])
                    if key not in existing_edges:
                        rec.setdefault("edges", []).append(ie)
                        existing_edges.add(key)

            _merge(cv_id, cv_shadows, cv_isa_edges)
            _merge(jd_id, jd_shadows, jd_isa_edges)

            insts = {f"{m['cv_id']}|||{m['jd_id']}": 1.0 for m in matches}
            inst_data["pairs"][pair_key] = {
                "cv_id":         cv_id,
                "jd_id":         jd_id,
                "occ_code":      records[cv_id].get("occ_code", ""),
                "instantiations": insts,
            }

        inst_path.write_text(json.dumps(inst_data, ensure_ascii=False))
        # Save updated graph_data with merged shadows
        graph_path.write_text(json.dumps(graph_data, ensure_ascii=False))

    print(f"Instantiation done → {inst_path}")


# ── Phase 2b: Instantiate (string-match baseline) ────────────────────────────

# Valid (cv_type, jd_type) pairs for instantiation, derived from _CATEGORIES in analyse.py
_INST_TYPE_PAIRS: set[tuple[str, str]] = {
    ("Skill",        "SkillType"),
    ("Resource",     "ResourceType"),
    ("Knowledge",    "KnowledgeType"),
    ("Attitude",     "AttitudeType"),
    ("HumanTrait",   "AttitudeType"),
    ("HumanQuality", "AttitudeType"),
    ("HumanTask",    "TaskType"),
}

def _strmatch(a: str, b: str, threshold: float = 0.5) -> bool:
    import re as _re
    def _tok(s: str) -> set[str]:
        return set(_re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
    ta, tb = _tok(a), _tok(b)
    if not ta or not tb:
        return False
    sa, sb = " ".join(sorted(ta)), " ".join(sorted(tb))
    if sa in sb or sb in sa:
        return True
    return len(ta & tb) / len(ta | tb) >= threshold


def cmd_instantiate_strmatch(run_name: str, data_dir: str = "data",
                             pairs_mode: str = "occ_match",
                             threshold: float = 0.5):
    data_root  = Path(data_dir) / "condition_E"
    graph_path = data_root / "graph_data" / f"{run_name}.json"
    inst_path  = data_root / "inst_data"  / f"{run_name}.json"
    inst_path.parent.mkdir(parents=True, exist_ok=True)

    if not graph_path.exists():
        raise FileNotFoundError(f"graph_data not found — run --extract first: {graph_path}")

    graph_data = json.loads(graph_path.read_text())
    records    = graph_data["records"]
    inst_data  = json.loads(inst_path.read_text()) if inst_path.exists() else {"run_name": run_name, "pairs": {}}

    cv_ids = [rid for rid in records if rid.startswith("cv::")]
    jd_ids = [rid for rid in records if rid.startswith("jd::")]

    def _occ(rid: str) -> str:
        return rid.split("::")[2] if rid.count("::") >= 2 else ""

    all_pairs = [
        (cv, jd) for cv in cv_ids for jd in jd_ids
        if pairs_mode != "occ_match" or _occ(cv) == _occ(jd)
    ]
    todo = [(cv, jd) for cv, jd in all_pairs if f"{cv}|||{jd}" not in inst_data["pairs"]]
    print(f"INST (strmatch): {len(todo)} to process, {len(all_pairs) - len(todo)} already done")

    for cv_id, jd_id in todo:
        cv_nodes = [n for n in records[cv_id]["nodes"]
                    if n.get("source") not in ("inferred", "anchor")]
        jd_nodes = [n for n in records[jd_id]["nodes"]
                    if n.get("source") not in ("inferred", "anchor")]

        insts: dict[str, float] = {}
        for cn in cv_nodes:
            for jn in jd_nodes:
                if (cn.get("type"), jn.get("type")) in _INST_TYPE_PAIRS:
                    if _strmatch(cn["label"], jn["label"], threshold):
                        insts[f"{cn['id']}|||{jn['id']}"] = 1.0

        pair_key = f"{cv_id}|||{jd_id}"
        inst_data["pairs"][pair_key] = {
            "cv_id":          cv_id,
            "jd_id":          jd_id,
            "occ_code":       records[cv_id].get("occ_code", ""),
            "instantiations": insts,
        }

    inst_path.write_text(json.dumps(inst_data, ensure_ascii=False))
    print(f"Instantiation (strmatch) done → {inst_path}")


# ── Phase 3: Score ────────────────────────────────────────────────────────────

def cmd_score(run_name: str, data_dir: str = "data", discount: float = 0.5):
    sys.path.insert(0, str(_ROOT / "src" / "condition_E"))
    from scorer_e import score_pair_E

    data_root   = Path(data_dir) / "condition_E"
    graph_path  = data_root / "graph_data" / f"{run_name}.json"
    inst_path   = data_root / "inst_data"  / f"{run_name}.json"
    scores_path = data_root / "scores" / f"scores_{run_name}.jsonl"
    detail_path = data_root / "scores" / f"scores_{run_name}_detail.jsonl"
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    graph_data = json.loads(graph_path.read_text())
    inst_data  = json.loads(inst_path.read_text())
    records    = graph_data["records"]

    n = 0
    with scores_path.open("w") as f, detail_path.open("w") as fd:
        for pair in inst_data["pairs"].values():
            cv_id = pair["cv_id"]
            jd_id = pair["jd_id"]
            cv = records.get(cv_id)
            jd = records.get(jd_id)
            if not cv or not jd:
                continue

            instantiations = {
                tuple(k.split("|||")): v
                for k, v in pair["instantiations"].items()
            }
            score, detail = score_pair_E(cv, jd, instantiations, discount=discount)

            row = {
                "cv_id":          cv_id,
                "jd_id":          jd_id,
                "occ_code":       cv.get("occ_code"),
                "cv_style":       cv.get("style"),
                "jd_style":       jd.get("style"),
                "variant_id":     cv.get("variant_id", "base"),
                "counterfactual": cv.get("counterfactual"),
                "score":          score,
                "discount":       discount,
            }
            f.write(json.dumps(row) + "\n")
            fd.write(json.dumps({"cv_id": cv_id, "jd_id": jd_id, "detail": detail}) + "\n")
            n += 1

    print(f"Scored {n} pairs → {scores_path}")


# ── Metadata ─────────────────────────────────────────────────────────────────

def _write_metadata(run_name: str, args) -> None:
    from datetime import datetime, timezone
    meta_path = _ROOT / "data" / "metadata" / "runs.jsonl"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "condition": "E",
        "run_name":  run_name,
        "job_id":    "local_manual",
        "params": {
            "model":  _MODEL,
            "input":  args.input or "",
            "local":  True,
            "pairs":  args.pairs,
        },
    }
    with meta_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  [meta] written → {meta_path.relative_to(_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Condition E pipeline (batch mode)")
    parser.add_argument("--extract",     action="store_true")
    parser.add_argument("--instantiate", action="store_true")
    parser.add_argument("--strmatch",    action="store_true",
                        help="Use string-matching instead of LLM for --instantiate")
    parser.add_argument("--strmatch-threshold", type=float, default=0.5, dest="strmatch_threshold",
                        help="Jaccard threshold for string-match instantiation (default 0.5)")
    parser.add_argument("--score",       action="store_true")
    parser.add_argument("--run",         required=True)
    parser.add_argument("--input",       default=None, help="Dataset JSONL (required for --extract)")
    parser.add_argument("--data-dir",    default="data")
    parser.add_argument("--pairs",       default="occ_match", choices=["occ_match", "all"])
    parser.add_argument("--discount",    type=float, default=0.5,
                        help="Per-hop discount factor for graph propagation scoring (default 0.5)")
    parser.add_argument("--model",       default="claude-opus-4-6",
                        help="LLM model ID (e.g. gpt-4o-mini, claude-opus-4-6)")
    args = parser.parse_args()

    set_model(args.model)

    if args.extract:
        if not args.input:
            parser.error("--extract requires --input")
        cmd_extract(args.input, args.run, args.data_dir)

    if args.instantiate:
        if args.strmatch:
            cmd_instantiate_strmatch(args.run, args.data_dir, args.pairs, args.strmatch_threshold)
        else:
            cmd_instantiate(args.run, args.data_dir, args.pairs)

    if args.score:
        cmd_score(args.run, args.data_dir, args.discount)

    if args.extract:
        _write_metadata(args.run, args)


if __name__ == "__main__":
    main()
