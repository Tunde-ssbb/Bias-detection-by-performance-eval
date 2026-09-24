# scoring for condition E: graph-based propagation of JD node matches to nearby nodes

from __future__ import annotations
from collections import deque
from src.shared.scoring_weights import IMPORTANCE_WEIGHTS

DEFAULT_WEIGHT = IMPORTANCE_WEIGHTS["important"]

DEFAULT_DISCOUNT = 0.5


def _build_adjacency(edges: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """Undirected adjacency list: node_id → [(neighbour_id, relation)]."""
    adj: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        s, t, r = e["source"], e["target"], e.get("relation", "")
        adj.setdefault(s, []).append((t, r))
        adj.setdefault(t, []).append((s, r))
    return adj


def score_pair_E(
    cv_record: dict,
    jd_record: dict,
    instantiations: dict[tuple[str, str], float],
    discount: float = DEFAULT_DISCOUNT,
) -> tuple[float, dict]:

    #filter to nodes coming from extraction directly (not inferred)
    jd_nodes = [n for n in jd_record.get("nodes", []) if n.get("source") == "llm"]

    #default score
    if not jd_nodes:
        return 5.5, {}

    # get anchors
    cv_anchors = {n["id"] for n in cv_record.get("nodes", []) if n.get("source") == "anchor"}
    jd_anchors = {n["id"] for n in jd_record.get("nodes", []) if n.get("source") == "anchor"}

    #filter out anchor instantiations if they are there
    filtered_inst = {
        (cv_id, jd_id): conf
        for (cv_id, jd_id), conf in instantiations.items()
        if cv_id not in cv_anchors and jd_id not in jd_anchors
    }

    # Which jd nodes are directly matched?
    matched_jd: set[str] = {jd_id for (_, jd_id) in filtered_inst}

    # Build a combined undirected graph across both documents.
    cv_edges = cv_record.get("edges", [])
    jd_edges = jd_record.get("edges", [])

    def _prefixed(edges, prefix):
        out = []
        for e in edges:
            out.append({
                "source": prefix + e["source"],
                "target": prefix + e["target"],
                "relation": e.get("relation", ""),
            })
        return out

    combined_edges = _prefixed(cv_edges, "cv:") + _prefixed(jd_edges, "jd:")

    # Add instantiation edges (bidirectional), anchors already excluded
    for (cv_id, jd_id), _conf in filtered_inst.items():
        combined_edges.append({"source": "cv:" + cv_id, "target": "jd:" + jd_id, "relation": "instantiates"})

    adj = _build_adjacency(combined_edges)

    # BFS from every directly-matched JD node outward through the JD subgraph.
    # We want shortest distance from each JD node to any matched JD node,

    jd_dist: dict[str, int]         = {}  # jd_node_id: hop distance from nearest matched JD node
    jd_path: dict[str, list[str]]   = {}  # jd_node_id: path of jd node ids

    queue: deque[tuple[str, int, list[str]]] = deque()
    for jd_id in matched_jd:
        jd_dist[jd_id] = 0
        jd_path[jd_id] = [jd_id]
        queue.append((jd_id, 0, [jd_id]))

    # BFS over JD-intra edges only.
    # isA edges  are zero-cost: they express
    # type membership, not semantic distance, so traversing them does not
    # increment the hop counter.
    jd_adj = _build_adjacency(jd_edges)
    while queue:
        node, dist, path = queue.popleft()
        for neighbour, rel in jd_adj.get(node, []):
            if neighbour in jd_anchors:
                continue
            if neighbour not in jd_dist:
                next_dist = dist if rel == "isA" else dist + 1
                new_path = path + [neighbour]
                jd_dist[neighbour] = next_dist
                jd_path[neighbour] = new_path
                queue.append((neighbour, next_dist, new_path))

    # Score over each jd requirement
    total_w = 0.0
    raw = 0.0
    detail: dict[str, dict] = {}

    for node in jd_nodes:
        nid = node["id"]
        w = IMPORTANCE_WEIGHTS.get(node.get("importance"), DEFAULT_WEIGHT)
        total_w += w

        if nid in matched_jd:
            hop_distance = 1
            node_score = 1.0
            path = [nid]
        elif nid in jd_dist:
            # Reached via JD-graph propagation; the actual crossing to CV is
            # through the matched JD node at the end of the path.
            hop_distance = jd_dist[nid] + 1   # +1 for the instantiation hop
            node_score = discount ** (hop_distance - 1)
            path = jd_path[nid]
        else:
            hop_distance = None
            node_score = 0.0
            path = []

        raw += w * (2 * node_score - 1)

        detail[nid] = {
            "label":        node.get("label", nid),
            "importance":   node.get("importance", "important"),
            "weight":       w,
            "hop_distance": hop_distance,
            "path":         path,
            "node_score":   round(node_score, 4),
        }

    if total_w == 0:
        return 5.5, detail

    normalized = 1 + (raw + total_w) * 9 / (2 * total_w)
    score = round(max(1.0, min(10.0, normalized)), 3)
    return score, detail
