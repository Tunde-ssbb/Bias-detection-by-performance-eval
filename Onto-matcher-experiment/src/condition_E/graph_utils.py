
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from condition_E.ontology import ANCHOR_NODES

# add anchor node into nodelist
def inject_anchor_nodes(nodes: list[dict], side: str) -> list[dict]:

    anchor = ANCHOR_NODES.get(side)
    if anchor is None:
        return nodes
    return [anchor] + [n for n in nodes if n.get("id") != anchor["id"]]
