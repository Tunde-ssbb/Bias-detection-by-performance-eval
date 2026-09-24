
"""
Ontology utilities for condition E.

Three public functions:

    parse_ontology(ttl_path)  — parse competence_ontology.ttl into an OntologySchema
    dump_schema(out_path)     — parse the TTL and write data/ontology_schema.json
    load_schema()             — lazily load ontology_schema.json (used by the pipeline)

Run as a script to regenerate the schema file:
    python src/condition_E/ontology.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_ROOT   = Path(__file__).parent.parent.parent
_TTL    = Path(__file__).parent / "ontology" / "competence_ontology.ttl"
_SCHEMA = _ROOT / "data" / "ontology_schema.json"

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "src" / "shared"))


def is_jd_heur(local_name: str) -> bool:
    #Name-based fallback: True if local_name looks like a JD-side class.
    return local_name.endswith("Type") or local_name.endswith("Role")


CATEGORIZES_PROP = "http://purl.org/nemo/gufo#categorizes"

ANCHOR_NODES = {
    "cv": {
        "id":     "person_anchor",
        "type":   "Person",
        "label":  "the candidate",
        "source": "anchor",
    },
    "jd": {
        "id":       "role_anchor",
        "type":     "CapabilityRequiringRole",
        "label":    "the role",
        "required": True,
        "source":   "anchor",
    },
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ClassInfo:
    uri: str
    label: str
    comment: str
    side: str  # "cv", "jd", or "structural"


@dataclass
class RelationInfo:
    uri: str
    label: str
    comment: str
    domain: str
    range_: str
    extraction_type: str  # "intra_jd", "intra_cv", or "cross_side"


@dataclass
class OntologySchema:
    cv_classes: list[ClassInfo]
    jd_classes: list[ClassInfo]
    all_coreo_classes: dict[str, ClassInfo]
    class_parents: dict[str, str]
    instantiation_pairs: list[tuple[str, str]]
    intra_jd_relations: list[RelationInfo]
    intra_cv_relations: list[RelationInfo]
    cross_side_relations: list[RelationInfo]


"""
Parse ontology classes and relations
"""
def parse_ontology(ttl_path: str) -> OntologySchema:
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef

    COREO = Namespace("http://purl.org/coreo#")

    g = Graph()
    g.parse(ttl_path, format="turtle")

    # get non leaf nodes
    all_classes = set(g.subjects(RDF.type, OWL.Class))
    has_subclass = set(
        o for _, _, o in g.triples((None, RDFS.subClassOf, None))
        if o in all_classes
    )
    # filter on comment
    candidate_uris = {
        str(cls) for cls in all_classes
        if cls not in has_subclass
        and str(cls).startswith(str(COREO))
        and g.value(cls, RDFS.comment) is not None
    }

    jd_uris: set[str] = set()
    cv_uris: set[str] = set()
    instantiation_pairs: list[tuple[str, str]] = []

    # get inst pairs
    categorizes_jd: set[str] = set()
    if CATEGORIZES_PROP:
        cat_prop = URIRef(CATEGORIZES_PROP)
        for s, _, o in g.triples((None, cat_prop, None)):
            jd_label = _get_label(g, s)
            cv_label = _get_label(g, o)
            if jd_label and cv_label:
                categorizes_jd.add(str(s))
                instantiation_pairs.append((jd_label, cv_label))

    def is_jd_side(uri: str) -> bool:
        return uri in categorizes_jd or is_jd_heur(uri.split("#")[-1])

    for uri in candidate_uris:
        if is_jd_side(uri):
            jd_uris.add(uri)
        else:
            cv_uris.add(uri)

    def _make_class_info(uri: str, side: str) -> ClassInfo:
        ref = URIRef(uri)
        return ClassInfo(
            uri=uri,
            label=_get_label(g, ref) or uri.split("#")[-1],
            comment=_get_comment(g, ref) or "",
            side=side,
        )

    cv_classes = [_make_class_info(uri, "cv") for uri in cv_uris]
    jd_classes = [_make_class_info(uri, "jd") for uri in jd_uris]

    # split classes on sides
    all_coreo_classes: dict[str, ClassInfo] = {}
    for cls in all_classes:
        uri = str(cls)
        if uri.startswith(str(COREO)):
            side = "jd" if is_jd_side(uri) else "cv"
            all_coreo_classes[uri] = _make_class_info(uri, side)

    # build superclass map
    class_parents: dict[str, str] = {}
    for child, _, parent in g.triples((None, RDFS.subClassOf, None)):
        child_uri, parent_uri = str(child), str(parent)
        if child_uri.startswith(str(COREO)) and parent_uri.startswith(str(COREO)):
            class_parents[child_uri] = parent_uri

    def _is_cv(uri: str) -> bool:
        return uri in cv_uris or (uri.startswith(str(COREO)) and not is_jd_side(uri))

    intra_jd: list[RelationInfo] = []
    intra_cv: list[RelationInfo] = []
    cross:    list[RelationInfo] = []

    # get relations
    for prop in g.subjects(RDF.type, OWL.ObjectProperty):
        domain = g.value(prop, RDFS.domain)
        range_ = g.value(prop, RDFS.range)
        if domain is None or range_ is None:
            continue
        label      = _get_label(g, prop) or str(prop).split("#")[-1]
        comment    = _get_comment(g, prop) or ""
        domain_str = str(domain)
        range_str  = str(range_)
        domain_jd  = is_jd_side(domain_str)
        domain_cv  = _is_cv(domain_str)
        range_jd   = is_jd_side(range_str)
        range_cv   = _is_cv(range_str)

        if domain_jd and range_jd:
            etype, bucket = "intra_jd", intra_jd
        elif domain_cv and range_cv:
            etype, bucket = "intra_cv", intra_cv
        elif (domain_jd and range_cv) or (domain_cv and range_jd):
            etype, bucket = "cross_side", cross
        else:
            continue

        bucket.append(RelationInfo(
            uri=str(prop), label=label, comment=comment,
            domain=domain_str.split("#")[-1], range_=range_str.split("#")[-1],
            extraction_type=etype,
        ))

    return OntologySchema(
        cv_classes=cv_classes, jd_classes=jd_classes,
        all_coreo_classes=all_coreo_classes, class_parents=class_parents,
        instantiation_pairs=instantiation_pairs,
        intra_jd_relations=intra_jd, intra_cv_relations=intra_cv,
        cross_side_relations=cross,
    )


def _get_label(g, ref) -> Optional[str]:
    from rdflib import RDFS
    label = g.value(ref, RDFS.label)
    return str(label) if label else None


def _get_comment(g, ref) -> Optional[str]:
    from rdflib import RDFS
    comment = g.value(ref, RDFS.comment)
    if comment:
        return str(comment).split("\n")[0].strip()[:300]
    return None


"""
Function to dump parsed ontology schema to readable file
"""
def dump_schema(out_path: "str | Path" = _SCHEMA, ttl_path: "str | Path" = _TTL):
    schema = parse_ontology(str(ttl_path))

    # JD side classes have no good descriptions so add them manually.
    _JD_ENRICHED: dict[str, str] = {
        "SkillType":     f"A job requirement specifying a needed Skill. {next((c.comment for c in schema.cv_classes if c.label == 'Skill'), '')}",
        "KnowledgeType": f"A job requirement specifying a needed Knowledge area. {next((c.comment for c in schema.cv_classes if c.label == 'Knowledge'), '')}",
        "AttitudeType":  f"A job requirement specifying a needed Attitude. {next((c.comment for c in schema.cv_classes if c.label == 'Attitude'), '')}",
        "ResourceType":  f"A job requirement specifying a needed Resource. {next((c.comment for c in schema.cv_classes if c.label == 'Resource'), '')}",
        "TaskType":      f"A job requirement specifying a needed Task. {next((c.comment for c in schema.cv_classes if c.label == 'HumanTask'), '')}",
        "ArtifactType":  (
            "A job requirement specifying a needed Artifact (tool, document, or deliverable). "
            f"{next((c.comment for c in schema.cv_classes if c.label == 'TaskInput'), '')} / "
            f"{next((c.comment for c in schema.cv_classes if c.label == 'TaskOutput'), '')}"
        ),
    }

    def _cls(c):
        return {"label": c.label, "uri": c.uri, "comment": c.comment}

    def _cls_jd(c):
        return {"label": c.label, "uri": c.uri, "comment": _JD_ENRICHED.get(c.label, c.comment)}

    def _rel(r):
        return {"label": r.label, "uri": r.uri, "comment": r.comment,
                "domain": r.domain, "range": r.range_, "type": r.extraction_type}

    uri_to_label = {c.uri: c.label for c in schema.all_coreo_classes.values()}
    superclass_map: dict[str, list[str]] = {}
    for cls in list(schema.cv_classes) + list(schema.jd_classes):
        ancestors = []
        current, seen = cls.uri, set()
        while True:
            parent_uri = schema.class_parents.get(current)
            if parent_uri is None or parent_uri in seen:
                break
            seen.add(parent_uri)
            parent_label = uri_to_label.get(parent_uri)
            if parent_label:
                ancestors.append(parent_label)
            current = parent_uri
        if ancestors:
            superclass_map[cls.label] = ancestors

    out = {
        "source_ttl": str(ttl_path),
        "cv_node_types": sorted([_cls(c)    for c in schema.cv_classes], key=lambda x: x["label"]),
        "jd_node_types": sorted([_cls_jd(c) for c in schema.jd_classes], key=lambda x: x["label"]),
        "instantiation_pairs": [{"jd_type": jd, "cv_type": cv} for jd, cv in schema.instantiation_pairs],
        "relations": {
            "intra_cv":   sorted([_rel(r) for r in schema.intra_cv_relations],   key=lambda x: x["label"]),
            "intra_jd":   sorted([_rel(r) for r in schema.intra_jd_relations],   key=lambda x: x["label"]),
            "cross_side": sorted([_rel(r) for r in schema.cross_side_relations], key=lambda x: x["label"]),
        },
        "superclass_map": dict(sorted(superclass_map.items())),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    n_cv  = len(out["cv_node_types"])
    n_jd  = len(out["jd_node_types"])
    n_rel = sum(len(v) for v in out["relations"].values())
    print(f"Schema written → {out_path}")
    print(f"  CV node types : {n_cv}")
    print(f"  JD node types : {n_jd}")
    print(f"  Relations     : {n_rel}  "
          f"(intra_cv={len(out['relations']['intra_cv'])}, "
          f"intra_jd={len(out['relations']['intra_jd'])}, "
          f"cross_side={len(out['relations']['cross_side'])})")



_schema_cache: "dict | None" = None

# load schema to use in pipeline
def load_schema() -> dict:
    global _schema_cache
    if _schema_cache is None:
        if not _SCHEMA.exists():
            raise FileNotFoundError(
                f"Ontology schema not found: {_SCHEMA}\n"
                f"Run:  python src/condition_E/ontology.py"
            )
        _schema_cache = json.loads(_SCHEMA.read_text())
    return _schema_cache


"""
CLI
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse competence_ontology.ttl and write ontology_schema.json")
    parser.add_argument("--ttl", default=str(_TTL),    help="Path to .ttl file")
    parser.add_argument("--out", default=str(_SCHEMA), help="Output path for schema JSON")
    args = parser.parse_args()
    dump_schema(out_path=args.out, ttl_path=args.ttl)
