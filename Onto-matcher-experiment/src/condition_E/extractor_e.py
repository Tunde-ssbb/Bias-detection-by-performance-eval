
#condition E prompt builders 


import json
from pathlib import Path

from jinja2 import Environment

from condition_E.ontology import load_schema  # noqa: E402


# for cv and jd side return labels and definitions from the schema, excluding anchor node types
def get_label_definitions(side: str) -> dict[str, str]:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
    from condition_E.ontology import ANCHOR_NODES
    anchor_type = ANCHOR_NODES[side]["type"]
    schema = load_schema()
    key = "cv_node_types" if side == "cv" else "jd_node_types"
    return {c["label"]: c["comment"] for c in schema[key] if c["label"] != anchor_type}


def get_labels(side: str) -> list[str]:
    return list(get_label_definitions(side).keys())


# ── NER: single-stage span detection + typing — spaCy-llm NER.v3 template ─────

# The spacy-llm NER.v3 Jinja2 template (source: spacy_llm.tasks.ner).

_SPACY_NER_V3_TEMPLATE = (
    "You are an expert Named Entity Recognition (NER) system.\n"
    "Your task is to accept Text as input and extract named entities.\n"
    "Entities must have one of the following labels: {{ ', '.join(labels) }}.\n"
    "If a span is not an entity label it: `==NONE==`.\n"
    "{# whitespace #}\n"
    "{# whitespace #}\n"
    "{%- if description -%}\n"
    "{# whitespace #}\n"
    "{{ description }}\n"
    "{# whitespace #}\n"
    "{%- endif -%}\n"
    "{%- if label_definitions -%}\n"
    "Below are definitions of each label to help aid you in what kinds of named entities to extract for each label.\n"
    "Assume these definitions are written by an expert and follow them closely.\n"
    "{# whitespace #}\n"
    "{%- for label, definition in label_definitions.items() -%}\n"
    "{{ label }}: {{ definition }}\n"
    "{# whitespace #}\n"
    "{%- endfor -%}\n"
    "{# whitespace #}\n"
    "{# whitespace #}\n"
    "{%- endif -%}\n"
    "{%- if prompt_examples -%}\n"
    "Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:\n"
    "{# whitespace #}\n"
    "{# whitespace #}\n"
    "{%- for example in prompt_examples -%}\n"
    "Paragraph: {{ example.text }}\n"
    "Answer:\n"
    "{# whitespace #}\n"
    "{%- for span in example.spans -%}\n"
    "{{ loop.index }}. {{ span.to_str() }}\n"
    "{# whitespace #}\n"
    "{%- endfor -%}\n"
    "{# whitespace #}\n"
    "{# whitespace #}\n"
    "{%- endfor -%}\n"
    "{%- else -%}\n"
    "{# whitespace #}\n"
    "Here is an example of the output format for a paragraph using different labels than this task requires.\n"
    "Only use this output format but use the labels provided\n"
    "above instead of the ones defined in the example below.\n"
    "Do not output anything besides entities in this output format.\n"
    "Output entities in the order they occur in the input paragraph regardless of label.\n"
    "\n"
    "Q: Given the paragraph below, identify a list of entities, and for each entry explain why it is or is not an entity:\n"
    "\n"
    "Paragraph: Sriracha sauce goes really well with hoisin stir fry, but you should add it after you use the wok.\n"
    "Answer:\n"
    "1. Sriracha sauce | True | INGREDIENT | is an ingredient to add to a stir fry\n"
    "2. really well | False | ==NONE== | is a description of how well sriracha sauce goes with hoisin stir fry\n"
    "3. hoisin stir fry | True | DISH | is a dish with stir fry vegetables and hoisin sauce\n"
    "4. wok | True | EQUIPMENT | is a piece of cooking equipment used to stir fry ingredients\n"
    "{# whitespace #}\n"
    "{# whitespace #}\n"
    "{%- endif -%}\n"
    "Paragraph: {{ text }}\n"
    "Answer:\n"
)

_jinja_env = Environment(keep_trailing_newline=True)

# spaCy uses {# whitespace #} as a blank-line that Jinja treats as a
# comment (empty output). Replace with a newline.
_NER_TEMPLATE_RENDERED = _SPACY_NER_V3_TEMPLATE.replace("{# whitespace #}", "{{ '\\n' }}")


#fill NER template. with labels and definitions
def _render_ner_template(text: str, side: str | None, description: str = "",
                          override_labels: list[str] | None = None) -> str:
    if override_labels is not None:
        labels = override_labels
        label_defs = None
    else:
        label_defs = get_label_definitions(side)
        labels = list(label_defs.keys())
    template = _jinja_env.from_string(_NER_TEMPLATE_RENDERED)
    return template.render(
        labels=labels,
        label_definitions=label_defs,
        description=description,
        prompt_examples=None,
        text=text,
    )


def build_ner_prompt(side: str, text: str) -> str:
    """
    NER prompt building.
    Uses the spacy llm  NER task  template on the raw document text.
    """
    return _render_ner_template(text, side)


# ── Response parser — uses spaCy's SpanReason.from_str ───────────────────────

import re as _re

from spacy_llm.tasks.span.examples import SpanReason

_IMP_RE = _re.compile(  # used by parse_importance_response
    r"importance=([a-z ]+?)(?:\s*$|\s+[a-z]+=)",
    _re.IGNORECASE,
)

# Short ID prefix per class label (mirrors extractor_d._ID_PREFIX)
_ID_PREFIX: dict[str, str] = {
    "Skill":              "skill",
    "Knowledge":          "know",
    "HumanTrait":         "trait",
    "HumanQuality":       "qual",
    "Attitude":           "att",
    "PersonalCompetence": "comp",
    "Resource":           "res",
    "TaskInput":          "tinp",
    "TaskOutput":         "tout",
    "HumanTask":          "htask",
    "Field":              "field",
    "Proficiency":        "prof",
    "Evidence":           "evid",
    "SkillType":               "stype",
    "KnowledgeType":           "ktype",
    "AttitudeType":            "atype",
    "TaskType":                "ttype",
    "ArtifactType":            "art",
    "ResourceType":            "rtype",
    "Person":                  "person",
    "CapabilityRequiringRole": "role",
}


def get_id_prefix(class_label: str) -> str:
    return _ID_PREFIX.get(class_label, class_label[:5].lower())


def _find_sentence(text: str, span: str) -> str:
    """Return the first sentence in text that contains span (case-insensitive)."""
    if not span or not text:
        return ""
    span_lower = span.lower()
    for sent in _re.split(r"(?<=[.!?])\s+|\n", text):
        if span_lower in sent.lower():
            return sent.strip()
    return ""

#parse responses from NER step
def parse_ner_response(raw: str, side: str, original_text: str = "") -> list[dict]:
    """
    Parse NER output (Core-O type labels) using SpanReason.from_str.
    """
    valid_labels = set(get_labels(side))
    type_counters: dict[str, int] = {}
    nodes = []

    for line in raw.splitlines():
        try:
            sr = SpanReason.from_str(line)
        except (ValueError, IndexError):
            continue
        if not sr.is_entity or sr.label == "==NONE==" or sr.label not in valid_labels:
            continue

        # collect each parsable node
        source_sentence = _find_sentence(original_text, sr.text)
        prefix = get_id_prefix(sr.label)
        type_counters[prefix] = type_counters.get(prefix, 0) + 1

        node = {
            "id":              f"{prefix}_{type_counters[prefix]}",
            "type":            sr.label,
            "label":           sr.text,
            "evidence":        sr.reason,
            "source_sentence": source_sentence,
            "source":          "llm",
        }

        nodes.append(node)

    return nodes


# ── Stage 3: Importance qualification (JD only) — spaCy NER.v3 template ──────

def build_importance_prompt(nodes: list[dict], original_text: str) -> str:
    """
    Stage 3 (JD only): assign an importance label to each extracted JD node.
    Uses the NER.v3 template with the 5 importance levels as labels.
    The 'paragraph' is a numbered list of node labels with their source sentence.
    """
    from shared.scoring_weights import IMPORTANCE_LEVELS, IMPORTANCE_CUES

    cue_lines = "\n".join(f'  "{level}": {IMPORTANCE_CUES[level]}' for level in IMPORTANCE_LEVELS)
    # add custom label description
    description = (
        "Classify each job requirement by how important it is to the role, "
        "based on the language used in the job description.\n\n"
        "Label definitions (assign based on language cues):\n"
        f"{cue_lines}"
    )

    # collect all nodes for paragraph
    span_paragraph = "\n".join(
        f'{i+1}. {n["label"]} — context: "{n.get("source_sentence") or n.get("evidence", "")}"'
        for i, n in enumerate(nodes)
    )

    #render template
    return _render_ner_template(
        span_paragraph,
        side=None,
        description=description,
        override_labels=IMPORTANCE_LEVELS,
    )


def parse_importance_response(raw: str, nodes: list[dict]) -> list[dict]:
    """
    Parse importance output and add to existing node records.
    Returns the same node list with 'importance' set where parseable.
    """
    from shared.scoring_weights import IMPORTANCE_LEVELS

    valid_labels = set(IMPORTANCE_LEVELS)
    node_by_idx  = {i + 1: n for i, n in enumerate(nodes)}

    for line in raw.splitlines():
        idx_match = _re.match(r"^(\d+)\.", line.strip())
        try:
            sr = SpanReason.from_str(line)
        except (ValueError, IndexError):
            continue
        if not sr.is_entity or sr.label not in valid_labels:
            continue
        if idx_match:
            node = node_by_idx.get(int(idx_match.group(1)))
            if node is not None:
                node["importance"] = sr.label

    return nodes


# ── Relation extraction — spaCy rel.v1 template ───────────────────────────────

# spaCy rel.v1 template: {# whitespace #} replaced with {{ '\n' }} so Jinja renders blank lines.
_SPACY_REL_V1_TEMPLATE = """\
The text below contains pre-extracted entities, denoted in the following format within the text:
{{ '\n' }}
<entity text>[ENT<entity id>:<entity label>]
{{ '\n' }}
From the text below, extract the following relations between entities:
{{ '\n' }}
{{ '\n' }}
{%- for label in labels -%}
{{ label }}
{{ '\n' }}
{%- endfor -%}
{{ '\n' }}
The extraction has to use the following format, with one line for each detected relation:
{{ '\n' }}
{"dep": <entity id>, "dest": <entity id>, "relation": <relation label>}
{{ '\n' }}
Make sure that only relevant relations are listed, and that each line is a valid JSON object.
{{ '\n' }}
{%- if label_definitions -%}
Below are definitions of each label to help aid you in what kinds of relationship to extract for each label.
Assume these definitions are written by an expert and follow them closely.
{{ '\n' }}
{{ '\n' }}
{%- for label, definition in label_definitions.items() -%}
{{ label }}: {{ definition }}
{{ '\n' }}
{%- endfor -%}
{{ '\n' }}
{{ '\n' }}
{%- endif -%}
Here is the text that needs labeling:
{{ '\n' }}
Text:
\'\'\'
{{ text }}
\'\'\'"""


# get sided relation definition from schema
def get_relation_definitions(side: str) -> dict[str, str]:
    schema = load_schema()
    key = "intra_cv" if side == "cv" else "intra_jd"
    return {
        r["label"]: r["comment"]
        for r in schema["relations"][key]
        if r.get("comment")
    }


def _get_superclass_map() -> dict[str, list[str]]:
    return load_schema().get("superclass_map", {})


#get superclass chain for a type
def _type_chain(type_: str) -> list[str]:
    return [type_] + _get_superclass_map().get(type_, [])


# get list of nodes as text fro relation prompt
def _annotate_nodes_as_text(nodes: list[dict], with_superclasses: bool = False) -> str:
    """Fallback: bare list of annotated node labels (no source text available)."""
    lines = []

    for i, node in enumerate(nodes):
        chain = "/".join(_type_chain(node["type"])) if with_superclasses else node["type"]
        lines.append(f"{node['label']}[ENT{i}:{chain}]")
    return "\n".join(lines)


def _build_inline_annotated_text(nodes: list[dict], original_text: str) -> str:
    """
    Reconstruct the original document with entity annotations injected inline.

    Each node whose label appears in its source_sentence is replaced with
    ``label[ENTi:Type/Super1/Super2]`` in that sentence. Sentences with no
    entities are included as-is so the LLM retains full context. Anchor and
    inferred nodes (no source_sentence) are appended as a labelled footer so
    the LLM still knows their ENT index.
    """
    # Index every node
    tag: dict[int, str] = {}  # node index → annotation tag
    for i, node in enumerate(nodes):
        chain = "/".join(_type_chain(node["type"]))
        tag[i] = f"[ENT{i}:{chain}]"

    # Split into sentences preserving original order
    sentences = _re.split(r"(?<=[.!?])\s+|\n", original_text)

    # Map sentence → [(node_index, label)] sorted by descending label length
    # (longest first so overlapping spans don't corrupt shorter ones)
    from collections import defaultdict as _dd
    sent_nodes: dict[str, list[tuple[int, str]]] = _dd(list)
    no_sentence: list[tuple[int, dict]] = []  # anchor / inferred nodes

    for i, node in enumerate(nodes):
        src = node.get("source_sentence", "").strip()
        if not src:
            no_sentence.append((i, node))
            continue
        # Match to the closest sentence
        for sent in sentences:
            if src.lower() == sent.strip().lower() or src.lower() in sent.strip().lower():
                sent_nodes[sent.strip()].append((i, node["label"]))
                break
        else:
            # source_sentence not found verbatim — fall back to label search
            label_lower = node["label"].lower()
            for sent in sentences:
                if label_lower in sent.lower():
                    sent_nodes[sent.strip()].append((i, node["label"]))
                    break
            else:
                no_sentence.append((i, node))

    # Build annotated sentences
    annotated: list[str] = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        entries = sent_nodes.get(sent, [])
        # Sort by descending label length so longer spans are replaced first
        entries_sorted = sorted(entries, key=lambda x: len(x[1]), reverse=True)
        result = sent
        for i, label in entries_sorted:
            # Case-insensitive replacement of first occurrence
            pat = _re.compile(_re.escape(label), _re.IGNORECASE)
            result = pat.sub(lambda m, _i=i: f"{m.group()}{tag[_i]}", result, count=1)
        annotated.append(result)

    body = "\n".join(annotated)

    # Append footer for nodes with no source sentence (e.g. anchors)
    if no_sentence:
        footer_lines = ["", "# Additional entities (no source sentence):"]
        for i, node in no_sentence:
            chain = "/".join(_type_chain(node["type"]))
            footer_lines.append(f"{node['label']}[ENT{i}:{chain}]")
        body += "\n".join(footer_lines)

    return body


def build_rel_prompt(side: str, nodes: list[dict], original_text: str = "") -> str:
    """
    Build a relation-extraction prompt using the spaCy rel.v1 template.
    When original_text is provided, entities are annotated inline within the
    source sentences; otherwise falls back to a bare node list.
    """
    label_defs = get_relation_definitions(side)
    if original_text:
        text = _build_inline_annotated_text(nodes, original_text)
    else:
        text = _annotate_nodes_as_text(nodes, with_superclasses=True)
    env = Environment()
    #render template
    tmpl = env.from_string(_SPACY_REL_V1_TEMPLATE)
    return tmpl.render(
        labels=list(label_defs.keys()),
        label_definitions=label_defs,
        text=text,
    )


def _shadow_id(concrete_id: str, target_type: str) -> str:
    return f"{concrete_id}__{target_type}"


#func to create the shadow nodes
def _get_or_create_shadow(
    concrete_node: dict,
    target_type: str,
    registry: dict,
) -> tuple[str, dict | None, dict | None]:
    """
    If the shadow already exists in registry, returns its id with no new objects.
    """
    key = (concrete_node["id"], target_type)
    if key in registry:
        return registry[key], None, None
    sid = _shadow_id(concrete_node["id"], target_type)
    shadow = {
        "id":     sid,
        "type":   target_type,
        "label":  concrete_node["label"],
        "source": "inferred",
    }
    isa_edge = {"source": concrete_node["id"], "target": sid, "relation": "isA"}
    registry[key] = sid
    return sid, shadow, isa_edge


def _build_shadow_registry(nodes: list[dict]) -> dict:
    """Seed the registry from any inferred nodes already present in the list."""
    registry = {}
    for node in nodes:
        if node.get("source") == "inferred":
            parts = node["id"].split("__", 1)
            if len(parts) == 2:
                registry[(parts[0], node["type"])] = node["id"]
    return registry


def _resolve_endpoint(
    node: dict,
    required_type: str,
    registry: dict,
    shadow_nodes: list,
    isa_edges: list,
) -> str:
    
    #return the node id to use as a relation endpoint.
    #if the node's concrete type is a proper subclass of required_type, a shadow
    #node at required_type level is created (lazily) and its id is returned.
    #f the node type already matches or required_type is unknown, returns the
    #concrete node id unchanged.

    if not required_type or node["type"] == required_type:
        return node["id"]
    #create shadow and isa edge if needed
    if required_type in _type_chain(node["type"])[1:]:
        sid, shadow, isa = _get_or_create_shadow(node, required_type, registry)
        if shadow:
            shadow_nodes.append(shadow)
            isa_edges.append(isa)
        return sid
    return node["id"]


def _rel_domain_range() -> dict[str, tuple[str, str]]:
    """Return {relation_label: (domain, range)} from all schema relations."""
    schema = load_schema()
    result = {}
    for bucket in schema["relations"].values():
        for r in bucket:
            result[r["label"]] = (r.get("domain", ""), r.get("range", ""))
    return result


def parse_rel_response(
    raw: str,
    nodes: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Parse the rel.v1 response into edge dicts.

    When a relation's domain/range is a superclass of the endpoint node's
    concrete type, a shadow node at that superclass level is lazily created
    and an isA edge is added linking the concrete node to its shadow.

    returns
    edges        : [{source, target, relation}]  (may reference shadow node ids)
    shadow_nodes : new inferred nodes to merge into the record's node list
    isa_edges    : isA edges to merge into the record's edge list
    """
    valid_labels = set(get_relation_definitions("cv")) | set(get_relation_definitions("jd"))
    dr_map = _rel_domain_range()
    n = len(nodes)
    edges: list[dict] = []
    shadow_nodes: list[dict] = []
    isa_edges: list[dict] = []
    seen: set = set()
    registry = _build_shadow_registry(nodes) #get shadow nodes already present

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        dep  = _parse_ent_id(obj.get("dep"))
        dest = _parse_ent_id(obj.get("dest"))
        rel  = obj.get("relation", "")
        if dep is None or dest is None:
            continue
        if dep < 0 or dep >= n or dest < 0 or dest >= n:
            continue
        if rel not in valid_labels:
            continue
        domain, range_ = dr_map.get(rel, ("", ""))
        #lazily create shadow nodes if needed
        src_id = _resolve_endpoint(nodes[dep],  domain, registry, shadow_nodes, isa_edges)
        tgt_id = _resolve_endpoint(nodes[dest], range_, registry, shadow_nodes, isa_edges)
        key = (src_id, tgt_id, rel)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": src_id, "target": tgt_id, "relation": rel})

    return edges, shadow_nodes, isa_edges


# ── Instantiation — spaCy rel.v1 template (cross-document) ───────────────────

# fuzzy matching for ent id numbers so paring is more flexible
def _parse_ent_id(val) -> int | None:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        nums = _re.findall(r"\d+", val)
        if len(nums) == 1:
            return int(nums[0])
    return None


def get_instantiation_pairs() -> list[dict]:
    return load_schema()["instantiation_pairs"]


_ANCHOR_INST_PAIR = "CapabilityRequiringRole"  # excluded from inst prompt and parsing

# is label definition builder for v2 prompts, add specific examples to make more clear
def _inst_label_definitions() -> dict[str, str]:
    """
    Template-generated definitions for each instantiation relation label.
    Relation labels use an _inst suffix to avoid confusion with type names.
    The anchor pair (Person → CapabilityRequiringRole) is excluded.
    """
    _examples = {
        "Artifact":         ("technical report",           "technical reports"),
        "Resource":         ("Microsoft Excel",            "spreadsheet software"),
        "Skill":            ("software development",       "software development competence type"),
        "HumanTask":        ("directed engineering teams", "engineering management"),
        "HumanAspect":      ("software development competence", "software development competence type"),
        "HumanCapability":  ("software development competence", "software development competence type"),
        "PersonalCompetence": ("leadership under pressure", "leadership competence type"),
    }
    return {
        f"{p['jd_type']}_inst": (
            f"A specific {p['cv_type']} from the candidate that is a concrete instance "
            f"of this exact {p['jd_type']} requirement. "
            f"For example, John's \"{_examples.get(p['cv_type'], ('X', 'Y'))[0]}\" ({p['cv_type']}) "
            f"instantiates the \"{_examples.get(p['cv_type'], ('X', 'Y'))[1]}\" ({p['jd_type']}). "
            f"Do not match merely because the type matches."
        )
        for p in get_instantiation_pairs()
        if p["jd_type"] != _ANCHOR_INST_PAIR
    }


_INST_TEMPLATE = """\
The entities below are pre-extracted from two documents — a candidate CV and a job description — \
denoted in the following format:

<entity text>[ENT<entity id>:<entity label>]

From the entities below, extract the following relations between entities:

{%- for label in labels %}
{{ label }}
{%- endfor %}

The extraction has to use the following format, with one line for each detected relation:

{"dep": <entity id>, "dest": <entity id>, "relation": <relation label>}

Make sure that only relevant relations are listed, and that each line is a valid JSON object.
{%- if label_definitions %}

Below are definitions of each label to help aid you in what kinds of relationship to extract for each label.
Assume these definitions are written by an expert and follow them closely.

{%- for label, definition in label_definitions.items() %}
{{ label }}: {{ definition }}
{%- endfor %}

{%- endif %}

Here are the entities that need labeling:

Entities:
\'\'\'
{{ text }}
\'\'\'"""


def _format_nodes_section(nodes: list[dict], offset: int, header: str) -> str:
    """Format a list of nodes as a named section with source sentences."""
    lines = [header]
    for i, node in enumerate(nodes):
        chain = "/".join(_type_chain(node["type"]))
        src   = node.get("source_sentence", "").strip()
        tag   = f'ENT{offset + i} [{chain}]: "{node["label"]}"'
        lines.append(f"  {tag}")
        if src:
            lines.append(f'    Context: "{src}"')
    return "\n".join(lines)


def build_inst_prompt(cv_nodes: list[dict], jd_nodes: list[dict]) -> str:
    """
    Build an instantiation prompt with separate CV and JD sections, each node
    shown with its source sentence for context.
    """
    # exclude inferred and anchor nodes
    cv_concrete = [n for n in cv_nodes if n.get("source") not in ("inferred", "anchor")]
    jd_concrete = [n for n in jd_nodes if n.get("source") not in ("inferred", "anchor")]
    n_cv = len(cv_concrete)
    n_jd = len(jd_concrete)


    label_defs     = _inst_label_definitions()
    valid_jd_types = {p["jd_type"] for p in get_instantiation_pairs()
                      if p["jd_type"] != _ANCHOR_INST_PAIR}
    present_jd_types = {n["type"] for n in jd_concrete} & valid_jd_types
    filtered_defs    = {k: v for k, v in label_defs.items()
                        if k.removesuffix("_inst") in present_jd_types}

    #seperate sections for readability in new_inst prompts
    cv_section = _format_nodes_section(cv_concrete, 0,    "=== CANDIDATE (CV) — ENTITIES ===")
    jd_section = _format_nodes_section(jd_concrete, n_cv, "=== JOB DESCRIPTION (JD) — ENTITIES ===")
    text = f"{cv_section}\n\n{jd_section}"

    env  = Environment()
    tmpl = env.from_string(_INST_TEMPLATE)
    return tmpl.render(
        labels=list(filtered_defs.keys()),
        label_definitions=filtered_defs,
        text=text,
    )


def _find_matching_pair(
    cv_type: str,
    jd_type: str,
    pairs: list[dict],
) -> dict | None:
    """
    Find the first instantiation pair that matches (cv_type, jd_type) via
    superclass walkup on both sides.  Returns the pair dict or None.
    """
    cv_chain = set(_type_chain(cv_type))
    jd_chain = set(_type_chain(jd_type))
    for pair in pairs:
        if pair["cv_type"] in cv_chain and pair["jd_type"] in jd_chain:
            return pair
    return None


def parse_inst_response(
    raw: str,
    cv_nodes: list[dict],
    jd_nodes: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """
    Parse instantiation rel.v1 response into match dicts.
    dep must be a CV node index, dest must be a JD node index.

    Superclass walkup is applied symmetrically: if a CV node's concrete type
    is a subclass of the pair's cv_type, a shadow node is created at cv_type
    level; same for the JD side.

    Returns
    -------
    matches       : [{cv_id, jd_id}]  (ids may reference shadow nodes)
    cv_shadows    : new inferred nodes for the CV record
    cv_isa_edges  : isA edges for the CV record
    jd_shadows    : new inferred nodes for the JD record
    jd_isa_edges  : isA edges for the JD record
    """
    pairs  = get_instantiation_pairs()
    n_cv   = len(cv_nodes)
    n_all  = n_cv + len(jd_nodes)
    seen   = set()
    matches: list[dict] = []
    cv_shadows: list[dict] = []
    cv_isa_edges: list[dict] = []
    jd_shadows: list[dict] = []
    jd_isa_edges: list[dict] = []
    cv_registry = _build_shadow_registry(cv_nodes)
    jd_registry = _build_shadow_registry(jd_nodes)

    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        dep  = _parse_ent_id(obj.get("dep"))
        dest = _parse_ent_id(obj.get("dest"))
        if dep is None or dest is None:
            continue
        if dep < 0 or dep >= n_all or dest < 0 or dest >= n_all:
            continue
        if dep >= n_cv or dest < n_cv:
            continue
        cv_node = cv_nodes[dep]
        jd_node = jd_nodes[dest - n_cv]
        # Skip anchor nodes — they are no longer in the prompt and should not be matched
        if cv_node.get("source") == "anchor" or jd_node.get("source") == "anchor":
            continue
        pair = _find_matching_pair(cv_node["type"], jd_node["type"], pairs)
        # Also skip if the matched pair is the anchor pair
        if pair and pair["jd_type"] == _ANCHOR_INST_PAIR:
            continue
        if pair is None:
            continue
        cv_id = _resolve_endpoint(cv_node, pair["cv_type"], cv_registry, cv_shadows, cv_isa_edges)
        jd_id = _resolve_endpoint(jd_node, pair["jd_type"], jd_registry, jd_shadows, jd_isa_edges)
        key = (cv_id, jd_id)
        if key in seen:
            continue
        seen.add(key)
        matches.append({"cv_id": cv_id, "jd_id": jd_id})

    return matches, cv_shadows, cv_isa_edges, jd_shadows, jd_isa_edges
