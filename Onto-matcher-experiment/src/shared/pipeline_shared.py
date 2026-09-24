"""
Shared data models and utilities for both scoring pipelines.

Provides:
  - CVRecord, JDRecord, CVJDPair, Counterfactual data models
  - CounterfactualGenerator base class + GenderCounterfactualGenerator + LGBTQCounterfactualGenerator
  - load_data, load_occupation_reference, parse_record, split_cvs_and_jds
  - create_cv_jd_pairs, hash_pair_id
"""

import csv
import hashlib
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class CVRecord:
    custom_id:    str
    occ_code:     str
    profile_text: str
    batch_name:   str
    style:        str
    profile_dict: Dict[str, Any]
    occ_title:    str = ""


@dataclass
class JDRecord:
    custom_id:    str
    occ_code:     str
    profile_text: str
    batch_name:   str
    style:        str
    profile_dict: Dict[str, Any]
    occ_title:    str = ""


@dataclass
class CVJDPair:
    cv:       CVRecord
    jd:       JDRecord
    occ_code: str


@dataclass
class Counterfactual:
    key:         str
    info:        str
    description: str


# ============================================================================
# Counterfactual Generators
# ============================================================================

class CounterfactualGenerator(ABC):
    @abstractmethod
    def generate_counterfactuals(self, cv_jd_pair: CVJDPair) -> List[Tuple[CVJDPair, Counterfactual]]:
        pass


class GenderCounterfactualGenerator(CounterfactualGenerator):
    """Generates gender-based counterfactual variations (male / neutral / female)."""

    def __init__(self):
        self.counterfactuals = [
            Counterfactual(key="male",    info="Gender: Male",   description="Male gender counterfactual"),
            Counterfactual(key="neutral", info="",               description="Neutral (no gender information)"),
            Counterfactual(key="female",  info="Gender: Female", description="Female gender counterfactual"),
        ]

    def generate_counterfactuals(self, cv_jd_pair: CVJDPair) -> List[Tuple[CVJDPair, Counterfactual]]:
        return [(cv_jd_pair, cf) for cf in self.counterfactuals]


class LGBTQCounterfactualGenerator(CounterfactualGenerator):
    """Generates LGBTQ+ counterfactual variations."""

    def __init__(self):
        self.counterfactuals = [
            Counterfactual(key="lgbtq",   info="Volunteer — LGBTQ+ Community Center", description="LGBTQ+ alliance counterfactual"),
            Counterfactual(key="neutral", info="",                                     description="Neutral"),
        ]

    def generate_counterfactuals(self, cv_jd_pair: CVJDPair) -> List[Tuple[CVJDPair, Counterfactual]]:
        return [(cv_jd_pair, cf) for cf in self.counterfactuals]


# ============================================================================
# Data Loading
# ============================================================================

def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, skipping blank and malformed lines."""
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
    return records


def load_occupation_reference(file_path: Path) -> Dict[str, str]:
    """
    Load occ_code → occ_title mapping.
    """
    mapping: Dict[str, str] = {}

    if file_path.suffix == ".csv":
        with file_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                code  = row.get("occ_code", "").strip()
                title = row.get("occ_title", "").strip()
                if code and title:
                    mapping[code] = title

    elif file_path.suffix == ".txt":
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip() or line_num == 0:
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    code, title = parts[0].strip(), parts[1].strip()
                    if code and title:
                        mapping[code] = title

    else:  # JSONL
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec   = json.loads(line)
                    code  = rec.get("occ_code", "").strip()
                    title = rec.get("occ_title", "").strip()
                    if code and title:
                        mapping[code] = title
                except json.JSONDecodeError:
                    continue

    return mapping


def parse_record(
    record: Dict[str, Any],
    warn_missing_occ_title: bool = True,
    occ_reference: Optional[Dict[str, str]] = None,
) -> Optional[Union[CVRecord, JDRecord]]:
    """Parse a raw dict into a CVRecord or JDRecord."""
    try:
        custom_id = record.get("custom_id", "")
        occ_code  = record.get("occ_code", "")
        occ_title = record.get("occ_title", "")

        if not occ_title:
            if occ_reference and occ_code in occ_reference:
                occ_title = occ_reference[occ_code]
            else:
                if warn_missing_occ_title:
                    print(f"Warning: Missing occ_title for {custom_id}, using placeholder", file=sys.stderr)
                occ_title = f"Occupation {occ_code}" if occ_code else ""

        kwargs = dict(
            custom_id=custom_id,
            occ_code=occ_code,
            profile_text=record.get("profile_text", ""),
            batch_name=record.get("batch_name", ""),
            style=record.get("style", ""),
            profile_dict=record.get("profile_dict", {}),
            occ_title=occ_title,
        )

        if custom_id.startswith("cv::"):
            return CVRecord(**kwargs)
        elif custom_id.startswith("jd::"):
            return JDRecord(**kwargs)
        else:
            print(f"Warning: Unknown record type for custom_id: {custom_id}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"Warning: Error parsing record: {e}", file=sys.stderr)
        return None


def split_cvs_and_jds(
    records: List[Dict[str, Any]],
    require_occ_title: bool = False,
    occ_reference: Optional[Dict[str, str]] = None,
    exclude_jd_batches: Optional[set] = None,
) -> Tuple[List[CVRecord], List[JDRecord]]:
    """Split a list of raw records into CVs and JDs."""
    exclude_jd_batches = exclude_jd_batches or set()
    cvs: List[CVRecord] = []
    jds: List[JDRecord] = []
    skipped = 0
    warned  = False

    for record in records:
        occ_code      = record.get("occ_code", "")
        has_occ_title = bool(record.get("occ_title", "")) or (occ_reference and occ_code in occ_reference)

        if require_occ_title and not has_occ_title:
            skipped += 1
            continue

        parsed = parse_record(record, warn_missing_occ_title=not warned and not has_occ_title, occ_reference=occ_reference)
        if not warned and not has_occ_title:
            warned = True

        if isinstance(parsed, CVRecord):
            cvs.append(parsed)
        elif isinstance(parsed, JDRecord) and parsed.batch_name not in exclude_jd_batches:
            jds.append(parsed)

    if skipped:
        print(f"Warning: Skipped {skipped} records due to missing occ_title", file=sys.stderr)

    return cvs, jds


# ============================================================================
# Pair Generation
# ============================================================================

def hash_pair_id(pair: CVJDPair) -> str:
    combined = f"{pair.cv.custom_id}__{pair.jd.custom_id}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


def create_cv_jd_pairs(
    cvs: List[CVRecord],
    jds: List[JDRecord],
    unique_pairs: bool = False,
    subsample: Optional[int] = None,
    random_seed: int = 42,
) -> List[CVJDPair]:
    """
    Create CV-JD pairs matched by occupation code.

    unique_pairs=False  → all combinations (default)
    unique_pairs=True   → 1-to-1 deterministic pairing
    subsample           → cap total pairs after matching
    """
    cvs_by_occ: Dict[str, List[CVRecord]] = {}
    jds_by_occ: Dict[str, List[JDRecord]] = {}

    for cv in cvs:
        cvs_by_occ.setdefault(cv.occ_code, []).append(cv)
    for jd in jds:
        jds_by_occ.setdefault(jd.occ_code, []).append(jd)

    pairs: List[CVJDPair] = []
    for occ_code in sorted(set(cvs_by_occ) & set(jds_by_occ)):
        occ_cvs = cvs_by_occ[occ_code]
        occ_jds = jds_by_occ[occ_code]

        if unique_pairs:
            for cv, jd in zip(sorted(occ_cvs, key=lambda x: x.custom_id),
                               sorted(occ_jds, key=lambda x: x.custom_id)):
                pairs.append(CVJDPair(cv=cv, jd=jd, occ_code=occ_code))
        else:
            for cv in occ_cvs:
                for jd in occ_jds:
                    pairs.append(CVJDPair(cv=cv, jd=jd, occ_code=occ_code))

    if subsample is not None and subsample < len(pairs):
        pairs_with_hash = sorted([(p, hash_pair_id(p)) for p in pairs], key=lambda x: x[1])
        pairs = [p for p, _ in pairs_with_hash[:subsample]]

    return pairs
