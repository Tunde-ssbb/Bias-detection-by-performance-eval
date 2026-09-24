"""
Add features to raw data
"""

import json
import re
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from nltk.corpus import stopwords
import nltk

from src.ground_truth_scoring import (
    compute_ground_truth_score,
    check_education_experience_match,
    ED_EXP_IMPORTANCE,
)


# ---------------------------------------------------------------------------
# Step 1: Group ID
# ---------------------------------------------------------------------------

def _add_group_id(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Add group_id column identifying matched CV-JD pairs across counterfactuals."""
    df = df.copy()
    df["group_id"] = df["cv_id"].astype(str) + "_" + df["jd_id"].astype(str)
    if verbose:
        print(f"    group_id                : {df['group_id'].nunique()} unique pairs")
    return df


# ---------------------------------------------------------------------------
# Step 2: Ground truth score  [comp]
# ---------------------------------------------------------------------------

def _add_ground_truth(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Add gt_score_1_10 by comparing cv_profile_dict to jd_profile_dict."""
    scores = []
    n_errors = 0

    for _, row in df.iterrows():
        try:
            cv_dict = row["cv_profile_dict"]
            jd_dict = row["jd_profile_dict"]
            if isinstance(cv_dict, str):
                cv_dict = json.loads(cv_dict)
                jd_dict = json.loads(jd_dict)
            scores.append(compute_ground_truth_score(cv_dict, jd_dict, normalize=True))
        except Exception:
            scores.append(None)
            n_errors += 1

    df = df.copy()
    df["gt_score_1_10"] = scores

    if verbose:
        n_valid = sum(s is not None for s in scores)
        print(f"    gt_score_1_10      : {n_valid}/{len(df)} computed  ({n_errors} errors)")

    return df


# ---------------------------------------------------------------------------
# Step 3: Competence overlap features
# ---------------------------------------------------------------------------

def _add_competence_features(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Add competence overlap and per-kind importance distribution features."""
    df = df.copy()

    def compute(cv_dict, jd_dict) -> dict:
        if isinstance(cv_dict, str):
            cv_dict = json.loads(cv_dict)
        if isinstance(jd_dict, str):
            jd_dict = json.loads(jd_dict)

        if not jd_dict:
            return {"comp_item_overlap": float("nan")}

        per_kind: dict[str, dict] = {}
        total_matched_importance = 0.0
        total_importance = 0.0

        for jd_code, jd_item in jd_dict.items():
            kind = jd_item.get("kind", "unknown")
            importance = jd_item.get("importance", 1.0)

            if kind in ("required education", "related experience"):
                importance = ED_EXP_IMPORTANCE
                matched = check_education_experience_match(jd_code, cv_dict, kind)
            else:
                matched = jd_code in cv_dict

            bucket = per_kind.setdefault(
                kind, {"matched": 0, "unmatched": 0,
                       "matched_importance": 0.0, "unmatched_importance": 0.0,
                       "importances": []}
            )
            bucket["importances"].append(importance)
            if matched:
                bucket["matched"] += 1
                bucket["matched_importance"] += importance
                total_matched_importance += importance
            else:
                bucket["unmatched"] += 1
                bucket["unmatched_importance"] += importance

            total_importance += importance


        cv_kind_totals: dict[str, float] = {}
        for cv_item in cv_dict.values():
            kind = cv_item.get("kind", "unknown")
            if kind in ("required education", "related experience"):
                importance = ED_EXP_IMPORTANCE
            else:
                importance = cv_item.get("importance", 1.0)
            cv_kind_totals[kind] = cv_kind_totals.get(kind, 0.0) + importance

        features: dict = {
            "comp_item_overlap": (
                total_matched_importance / total_importance
                if total_importance > 0 else float("nan")
            ),
        }

        for kind, stats in per_kind.items():
            key = kind.replace(" ", "_")
            kind_total = stats["matched_importance"] + stats["unmatched_importance"]
            n_items = stats["matched"] + stats["unmatched"]
            imps = stats["importances"]

            features[f"comp_overlap_{key}_matched"]                  = stats["matched"]
            features[f"comp_overlap_{key}_unmatched"]                = stats["unmatched"]
            features[f"{key}_matched_importance"]                    = stats["matched_importance"]
            features[f"{key}_unmatched_importance"]                  = stats["unmatched_importance"]
            features[f"comp_overlap_{key}_avg_matched_importance"]   = (
                stats["matched_importance"] / stats["matched"]
                if stats["matched"] > 0 else float("nan")
            )
            features[f"comp_overlap_{key}_avg_unmatched_importance"] = (
                stats["unmatched_importance"] / stats["unmatched"]
                if stats["unmatched"] > 0 else float("nan")
            )
            features[f"comp_overlap_{key}_ratio"]                    = (
                stats["matched_importance"] / kind_total if kind_total > 0 else float("nan")
            )
            features[f"comp_jd_{key}_n_items"]          = n_items
            features[f"comp_jd_{key}_importance_share"] = (
                kind_total / total_importance if total_importance > 0 else float("nan")
            )
            features[f"comp_jd_{key}_avg_importance"]   = kind_total / n_items if n_items > 0 else float("nan")
            features[f"comp_jd_{key}_max_importance"]   = max(imps)

        for kind, total in cv_kind_totals.items():
            key = kind.replace(" ", "_")
            features[f"cv_{key}_total_importance"] = total

        return features

    comp_df = pd.DataFrame(
        df.apply(lambda row: compute(row["cv_profile_dict"], row["jd_profile_dict"]), axis=1).tolist(),
        index=df.index,
    )
    df = pd.concat([df, comp_df], axis=1)

    if verbose:
        print(f"    comp_item_overlap       : avg {df['comp_item_overlap'].mean():.2f}")
        per_kind_cols = [c for c in comp_df.columns if c != "comp_item_overlap"]
        print(f"    comp per-kind cols      : {len(per_kind_cols)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 4: Linguistic features 
# ---------------------------------------------------------------------------

def _load_stopwords() -> frozenset[str]:
    """NLTK's English stopword list (downloaded once on first use)."""
    try:
        return frozenset(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(stopwords.words("english"))


_STOPWORDS = _load_stopwords()


def _count_syllables(word: str) -> int:
    """Rough syllable count via vowel-group heuristic."""
    word = word.lower().rstrip("e")
    count = len(re.findall(r"[aeiou]+", word))
    return max(1, count)


def _ling_features(text: str, prefix: str) -> dict:
    """Compute linguistic features for one text block, keys prefixed with `prefix`."""
    if not text or not text.strip():
        return {f"{prefix}_{k}": float("nan") for k in [
            "word_count", "sentence_count", "char_count",
            "avg_word_len", "avg_sentence_len",
            "type_token_ratio", "hapax_ratio",
            "stopword_ratio", "number_ratio", "uppercase_ratio",
            "avg_syllables_per_word", "flesch_reading_ease",
            "punctuation_density", "bullet_count",
        ]}

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\b[a-zA-Z0-9']+\b", text)
    alpha_words = [w for w in words if re.search(r"[a-zA-Z]", w)]

    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    n_chars = len(text.replace(" ", "").replace("\n", ""))

    word_lengths = [len(w) for w in alpha_words]
    avg_word_len = sum(word_lengths) / len(word_lengths) if word_lengths else float("nan")
    avg_sentence_len = n_words / n_sentences

    lower_words = [w.lower() for w in words]
    freq = Counter(lower_words)
    n_unique = len(freq)
    type_token_ratio = n_unique / n_words if n_words > 0 else float("nan")
    hapax_ratio = sum(1 for c in freq.values() if c == 1) / n_words if n_words > 0 else float("nan")

    stopword_ratio = sum(1 for w in lower_words if w in _STOPWORDS) / n_words if n_words > 0 else float("nan")
    number_ratio = sum(1 for w in words if re.fullmatch(r"\d+[\d.,]*", w)) / n_words if n_words > 0 else float("nan")
    uppercase_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / n_words if n_words > 0 else float("nan")

    syllable_counts = [_count_syllables(w) for w in alpha_words]
    avg_syllables = sum(syllable_counts) / len(syllable_counts) if syllable_counts else float("nan")
    if not math.isnan(avg_syllables):
        flesch = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
        flesch = max(0.0, min(100.0, flesch))
    else:
        flesch = float("nan")

    punct_chars = sum(1 for c in text if c in r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
    punctuation_density = punct_chars / n_chars if n_chars > 0 else float("nan")
    bullet_lines = re.findall(r"(?m)^\s*[-•*–]|\b\d+\.\s", text)
    bullet_count = len(bullet_lines)

    return {
        f"{prefix}_word_count":             n_words,
        f"{prefix}_sentence_count":         len(sentences),
        f"{prefix}_char_count":             n_chars,
        f"{prefix}_avg_word_len":           avg_word_len,
        f"{prefix}_avg_sentence_len":       avg_sentence_len,
        f"{prefix}_type_token_ratio":       type_token_ratio,
        f"{prefix}_hapax_ratio":            hapax_ratio,
        f"{prefix}_stopword_ratio":         stopword_ratio,
        f"{prefix}_number_ratio":           number_ratio,
        f"{prefix}_uppercase_ratio":        uppercase_ratio,
        f"{prefix}_avg_syllables_per_word": avg_syllables,
        f"{prefix}_flesch_reading_ease":    flesch,
        f"{prefix}_punctuation_density":    punctuation_density,
        f"{prefix}_bullet_count":           bullet_count,
    }


def _add_ling_features(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Add linguistic features for cv_profile_text (cv_ling_*) and jd_profile_text (jd_ling_*)."""
    df = df.copy()

    ling_rows = df.apply(
        lambda row: {
            **_ling_features(row.get("cv_profile_text", ""), "cv_ling"),
            **_ling_features(row.get("jd_profile_text", ""), "jd_ling"),
        },
        axis=1,
    )
    ling_df = pd.DataFrame(ling_rows.tolist(), index=df.index)
    df = pd.concat([df, ling_df], axis=1)

    if verbose:
        print(f"    cv_ling_word_count      : avg {df['cv_ling_word_count'].mean():.1f}")
        print(f"    cv_ling_type_token_ratio: avg {df['cv_ling_type_token_ratio'].mean():.3f}")
        print(f"    cv_ling_flesch          : avg {df['cv_ling_flesch_reading_ease'].mean():.1f}")
        print(f"    ling cols               : {len(ling_df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 5: CV–JD text overlap  [ling_text_*]
# ---------------------------------------------------------------------------

def _content_tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[a-z]+\b", text.lower()) if w not in _STOPWORDS]


def _add_text_overlap(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Token Jaccard, JD coverage, and bigram Jaccard between CV and JD text."""
    df = df.copy()

    def compute(cv_text, jd_text) -> dict:
        if not isinstance(cv_text, str) or not isinstance(jd_text, str):
            return {"ling_text_token_jaccard": float("nan"),
                    "ling_text_jd_coverage":   float("nan"),
                    "ling_text_bigram_jaccard": float("nan")}

        cv_toks = set(_content_tokens(cv_text))
        jd_toks = set(_content_tokens(jd_text))
        inter = cv_toks & jd_toks
        union = cv_toks | jd_toks

        token_jaccard  = len(inter) / len(union) if union else float("nan")
        jd_coverage    = len(inter) / len(jd_toks) if jd_toks else float("nan")

        cv_bi = set(zip(cv_toks, list(cv_toks)[1:]))
        jd_bi = set(zip(jd_toks, list(jd_toks)[1:]))
        bi_union = cv_bi | jd_bi
        bigram_jaccard = len(cv_bi & jd_bi) / len(bi_union) if bi_union else float("nan")

        return {"ling_text_token_jaccard":  token_jaccard,
                "ling_text_jd_coverage":    jd_coverage,
                "ling_text_bigram_jaccard": bigram_jaccard}

    feat_df = pd.DataFrame(
        df.apply(lambda r: compute(r.get("cv_profile_text", ""), r.get("jd_profile_text", "")), axis=1).tolist(),
        index=df.index,
    )
    df = pd.concat([df, feat_df], axis=1)

    if verbose:
        print(f"    ling_text_token_jaccard : avg {df['ling_text_token_jaccard'].mean():.3f}")
        print(f"    ling_text_jd_coverage   : avg {df['ling_text_jd_coverage'].mean():.3f}")
        print(f"    ling_text_bigram_jaccard: avg {df['ling_text_bigram_jaccard'].mean():.3f}")
    return df


# ---------------------------------------------------------------------------
# Step 6: JD requirement structure  [comp_jd_*]
# ---------------------------------------------------------------------------

def _add_jd_profile_features(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Aggregate features describing the JD's requirement structure."""
    df = df.copy()

    def compute(jd_dict) -> dict:
        if isinstance(jd_dict, str):
            jd_dict = json.loads(jd_dict)
        if not jd_dict:
            return {k: float("nan") for k in [
                "comp_jd_n_requirements", "comp_jd_sum_importance", "comp_jd_avg_importance",
                "comp_jd_max_importance", "comp_jd_importance_std", "comp_jd_edu_exp_ratio",
            ]}

        importances, n_edu_exp = [], 0
        for item in jd_dict.values():
            kind = item.get("kind", "")
            imp  = ED_EXP_IMPORTANCE if kind in ("required education", "related experience") \
                   else item.get("importance", 1.0)
            importances.append(imp)
            if kind in ("required education", "related experience"):
                n_edu_exp += 1

        n   = len(importances)
        avg = sum(importances) / n
        std = math.sqrt(sum((x - avg) ** 2 for x in importances) / n) if n > 1 else 0.0

        return {
            "comp_jd_n_requirements": n,
            "comp_jd_sum_importance": sum(importances),
            "comp_jd_avg_importance": avg,
            "comp_jd_max_importance": max(importances),
            "comp_jd_importance_std": std,
            "comp_jd_edu_exp_ratio":  n_edu_exp / n,
        }

    feat_df = pd.DataFrame(
        df["jd_profile_dict"].apply(compute).tolist(),
        index=df.index,
    )
    df = pd.concat([df, feat_df], axis=1)

    if verbose:
        print(f"    comp_jd_n_requirements  : avg {df['comp_jd_n_requirements'].mean():.1f}")
        print(f"    comp_jd_avg_importance  : avg {df['comp_jd_avg_importance'].mean():.2f}")
        print(f"    comp_jd_edu_exp_ratio   : avg {df['comp_jd_edu_exp_ratio'].mean():.2f}")
    return df


# ---------------------------------------------------------------------------
# Step 7: CV seniority signals  [cv_ling_*]
# ---------------------------------------------------------------------------

#custom action verbs
_ACTION_VERB_RE = re.compile(
    r"\b(led|managed|built|developed|designed|created|launched|delivered|drove|"
    r"improved|increased|decreased|reduced|optimized|implemented|established|"
    r"spearheaded|oversaw|coordinated|architected|deployed|scaled|mentored|"
    r"negotiated|authored|directed|generated|transformed)\b", re.I)




def _add_cv_signals(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Action verb count from CV text."""
    df = df.copy()

    def compute(text) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {"cv_ling_action_verb_count": float("nan")}
        return {"cv_ling_action_verb_count": len(_ACTION_VERB_RE.findall(text))}

    feat_df = pd.DataFrame(
        df["cv_profile_text"].apply(compute).tolist(),
        index=df.index,
    )
    df = pd.concat([df, feat_df], axis=1)

    if verbose:
        print(f"    cv_ling_action_verb_count: avg {df['cv_ling_action_verb_count'].mean():.1f}")
    return df


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_processed(enriched: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in enriched.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"[post_process] Saved → {path}")


def load_processed(
    experiments: dict[str, object],
    out_dir: Path,
    fallback: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load post-processed parquet files for each experiment.
    """
    out_dir = Path(out_dir)
    result: dict[str, pd.DataFrame] = {}
    for name in experiments:
        path = out_dir / f"{name}.parquet"
        if path.exists():
            result[name] = pd.read_parquet(path)
            print(f"[post_process] Loaded {name} from {path}")
        elif fallback and name in fallback:
            print(
                f"[post_process] WARNING: no processed file for '{name}' "
                f"using raw parsed data (comp features absent). Run with post_process=True first."
            )
            result[name] = fallback[name]
        else:
            print(f"[post_process] WARNING: skipping '{name}'becasue no processed file found.")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def enrich(
    parsed: dict[str, pd.DataFrame],
    verbose: bool = False,
    out_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Apply all row-wise enrichment steps to every experiment DataFrame.

    Returns:
        New dict {experiment_name: enriched_df} (originals are not mutated).
    """
    enriched: dict[str, pd.DataFrame] = {}

    for name, df in parsed.items():
        if verbose:
            print(f"[post_process] {name}:")

        df = _add_group_id(df, verbose)
        df = _add_ground_truth(df, verbose)
        df = _add_competence_features(df, verbose)
        df = _add_ling_features(df, verbose)
        df = _add_text_overlap(df, verbose)
        df = _add_jd_profile_features(df, verbose)
        df = _add_cv_signals(df, verbose)

        if "cv_style" in df.columns:
            df["cv_ling_style"] = df["cv_style"]
        if "jd_style" in df.columns:
            df["jd_ling_style"] = df["jd_style"]

        enriched[name] = df

    if out_dir is not None:
        save_processed(enriched, out_dir)

    return enriched
