#!/usr/bin/env python3
"""
Expand a raw CV/JD dataset into a counterfactual dataset.

Injects demographic signals into each CV's profile_text, producing one record
per (CV × counterfactual variant). JDs pass through unchanged.
"""

import argparse
import json
import sys
from pathlib import Path

from shared.pipeline_shared import (
    CVRecord, CVJDPair, JDRecord,
    GenderCounterfactualGenerator, LGBTQCounterfactualGenerator,
    load_data,
)


def inject_signal(profile_text: str, info: str) -> str:
    """Prepend counterfactual signal to profile_text: "<info>\\n\\n<text>"."""
    if not info:
        return profile_text
    return f"{info}\n\n{profile_text}"


def expand_dataset(records: list, counterfactual_type: str = "gender") -> list:
    """
    Expand a dataset by creating counterfactual variants of every CV.
    JDs pass through once, unchanged (counterfactual_key = "original").
    """
    cf_gen = (LGBTQCounterfactualGenerator() if counterfactual_type == "lgbtq"
              else GenderCounterfactualGenerator())

    dummy_jd = JDRecord(
        custom_id="jd::placeholder", occ_code="", profile_text="",
        batch_name="", style="", profile_dict={},
    )

    output = []
    for rec in records:
        custom_id = rec.get("custom_id", "")

        if custom_id.startswith("jd::"):
            out = dict(rec)
            out["counterfactual_key"]  = "original"
            out["counterfactual_info"] = ""
            output.append(out)

        elif custom_id.startswith("cv::"):
            cv = CVRecord(
                custom_id=custom_id,
                occ_code=rec.get("occ_code", ""),
                profile_text=rec.get("profile_text", ""),
                batch_name=rec.get("batch_name", ""),
                style=rec.get("style", ""),
                profile_dict=rec.get("profile_dict", {}),
                occ_title=rec.get("occ_title", ""),
            )
            pair = CVJDPair(cv=cv, jd=dummy_jd, occ_code=cv.occ_code)

            for _, cf in cf_gen.generate_counterfactuals(pair):
                out = dict(rec)
                out["custom_id"]           = f"{custom_id}::cf_{cf.key}"
                out["profile_text"]        = inject_signal(cv.profile_text, cf.info)
                out["counterfactual_key"]  = cf.key
                out["counterfactual_info"] = cf.info
                out.setdefault("profile_type", "cv")
                output.append(out)

        else:
            print(f"Warning: unknown record type '{custom_id}', skipping", file=sys.stderr)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Expand a dataset with counterfactual CV variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  "-i", required=True, help="Input JSONL dataset")
    parser.add_argument("--output", "-o", required=True, help="Output expanded JSONL")
    parser.add_argument("--counterfactual-type", default="gender", choices=["gender", "lgbtq"],
                        help="Counterfactual type (default: gender)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = load_data(input_path)
    n_cv = sum(1 for r in records if r.get("custom_id", "").startswith("cv::"))
    n_jd = sum(1 for r in records if r.get("custom_id", "").startswith("jd::"))
    print(f"Loaded {len(records)} records  ({n_cv} CVs, {n_jd} JDs)")

    expanded = expand_dataset(records, args.counterfactual_type)
    n_cv_out = sum(1 for r in expanded if r.get("custom_id", "").startswith("cv::"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in expanded:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cf_keys = sorted({r.get("counterfactual_key", "") for r in expanded
                      if r.get("custom_id", "").startswith("cv::")})

    print(f"\n{'='*60}")
    print(f"Output: {output_path}")
    print(f"  {n_cv} CVs × {len(cf_keys)} variants = {n_cv_out} CV records  +  {n_jd} JDs")
    print(f"  Counterfactual keys: {', '.join(cf_keys)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
