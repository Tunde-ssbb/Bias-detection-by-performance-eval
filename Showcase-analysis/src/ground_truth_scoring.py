"""
Ground Truth Scoring Module

Computes ground truth scores by comparing CV and JD profile dictionaries.
"""

import pandas as pd
import json
from typing import Dict, Any
from pathlib import Path

ED_EXP_IMPORTANCE = 5.0  # Fixed importance (max O*NET level) for education and experience items

def compute_max_score(jd_dict: Dict[str, Any]) -> float:
        # Calculate theoretical min and max scores based on JD importances
    max_score = 0.0
    min_score = 0.0

    for item in jd_dict.values():
        if item.get('kind', '') in ['required education', 'related experience']:
            # Special handling for education/experience items: importance is fixed
            importance = ED_EXP_IMPORTANCE
        else:
            importance = item.get('importance', 1.0)

        max_score += importance  # Matched
        min_score -= importance  # Unmatched


    return max_score, min_score

def normalize_score(raw_score: float, max_score: float, min_score: float) -> float:

    #fallback
    if max_score == min_score:
        return 5.5 

    # Min-max normalization
    normalized = 1 + 9* (raw_score - min_score)  / (max_score - min_score)
    
    return max(1.0, min(10.0, normalized))




def compute_ground_truth_score(cv_dict: Dict[str, Any], jd_dict: Dict[str, Any], normalize: bool = True) -> float:
    """
    Compute ground truth score by comparing CV and JD profile dictionaries.
    
    Scoring Rules:
    - For each item in jd_dict:
      - If item code exists in cv_dict: add importance, linearly, no weighting
        transform (default 1.0)
      - If item code doesn't exist: subtract importance, same convention (default 1.0)
    - Special handling for "required education" and "related experience":
      - Match if prefix before last dot is same AND cv integer >= jd integer

      
    """
    raw_score = 0.0
    max_score, min_score = compute_max_score(jd_dict)
    
    for jd_code, jd_item in jd_dict.items():
        importance = jd_item.get('importance', 1.0)
        kind = jd_item.get('kind', '')
        
        # Check if this is a special case (education or experience)
        if kind in ['required education', 'related experience']:
            matched = check_education_experience_match(jd_code, cv_dict, kind)
            importance = ED_EXP_IMPORTANCE  # Override importance for these items
        else:
            # Regular match - just check if code exists in cv_dict
            matched = jd_code in cv_dict
        
        if matched:
            raw_score += importance
        else:
            raw_score -= importance
    
    if normalize:
        return normalize_score(raw_score, max_score, min_score)
    else:
        return raw_score


def check_education_experience_match(jd_code: str, cv_dict: Dict[str, Any], kind: str) -> bool:
    """
    Check if education/experience requirement is met.
    
    For education and experience items:
    - Codes are like "2.D.1.6" or "3.A.1.7" where the final integer indicates level
    - Extract prefix (e.g., "2.D.1") and final integer (e.g., 6)
    - Match if CV has same prefix AND CV integer >= JD integer
    """

    # Extract prefix and integer from JD code
    parts = jd_code.rsplit('.', 1)
    if len(parts) != 2:
        # No final dot or can't split - fall back to exact match
        return jd_code in cv_dict
    
    jd_prefix, jd_level_str = parts
    try:
        jd_level = int(jd_level_str)
    except ValueError:
        # Not an integer after the last dot - fall back to exact match
        return jd_code in cv_dict
    
    # Look for matching prefix in cv_dict
    for cv_code, cv_item in cv_dict.items():
        # Check if this item has the same kind
        if cv_item.get('kind', '') != kind:
            continue
        
        # Extract prefix and integer from CV code
        cv_parts = cv_code.rsplit('.', 1)
        if len(cv_parts) != 2:
            continue
        
        cv_prefix, cv_level_str = cv_parts
        try:
            cv_level = int(cv_level_str)
        except ValueError:
            continue
        
        # Match if prefix is same and CV level >= JD level
        if cv_prefix == jd_prefix and cv_level >= jd_level:
            return True
    
    return False


def add_ground_truth_scores(csv_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Read CSV with cv_profile_dict and jd_profile_dict columns,
    compute ground truth scores, and add them as a new column.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if 'cv_profile_dict' not in df.columns or 'jd_profile_dict' not in df.columns:
        raise ValueError("CSV must contain 'cv_profile_dict' and 'jd_profile_dict' columns")
    
    print("Computing ground truth scores...")
    scores = []
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse the dict strings (they're stored as JSON strings in CSV)
            cv_dict_str = row['cv_profile_dict']
            jd_dict_str = row['jd_profile_dict']
            
            # Handle missing values
            if pd.isna(cv_dict_str) or pd.isna(jd_dict_str):
                scores.append(None)
                errors += 1
                continue
            
            # Parse JSON
            cv_dict = json.loads(cv_dict_str)
            jd_dict = json.loads(jd_dict_str)
            
            # Compute score
            score = compute_ground_truth_score(cv_dict, jd_dict)
            scores.append(score)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            scores.append(None)
            errors += 1
    
    # Add scores to dataframe
    df['ground_truth_score'] = scores
    
    # Report statistics
    valid_scores = [s for s in scores if s is not None]
    if valid_scores:
        print(f"\nGround Truth Score Statistics:")
        print(f"  Rows processed: {len(df)}")
        print(f"  Valid scores: {len(valid_scores)}")
        print(f"  Errors: {errors}")
        print(f"  Mean score: {sum(valid_scores) / len(valid_scores):.2f}")
        print(f"  Min score: {min(valid_scores):.2f}")
        print(f"  Max score: {max(valid_scores):.2f}")
    
    # Save results
    if output_path is None:
        output_path = csv_path
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved results with ground truth scores to {output_path}")
    
    return df

