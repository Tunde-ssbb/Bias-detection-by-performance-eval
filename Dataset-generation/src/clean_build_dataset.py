"""
Dataset Generation Pipeline - Core Logic

Orchestrates the complete pipeline for generating synthetic CVs and job descriptions:
1. Load and sample O*NET occupational data
2. Build generation prompts with competence items
3. Submit batch requests to LLM APIs (OpenAI & Anthropic)
4. Validate and refine generated content through multiple steps
5. Compile final datasets with metadata

"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union, Optional
from src.llm.llm_client import BatchLLM, AnthropicBatchLLM
from src.llm.prompts import PROMPTS, prompt_builder
from src.util.util import timeit, safe_float
import os
import json
import shutil
import hashlib



DATA = Path("data/Onet_text_files")  # O*NET source data directory

# Style requirements for generation (used when --use-multiple-styles is enabled)
STYLE_REQUIREMENTS = {
    "concise": "Write in a concise and brief style, using short sentences and minimal elaboration",
    "detailed": "Write in a detailed and comprehensive style, providing thorough explanations and context",
    "formal": "Write in a formal and professional style, using industry-standard terminology and structured language",
    "casual": "Write in a conversational and approachable style, using natural language while maintaining professionalism",
}

# Functions to read and parse O*NET TSV files with appropriate column types
# and merge related datasets.

def read_onet_tsv(name, usecols=None, dtype=None):
    """
    Generic TSV reader for O*NET files.
    
    Args:
        name: Filename (without .txt extension) in data/Onet_text_files/
        usecols: Optional list of columns to load
        dtype: Optional dict of column dtypes
    
    Returns:
        Pandas DataFrame
    """
    return pd.read_csv(
        DATA / f"{name}.txt",
        sep="\t",
        quotechar='"',
        encoding="utf-8",
        low_memory=False,
        usecols=usecols,
        dtype=dtype,
    )

def read_data_occupations() -> pd.DataFrame:
    """
    Load O*NET occupations with employment statistics.
    
    Filters to broad and detailed occupations (excludes major/minor groups).
    Merges with BLS employment data when available.
    
    Returns:
        DataFrame with O*NET-SOC Code, Title, Description, and employment stats
    """
    new_names = {"O*NET-SOC 2019 Code": "O*NET-SOC Code", "O*NET-SOC 2019 Title": "Title","O*NET-SOC 2019 Description": "Description"}
    data_occs =  pd.read_csv(DATA/"Data_Collection_Plan.csv",
            quotechar='"',
            encoding="utf-8",
            low_memory=False,
            )
    data_occs.rename(columns=new_names, inplace=True)
    data_occs["SOC_BASE"] = data_occs["O*NET-SOC Code"].str.split(".").str[0]

    # Remove major and minor class occupations
    data_occs = data_occs[~data_occs["SOC_BASE"].str.endswith("000")]

    # adding number of employees per occupation
    bls_df = read_bls_data()

    data_occs = data_occs.merge(
        bls_df[["SOC_BASE", "TOT_EMP"]],
        on="SOC_BASE",
        how="left",
        validate="m:1"
    )

    return data_occs

def read_bls_data() -> pd.DataFrame:
    """
    Read employement statistics from BLS
    """
    bls_df = pd.read_excel(DATA/"national_M2024_dl.xlsx",
    usecols=["OCC_CODE", "TOT_EMP"]
    )
    bls_df.rename(columns={"OCC_CODE":"SOC_BASE"}, inplace=True)
    bls_df = bls_df.drop_duplicates(subset="SOC_BASE")
    return bls_df

def read_content_reference() -> pd.DataFrame :
    """
    Read element descriptions from O*net
    """
    reference = read_onet_tsv(
    "Content Model Reference",
    usecols=["Element ID", "Element Name", "Description"],
    dtype={"Element ID": "string", "Element Name": "string", "Description":"string"}   
    )
    return reference

def read_skills() -> pd.DataFrame:
    """
    Read occupation skills from O*net
    - Supress "recommend supress" adn "not relevant" skills
    - Pivot importance and level for each skill
    """
    skills = read_onet_tsv(
    "Skills",
    usecols=["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value", "Recommend Suppress", "Not Relevant"],
    dtype={"O*NET-SOC Code": "string", "Element ID": "string", "Element Name": "string", "Scale ID":"string", "Data Value": "Float64", "Recommend Suppress": "string", "Not Relevant": "string"}
    )

    # Filter irrelevant or low accuracy skills
    skills = skills[
        (skills["Recommend Suppress"] != "Y") &
        (skills["Not Relevant"].isna() | (skills["Not Relevant"] != "Y"))
    ]

    skills = skills.drop(["Recommend Suppress", "Not Relevant"], axis = 1)

    # Pivot table for usability
    skills = (
    skills.pivot_table(
        index=["O*NET-SOC Code", "Element ID", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="first",  
    )
    .rename(columns={"IM": "importance", "LV": "level"})
    .reset_index()
    )
    return skills

def read_knowledge() -> pd.DataFrame:
    """
    Read occupation knowledge from O*net
    - Supress "recommend supress" adn "not relevant" knowledge
    - Pivot importance and level for each knowledge item
    """

    knowledge = read_onet_tsv(
    "Knowledge",
    usecols=["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value", "Recommend Suppress", "Not Relevant"],
    dtype={"O*NET-SOC Code": "string", "Element ID": "string", "Element Name": "string", "Scale ID":"string", "Data Value": "Float64", "Recommend Suppress": "string", "Not Relevant": "string"}
    )


    # Filter irrelevant or low accuracy skills
    knowledge = knowledge[
        (knowledge["Recommend Suppress"] != "Y") &
        (knowledge["Not Relevant"].isna() | (knowledge["Not Relevant"] != "Y"))
    ]
     
    knowledge = knowledge.drop(["Recommend Suppress", "Not Relevant"], axis = 1)

    knowledge = (
    knowledge.pivot_table(
        index=["O*NET-SOC Code", "Element ID", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="first",   # or "mean" if you can have duplicates
    )
    .rename(columns={"IM": "importance", "LV": "level"})
    .reset_index()
    )
    return knowledge

def read_abilities() -> pd.DataFrame:
    """
    Read occupation abilities from O*net
    - Supress "recommend supress" adn "not relevant" abilities
    - Pivot importance and level for each ability
    """
    abilities = read_onet_tsv(
    "Abilities",
    usecols=["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Data Value", "Recommend Suppress", "Not Relevant"],
    dtype={"O*NET-SOC Code": "string", "Element ID": "string", "Element Name": "string", "Scale ID":"string", "Data Value": "Float64", "Recommend Suppress": "string", "Not Relevant": "string"}
    )


    # Filter irrelevant or low accuracy skills
    abilities = abilities[
        (abilities["Recommend Suppress"] != "Y") &
        (abilities["Not Relevant"].isna() | (abilities["Not Relevant"] != "Y"))
    ]

    abilities = abilities.drop(["Recommend Suppress", "Not Relevant"], axis = 1)

    abilities = (
    abilities.pivot_table(
        index=["O*NET-SOC Code", "Element ID", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="first",   # or "mean" if you can have duplicates
    )
    .rename(columns={"IM": "importance", "LV": "level"})
    .reset_index()
    )

    return abilities

def read_tech_skills() -> pd.DataFrame:
    """
    Read occupation tech skills from O*net
    """

    tech_skills = read_onet_tsv(
    "Technology Skills",
    usecols=["O*NET-SOC Code", "Example",	"Commodity Code",	"Commodity Title",	"Hot Technology",	"In Demand"],
    dtype={"O*NET-SOC Code": "string", "Example": "string", "Commodity Code": "string", "Commodity Title":"string", "Hot Technology": "string", "In Demand": "string"}
    )

    return tech_skills

def read_experience() -> pd.DataFrame:
    """
    Read occupation Education, and experience data from O*net
    Read category descirptions for edocation and experience
    """
    experience = read_onet_tsv(
    "Education, Training, and Experience",
    usecols=["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID", "Category", "Data Value", "Recommend Suppress"],
    dtype={"O*NET-SOC Code": "string", "Element ID": "string", "Element Name": "string", "Category": "Int64", "Scale ID":"string", "Data Value": "Float64", "Recommend Suppress": "string"}
    )

    experience_cat = read_onet_tsv(
    "Education, Training, and Experience Categories",
    usecols=[ "Element ID", "Element Name", "Scale ID", "Category", "Category Description"],
    dtype={ "Element ID": "string", "Element Name": "string", "Category": "Int64", "Scale ID":"string", "Category Description": "string"}
    )
    return experience, experience_cat

def read_work_styles() -> pd.DataFrame:
    """
    Read occupation work styles from O*net
    - Pivot style impact and distinctiveness for each skill
    """
    work_styles = read_onet_tsv(
    "Work Styles",
    usecols=["O*NET-SOC Code", "Element ID", "Element Name", "Scale ID","Data Value"],
    dtype={"O*NET-SOC Code": "string", "Element ID": "string", "Element Name": "string", "Scale ID":"string", "Data Value": "Float64"}
    )

    work_styles = (
    work_styles.pivot_table(
        index=["O*NET-SOC Code", "Element ID", "Element Name"],
        columns="Scale ID",
        values="Data Value",
        aggfunc="first",   # or "mean" if you can have duplicates
    )
    .rename(columns={"WI": "style impact", "DR": "distinctiveness"})
    .reset_index()
    )

    return work_styles


def read_tasks() -> pd.DataFrame:
    """
    Read occupation tasks from O*net
    Read task importance categories
    """

    tasks = read_onet_tsv(
    "Task Statements",
    usecols=["O*NET-SOC Code", "Task ID", "Task", "Task Type"],
    dtype={"O*NET-SOC Code": "string", "Task ID": "Int64", "Task": "string", "Task Type":"string"}
    )

    tasks_cat = read_onet_tsv(
    "Task Ratings",
    usecols=["O*NET-SOC Code", "Task ID", "Scale ID", "Data Value"],
    dtype={"O*NET-SOC Code": "string", "Task ID": "Int64", "Scale ID": "string", "Data Value":"Float64"}
    )

    tasks_cat = tasks_cat[tasks_cat["Scale ID"] == "IM"]
    return tasks, tasks_cat


"""
Other helpers
"""

def df_to_items(df, kind, id_col="Element ID", name_col=None, description_col=None, imp_col=None):
    """
    Helper to turn DFs into dict formatted items
    """
    if not isinstance(df, pd.DataFrame):
        return {}


    items = {
        row[id_col]: {
            **({"importance": imp} if imp is not None else {}),
            **({"name": str(row[name_col])} if name_col is not None else {}),
            **({"description": str(row[description_col])} if description_col is not None else {}),
            "kind": kind,
        }
        for _, row in df.iterrows()
        for imp in [safe_float(row[imp_col]) if imp_col is not None else None]
    }

    return items


def imp_fl_to_str(imp: float) -> str:
    """
    Importance rating into string
    strings taken form O*net original surveys
    """
    if imp < 0 or imp > 5:
        print(f"importance {imp} out of bounds")
        imp_str = ""
    elif imp <= 1:
        imp_str = "not important"
    elif imp <=2:
        imp_str = "somewhat important"
    elif imp <= 3:
        imp_str = "important"
    elif imp <= 4:
        imp_str = "very important"
    elif imp <= 5:
        imp_str = "extremely important"

    return imp_str

def profile_to_string(profile:dict, str_type = "cv"):
    """
    Helper to turn dict formatted items into readable string for prompt input
    - description and importance optiona;=l
    - output: '[kind] - [name][[importance]]: [description]'
   """

    if str_type == "jd":
        result = ""

        for item_code, item_info in profile.items():


            basic_str = f"{item_info.get('kind', '')} - {item_info.get('name', '')}"
            
            imp_str = (
                f" [{imp_fl_to_str(safe_float(item_info['importance']))}]"
                if "importance" in item_info
                else ""
            )

            desc_str = (
                f": {item_info.get('description')}"
                if item_info.get("description")
                else ""
            )

            item_str = basic_str + imp_str + desc_str + "\n"
            result += item_str

        return result
    if str_type == "cv":
        result = ""

        for item_code, item_info in profile.items():

            basic_str = f"{item_info.get('kind', '')} - {item_info.get('name', '')}"

            desc_str = (
                f": {item_info.get('description')}"
                if item_info.get("description")
                else ""
            )

            item_str = basic_str + desc_str + "\n"
            result += item_str

        return result


"""
Saving Utilities
""" 

def to_json_safe(obj):
    if obj is pd.NA:
        return None
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj

def copy_jsonl(src:str, dest:str):
    src = Path(src)
    dest = Path(dest)
    if not os.path.exists(src):
        raise ValueError(f"cannot copy because {src} does not exist")

    with open(src) as src_file, open(dest, "w") as dest_file:
        for line in src_file:
            dest_file.write(line)
    
    return


"""
Building utilities
"""
def top_majors_with_top_detailed_report(
    df: pd.DataFrame,
    soc_col: str = "SOC_BASE",
    emp_col: str = "TOT_EMP",
    title_col: str = "Title",
    n_major: int = 20,
    n_detailed: int = 5,
    ) -> pd.DataFrame:
    """
    1) Select the top n_major SOC major groups (xx-0000) by total employment.
    2) For each selected major group, select the top n_detailed *detailed* SOCs (xx-xxxx, last digit != 0).
    3) Print a report with the major title + indented detailed occupations.
    4) Return the subset of ORIGINAL df rows whose SOC is in the selected detailed SOC set.

    Notes:
    - If df contains duplicates per SOC (e.g., multiple geos), we aggregate for ranking,
      but we return original rows filtered by selected SOCs.
    - Major titles are derived from the row in df where soc == MAJOR_CODE, if present.
      Otherwise we fall back to a placeholder like "Major group 29-0000".
    """

    df = df.copy()

    # Normalize SOC codes (strip O*NET decimals if present)
    df[soc_col] = (
        df[soc_col].astype(str)
        .str.strip()
        .str.split(".", n=1).str[0]
    )
    df[emp_col] = pd.to_numeric(df[emp_col], errors="coerce")

    # Aggregate employment per SOC for ranking
    soc_emp = (
        df.dropna(subset=[soc_col, emp_col])
          .groupby(soc_col, as_index=False)[emp_col]
          .sum()
    )
    soc_emp["MAJOR_CODE"] = soc_emp[soc_col].str.slice(0, 2) + "-0000"

    # Top major groups by total employment
    major_totals = (
        soc_emp.groupby("MAJOR_CODE", as_index=False)[emp_col]
               .sum()
               .rename(columns={emp_col: "MAJOR_TOT_EMP"})
               .sort_values("MAJOR_TOT_EMP", ascending=False)
               .head(n_major)
    )

    # Keep only detailed SOCs (exclude xx-0000, xx-x000, xx-xx0x broad groups)
    is_detailed = (
        soc_emp[soc_col].str.match(r"^\d{2}-\d{4}$", na=False)
        & (soc_emp[soc_col].str[-1] != "0")
    )
    detailed = soc_emp[is_detailed].copy()

    # Helper to get a title for a code from the original df (if available)
    def title_for(code: str) -> str:
        if title_col in df.columns:
            rows = df.loc[df[soc_col] == code, title_col]
            if not rows.empty:
                return str(rows.iloc[0])
        return f"Major group {code}"

    selected_detailed_codes = set()

    # Print + select in one pass for consistency
    for _, maj in major_totals.iterrows():
        major_code = maj["MAJOR_CODE"]
        major_emp = maj["MAJOR_TOT_EMP"]

        major_title = title_for(major_code)
        print(f"{major_code}  |  {major_title}  (Total Employment: {major_emp:,.0f})")

        top_dets = (
            detailed[detailed["MAJOR_CODE"] == major_code]
            .sort_values(emp_col, ascending=False)
            .head(n_detailed)
        )

        for _, det in top_dets.iterrows():
            det_code = det[soc_col]
            det_emp = det[emp_col]

            det_title = title_for(det_code) if title_col in df.columns else ""
            if det_title:
                print(f"    └─ {det_code}  |  {det_title}  ({det_emp:,.0f})")
            else:
                print(f"    └─ {det_code}  ({det_emp:,.0f})")

            selected_detailed_codes.add(det_code)

        print()
    
    # Return subset of original df rows for selected detailed SOCs
    result_df = df[df[soc_col].isin(selected_detailed_codes)].copy()

    # remove duplicate soc codes (pick first sample for each selected code)
    result_df = result_df.drop_duplicates(subset=[soc_col]).copy()
    return  result_df

def build_competence_profiles(occ_code, 
                                n_profiles=1, 
                                n_skills = 1, 
                                n_knowledge = 1, 
                                n_abilities = 1,
                                n_tasks = 1, 
                                n_tech_skills = 1, 
                                n_work_styles = 1) -> List[Tuple[str, dict]]:
    """
    Builds competence profiles for the occupations represented by occ_code referencing O*net SOC codes.
    Each profile consists of:
    - n_skills skills (randomly sampled and weighted by importance)
    - n_knowlegde knowledge items (randomly sampled and weighted by importance)
    - n_abilities abilities (randomly sampled and weighted by importance)
    - n_tasks tasks (randomly sampled and weighted by importance)
    - n_tech_skills tech skills (randomly sampled)
    - n_work_styles work styles (randomly sampled)
    - 1 required education category
    - 1 related experience category

    The profiles are returned as a dictionary of O*net codes where each code is valued by:
    - Name
    - Kind
    - (Optional) Importance
    - (Optional) Description

    """
    max_fails = 3

    # Filter competence items to occupations and where applicable merge with element descriptions
    occ_skills = SKILLS[SKILLS["O*NET-SOC Code"] == occ_code]
    occ_skills = occ_skills.merge(CONTENT_REFERENCE[["Element ID", "Description"]], on="Element ID", how="left", validate="m:1")
        
    occ_knowledge = KNOWLEDGE[KNOWLEDGE["O*NET-SOC Code"] == occ_code]
    occ_knowledge = occ_knowledge.merge(CONTENT_REFERENCE[["Element ID", "Description"]], on="Element ID", how="left", validate="m:1")

    occ_abilities = ABILITIES[ABILITIES["O*NET-SOC Code"] == occ_code]
    occ_abilities = occ_abilities.merge(CONTENT_REFERENCE[["Element ID", "Description"]], on="Element ID", how="left", validate="m:1")

    occ_experience = EXPERIENCE[EXPERIENCE["O*NET-SOC Code"] == occ_code].merge(EXPERIENCE_CAT[["Element ID", "Category", "Category Description"]], on=["Element ID", "Category"], how="left")

    occ_req_ed = occ_experience[occ_experience["Element Name"] == "Required Level of Education"]
    occ_rel_exp = occ_experience[occ_experience["Element Name"] == "Related Work Experience"]
    # occ_prof_cert = occ_experience[occ_experience["Element Name"] == "Job-Related Professional Certification"] #include later
    # occ_appr = occ_experience[occ_experience["Element Name"] == "Job-related Apprenticeship"] #include later

    occ_tech_skills = TECH_SKILLS[TECH_SKILLS["O*NET-SOC Code"] == occ_code]

    occ_work_styles = WORK_STYLES[WORK_STYLES["O*NET-SOC Code"] == occ_code]
    occ_work_styles = occ_work_styles.merge(CONTENT_REFERENCE[["Element ID", "Description"]], on="Element ID", how="left", validate="m:1")

    occ_tasks = TASKS[TASKS["O*NET-SOC Code"] == occ_code].merge(TASKS_CAT[["Task ID", "Data Value"]], on=["Task ID"], how="left")

    # Build n profiles
    profiles = []
    n_succes = 0
    n_fails = 0
    while n_succes < n_profiles and n_fails <= max_fails:
        try:
            # Sample items from occupation items
            profile_skills = occ_skills.sample(weights=occ_skills["importance"]**2 if occ_skills["importance"].sum() > 0 else None, n = min(n_skills, len(occ_skills)))
            profile_knowledge = occ_knowledge.sample(weights=occ_knowledge["importance"]**2 if occ_knowledge["importance"].sum() > 0 else None, n=min(n_knowledge, len(occ_knowledge)))
            profile_abilities = occ_abilities.sample(weights=occ_abilities["importance"]**2  if occ_abilities["importance"].sum() > 0 else None, n=min(n_abilities, len(occ_abilities)))
            profile_tasks = occ_tasks.sample(weights=occ_tasks["Data Value"]**2  if occ_tasks["Data Value"].sum() > 0 else None, n=min(n_tasks, len(occ_tasks)))
            profile_tech_skills = occ_tech_skills.sample(n=min(n_tech_skills, len(occ_tech_skills)))
            profile_work_styles = occ_work_styles.sample(n=min(n_work_styles, len(occ_work_styles)))

            profile_req_ed = occ_req_ed.sample(weights=occ_req_ed["Data Value"] if occ_req_ed["Data Value"].sum() > 0 else None ) if len(occ_req_ed) > 0 else None
            profile_rel_exp = occ_rel_exp.sample(weights=occ_rel_exp["Data Value"]if occ_rel_exp["Data Value"].sum() > 0 else None) if len(occ_rel_exp) > 0 else None
   
            # Create composite IDs for experience items (Element ID + Category) to uniquely identify each category
            if profile_req_ed is not None and len(profile_req_ed) > 0:
                profile_req_ed = profile_req_ed.copy()
                profile_req_ed["Composite ID"] = profile_req_ed["Element ID"].astype(str) + '.' + profile_req_ed["Category"].astype(str)
            if profile_rel_exp is not None and len(profile_rel_exp) > 0:
                profile_rel_exp = profile_rel_exp.copy()
                profile_rel_exp["Composite ID"] = profile_rel_exp["Element ID"].astype(str) + '.' + profile_rel_exp["Category"].astype(str)

            # turn items into dict
            items_dict = {}

            skill_dict = df_to_items(profile_skills, "skill", id_col="Element ID", imp_col="importance", name_col="Element Name", description_col="Description")
            knowledge_dict = df_to_items(profile_knowledge, "knowledge",  id_col="Element ID", imp_col="importance", name_col="Element Name", description_col="Description")
            abilities_dict = df_to_items(profile_abilities, "abilities",  id_col="Element ID", imp_col="importance", name_col="Element Name", description_col="Description")
            task_dict = df_to_items(profile_tasks, "task", id_col="Task ID", imp_col="Data Value", description_col="Task")
            tech_skill_dict = df_to_items(profile_tech_skills, "tech_skill", id_col="Commodity Code", name_col="Commodity Title", description_col="Example")     
            work_style_dict = df_to_items(profile_work_styles, "work_style", id_col="Element ID", name_col="Element Name", description_col="Description") 
            req_ed_dict = df_to_items(profile_req_ed, "required education", id_col="Composite ID", name_col="Category Description")
            rel_exp_dict = df_to_items(profile_rel_exp, "related experience", id_col="Composite ID", name_col="Category Description")

            items_dict.update(skill_dict)
            items_dict.update(knowledge_dict)
            items_dict.update(abilities_dict)
            items_dict.update(task_dict)
            items_dict.update(tech_skill_dict)
            items_dict.update(work_style_dict)
            items_dict.update(req_ed_dict)
            items_dict.update(rel_exp_dict)

            profiles.append(items_dict)
            n_succes += 1
            # print(f"built profile {n_succes}/{n_profiles}")
            
        except Exception as e:
            n_fails += 1
            print(f"attempt failed, trying again: {repr(e)}")
            
            continue
    return profiles

def build_cv_prompt(occ_title: str, profile_dict:dict, style: str = "detailed") -> str:
    prompt_base = PROMPTS["build_cv_v4"]
    profile_string = profile_to_string(profile_dict, str_type="cv")
    style_req = STYLE_REQUIREMENTS.get(style, STYLE_REQUIREMENTS["detailed"])
    prompt = prompt_builder(prompt_base, {"JOB_TITLE": occ_title, "COMPETENCE_ITEMS": profile_string, "STYLE_REQ": style_req})
    return prompt

def build_jd_prompt(occ_title: str, profile_dict:dict, style: str = "detailed") -> str:
    prompt_base = PROMPTS["build_jd_v4"]
    profile_string = profile_to_string(profile_dict, str_type="jd")
    style_req = STYLE_REQUIREMENTS.get(style, STYLE_REQUIREMENTS["detailed"])
    prompt = prompt_builder(prompt_base, {"JOB_TITLE": occ_title, "COMPETENCE_ITEMS": profile_string, "STYLE_REQ": style_req})
    return prompt

def build_check_prompt(custom_id: str, prev_text: str, it_dict:dict) -> str:

    base_prompt = PROMPTS["checker_v2"]
    if "cv" in custom_id:
        it_str = profile_to_string(it_dict, str_type= "cv")
    else:
        it_str = profile_to_string(it_dict, str_type = "jd")
    prompt = prompt_builder(base_prompt, {"TEXT":prev_text, "COMPETENCE_ITEMS":it_str})

    return prompt

# Detect, categorize, and retry failed batch requests

def analyze_batch_failures(results_file: Path, batch_llm, input_file: Optional[Path] = None) -> dict:
    """
    Analyze batch results to identify and categorize failures.
    
    Categorizes errors into: token_limit, content_filter, http_error, other_error, missing
    
    Args:
        results_file: Path to batch results JSONL
        batch_llm: LLM client for parsing results
        input_file: Optional path to input file to detect completely missing results
    
    Returns:
        Dict with 'successful', 'failed', and 'categories' keys
    """
    """
    Analyze a batch results file and categorize failures.
    Returns dict with failure statistics and failed requests.
    """
    failures = {
        "token_limit": [],
        "content_filter": [],
        "http_error": [],
        "missing": [],  # Requests with no result at all
        "other_error": [],
        "success": 0
    }
    
    # Track which custom_ids we've seen in results
    result_ids = set()
    
    with results_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            result_line = json.loads(raw_line)
            custom_id, text, error = batch_llm.parse_batch_line(result_line)
            result_ids.add(custom_id)
            
            if error:
                if "Token limit" in error or "max_output_tokens" in error:
                    failures["token_limit"].append((custom_id, error, result_line))
                elif "content_filter" in error.lower():
                    failures["content_filter"].append((custom_id, error, result_line))
                elif "HTTP" in error:
                    failures["http_error"].append((custom_id, error, result_line))
                else:
                    failures["other_error"].append((custom_id, error, result_line))
            else:
                failures["success"] += 1
    
    # Check for completely missing results
    if input_file and input_file.exists():
        with input_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                input_line = json.loads(raw_line)
                input_id = input_line.get("custom_id")
                
                if not input_id:
                    continue
                
                # Decode input_id if needed (for Anthropic encoded IDs)
                # batch_llm.parse_batch_line returns decoded IDs, so we need to compare decoded to decoded
                input_id_to_compare = input_id
                if hasattr(batch_llm, 'decode_custom_id'):
                    # Check if ID looks encoded (base64 format, no colons)
                    if '::' not in input_id and ':' not in input_id:
                        try:
                            input_id_to_compare = batch_llm.decode_custom_id(input_id)
                        except:
                            # If decode fails, use original
                            pass
                
                if input_id_to_compare not in result_ids:
                    # Create a placeholder result_line for missing requests
                    placeholder_line = {
                        "custom_id": input_id_to_compare,  # Use decoded ID for consistency
                        "error": "Missing result - no response received from API",
                        "text": None
                    }
                    failures["missing"].append((input_id_to_compare, "Missing result - no response received from API", placeholder_line))
    
    return failures

def retry_failed_requests(failed_items: list, dict_path: Path, batch_llm, retry_batch_path: Path, input_file: Optional[Path] = None) -> Path:
    """
    Create a new batch file with retry requests for failed items.
    
    Args:
        failed_items: List of (custom_id, error, result_line) tuples
        dict_path: Path to dict file with competence profiles
        batch_llm: LLM client for building requests
        retry_batch_path: Path to save retry batch
        input_file: Optional path to input file (needed for missing results)
    
    Returns path to retry batch file.
    """
    # Load dict map
    dict_map = {}
    with dict_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if cid:
                dict_map[cid] = obj
    
    # Load input map for missing requests
    # Store both encoded and decoded versions as keys for easy lookup
    input_map = {}
    if input_file and input_file.exists():
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj.get("custom_id")
                if cid:
                    input_map[cid] = obj  # encoded key
                    # Also add decoded key if batch_llm supports decoding
                    if hasattr(batch_llm, 'decode_custom_id'):
                        if '::' not in cid and ':' not in cid:  # looks encoded
                            try:
                                decoded_cid = batch_llm.decode_custom_id(cid)
                                input_map[decoded_cid] = obj  # decoded key
                            except:
                                pass
    
    # Build retry batch
    retry_batch_path.parent.mkdir(parents=True, exist_ok=True)
    retry_count = 0
    
    for custom_id, error, original_line in failed_items:
        dict_entry = dict_map.get(custom_id)
        if not dict_entry:
            print(f"Warning: No dict entry found for {custom_id}")
            continue
        
        # Check if this is a missing result (no response from API)
        is_missing = "Missing result" in error
        
        if is_missing and custom_id in input_map:
            # For missing results, copy the original input request directly
            input_request = input_map[custom_id]
            with retry_batch_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(input_request, ensure_ascii=False) + "\n")
            retry_count += 1
            continue
        
        # Get original request from the failed line
        response_obj = original_line.get("response", {})
        body = response_obj.get("body", {})
        
        # Extract original prompt from the request
        # For step 1 failures, we need to rebuild from dict
        if dict_entry:
            profile_dict = dict_entry.get("dict")
            style = dict_entry.get("style", "realistic")
            
            # Extract occupation code from custom_id (format: type::batch::occ::style::idx)
            cid_parts = custom_id.split("::")
            occ_code = cid_parts[2] if len(cid_parts) >= 3 else None
            
            # Look up occupation title from DATA_OCCUPATIONS
            occ_title = "[Job]"  # fallback
            if occ_code:
                occ_row = DATA_OCCUPATIONS[DATA_OCCUPATIONS["O*NET-SOC Code"] == occ_code]
                if not occ_row.empty:
                    occ_title = occ_row.iloc[0]["Title"]
            
            # Determine if JD or CV from custom_id
            if custom_id.startswith("jd::"):
                prompt_base = PROMPTS["build_jd_v3"]
                profile_string = profile_to_string(profile_dict, str_type="jd")
                style_req = STYLE_REQUIREMENTS.get(style, "Write the job description in a realistic style")
                prompt = prompt_builder(prompt_base, {"JOB_TITLE": occ_title, "COMPETENCE_ITEMS": profile_string, "STYLE_REQ": style_req})
            else:  # CV
                prompt_base = PROMPTS["build_cv_v3"]
                profile_string = profile_to_string(profile_dict, str_type="cv")
                style_req = STYLE_REQUIREMENTS.get(style, "Write the resume in a realistic style")
                prompt = prompt_builder(prompt_base, {"JOB_TITLE": occ_title, "COMPETENCE_ITEMS": profile_string, "STYLE_REQ": style_req})
            
            batch_llm.build_request(prompt, custom_id, save_path=retry_batch_path)
            retry_count += 1
    
    print(f"Created retry batch with {retry_count} requests at {retry_batch_path}")
    return retry_batch_path

def merge_retry_results(original_results: Path, retry_results: Path, output_path: Path):
    """
    Merge retry results back into original results, replacing failed entries and adding missing ones.
    """
    # Load retry results into map
    retry_map = {}
    with retry_results.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            if custom_id:
                retry_map[custom_id] = obj
    
    # Track which retry results we've used
    used_retry_ids = set()
    
    # Merge with original (replace existing entries with retry results)
    replaced_count = 0
    with original_results.open("r", encoding="utf-8") as orig, output_path.open("w", encoding="utf-8") as out:
        for line in orig:
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            
            # If we have a retry result, use it instead
            if custom_id in retry_map:
                out.write(json.dumps(retry_map[custom_id]) + "\n")
                used_retry_ids.add(custom_id)
                replaced_count += 1
            else:
                out.write(line)
        
        # Append any retry results that weren't in the original (missing requests)
        added_count = 0
        for custom_id, retry_obj in retry_map.items():
            if custom_id not in used_retry_ids:
                out.write(json.dumps(retry_obj) + "\n")
                added_count += 1
    
    print(f"Merged retry results: {replaced_count} replaced, {added_count} added (were missing)")
    return replaced_count + added_count


def build_new_batch_from_results(
    batch_llm_results,  # BatchLLM instance
    batch_llm_input,
    results_file: Path,
    dict_file: Path,
    next_batch_file: Path,
):
    """
    b -> BatchLLM instance
    results_file -> JSONL from batch output
    dict_file -> Original dicts JSONL containing custom_id
    processor -> function(text, original_obj) -> processed_result

    Skips:
      - errored responses
      - unmatched custom_id
      - processor failures
    """

    next_batch_file = Path(next_batch_file)
    results_file = Path(results_file)
    dict_file = Path(dict_file)

    if not next_batch_file.exists():
        next_batch_file.parent.mkdir(parents=True, exist_ok=True)
        next_batch_file.touch()

    if next_batch_file.stat().st_size != 0:
        print(f" file {next_batch_file} is not empty, current porcess will append lines which may result in duplicate id's. Clear file and continue? (yes, no)")
        answer = input()
        if answer == "yes":
            next_batch_file.write_text("")  # clear file
            print("File cleared.\n")
        else:
            print(f" Do you want to append lines which may result in duplicate id's and continue? (yes, no)")
            answer = input()
            if answer == "no":
                return

    # 1️⃣ Build lookup map
    dict_map = {}

    with dict_file.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            item_dict = obj.get("dict")
            if cid:
                dict_map[cid] = item_dict
    print("Loaded dict entries:", len(dict_map))

    processed_count = 0
    skipped_errors = 0
    skipped_missing = 0
    skipped_incomplete = 0
    error_details = []

    with results_file.open("r", encoding="utf-8") as f:

        for raw_line in f:
            result_line = json.loads(raw_line)
   
            custom_id, text, error = batch_llm_results.parse_batch_line(result_line)

            if error:
                skipped_errors += 1
                error_details.append((custom_id, error))
                continue

            item_dict = dict_map.get(custom_id)
            if not item_dict:
                skipped_missing += 1
                print(f"Warning: No dict entry found for {custom_id}")
                continue

            # 4️⃣ Process text
            try:
                prompt = build_check_prompt(custom_id, text, item_dict)

                batch_llm_input.build_request(prompt, custom_id, next_batch_file)

                processed_count += 1

            except Exception as ex:
                print(f"Processing failure for {custom_id}: {repr(ex)}")

    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"✓ Processed: {processed_count}")
    print(f"✗ Skipped (errors): {skipped_errors}")
    print(f"✗ Skipped (missing dict): {skipped_missing}")
    
    if error_details:
        print(f"\nError details (showing first 10 of {len(error_details)}):")
        for custom_id, error in error_details[:10]:
            print(f"  • {custom_id}: {error}")
        if len(error_details) > 10:
            print(f"  ... and {len(error_details) - 10} more errors")
    print("="*60 + "\n")



"""
Batch utilities
"""   
def build_batch_jsonl(
    batch_llm,
    occs,
    n_cv_per_occ: int,
    n_jd_per_occ: int,
    n_competence_items_per_cat: int,
    batch_path: str,
    dict_path:str,
    batch_name: str,
    use_multiple_styles: bool = False,
    model: str = "gpt-5-mini-2025-08-07"
    ):
    batch_path = Path(batch_path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    dict_path = Path(dict_path)
    dict_path.parent.mkdir(parents=True, exist_ok=True)

    request_count = 0

    print(f"Clear file contents for {batch_path} first? If not the process will append lines possibly resulting in duplicate ids (yes/no)?")
    answer = input()
    if answer == "yes":
        batch_path.write_text("")
        print("file cleared")

    with dict_path.open("w", encoding="utf-8") as dict_file:
        for occ in occs.itertuples():
            occ_code = occ._1
            occ_title = occ.Title
            print(f"building profiles for '{occ_title}'")

            # Use multiple styles or default "realistic" style
            if use_multiple_styles:
                # Use all 4 styles from STYLE_REQUIREMENTS plus "realistic"
                styles = list(STYLE_REQUIREMENTS.keys()) + ["realistic"]
                profiles_per_style = max(1, (n_cv_per_occ + n_jd_per_occ) // len(styles))
            else:
                styles = ["realistic"]
                profiles_per_style = n_cv_per_occ + n_jd_per_occ
            
            for style in styles:
                profiles = build_competence_profiles(
                    occ_code,
                    n_profiles=profiles_per_style,
                    n_skills=n_competence_items_per_cat,
                    n_knowledge=n_competence_items_per_cat,
                    n_abilities=n_competence_items_per_cat,
                    n_tasks=n_competence_items_per_cat,
                    n_tech_skills=n_competence_items_per_cat,
                    n_work_styles=n_competence_items_per_cat,
                )

                # Split profiles between JD and CV (roughly half each)
                n_jd_for_style = profiles_per_style // 2
                n_cv_for_style = profiles_per_style - n_jd_for_style

                # JD requests
                for i, profile_dict in enumerate(profiles[:n_jd_for_style]):
                    # Use style parameter or default "realistic" prompt
                    if use_multiple_styles:
                        prompt = build_jd_prompt(occ_title, profile_dict, style=style)
                    else:
                        prompt_base = PROMPTS["build_jd_v3"]
                        profile_string = profile_to_string(profile_dict, str_type="jd")
                        prompt = prompt_builder(prompt_base, {"JOB_TITLE": occ_title, "COMPETENCE_ITEMS": profile_string, "STYLE_REQ": "Write the job description in a realistic style"})
                    
                    # Always include style in custom_id for consistency
                    current_style = style if use_multiple_styles else "realistic"
                    jd_id = f"jd::{batch_name}::{occ_code}::{current_style}::{i}"

                    batch_llm.build_request(prompt, str(jd_id), save_path=batch_path)

                    dict_line = {
                        "custom_id": str(jd_id),
                        "dict": profile_dict,
                        "style": current_style,
                        "profile_type": "jd"
                    }

                    dict_file.write(json.dumps(dict_line) + "\n")
                    request_count += 1

                # CV requests
                for i, profile_dict in enumerate(profiles[n_jd_for_style:]):
                    if use_multiple_styles:
                        prompt = build_cv_prompt(occ_title, profile_dict, style=style)
                    else:
                        prompt_base = PROMPTS["build_cv_v3"]
                        profile_string = profile_to_string(profile_dict, str_type="cv")
                        prompt = prompt_builder(prompt_base, {"JOB_TITLE": occ_title, "COMPETENCE_ITEMS": profile_string, "STYLE_REQ": "Write the resume in a realistic style"})
                    
                    # Always include style in custom_id for consistency
                    current_style = style if use_multiple_styles else "realistic"
                    cv_id = f"cv::{batch_name}::{occ_code}::{current_style}::{i}"

                    batch_llm.build_request(prompt, str(cv_id), save_path=batch_path)

                    dict_line = {
                        "custom_id": str(cv_id),
                        "dict": profile_dict,
                        "style": current_style,
                        "profile_type": "cv"
                    }

                    dict_file.write(json.dumps(dict_line) + "\n")
                    request_count += 1

    print(f"Wrote {request_count} requests to {batch_path}")
    return batch_path

# ============================================================================
# DATA COMPILATION
# ============================================================================
# Merge batch results with metadata into final dataset format

def compile_data(batch_name: str, final_batch_llm, final_result_batch, dict_path, final_path):
    """
    Compile final dataset from batch results and competence dictionaries.
    
    Creates structured JSONL with profile_text, profile_dict, and metadata.
    Handles both original and Anthropic-shortened custom_id formats.
    
    Args:
        batch_name: Batch identifier
        final_batch_llm: LLM client for parsing results
        final_result_batch: Path to final step results
        dict_path: Path to competence dictionaries
        final_path: Output path for compiled dataset
    """
    final_results_path = Path(final_result_batch)
    dict_path = Path(dict_path)
    final_path = Path(final_path)

    # Build dict_map with both original and compact formats as keys
    dict_map = {}
    profile_type_map = {}  # Track profile_type for each custom_id
    compact_to_original = {}  # Map compact format to original
    
    with dict_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            item_dict = obj.get("dict")
            profile_type = obj.get("profile_type")  # Get profile_type from dict file
            if cid:
                dict_map[cid] = item_dict
                profile_type_map[cid] = profile_type
                
                # Also create compact version for Anthropic results
                # Format: type::batch::occ::style::idx -> type:hash:occ:abbrev:idx
                # Use consistent style abbreviations to avoid collisions
                STYLE_ABBREV_MAP = {
                    "concise": "cn",
                    "casual": "cs",
                    "detailed": "d",
                    "formal": "f",
                    "realistic": "r"
                }
                parts = cid.split("::")
                if len(parts) == 5:
                    type_id, batch, occ, style, idx = parts
                    style_abbrev = STYLE_ABBREV_MAP.get(style, style[0] if style else "r")
                    if len(batch) > 10:
                        batch_hash = hashlib.md5(batch.encode()).hexdigest()[:8]
                    else:
                        batch_hash = batch
                    compact_cid = f"{type_id}:{batch_hash}:{occ}:{style_abbrev}:{idx}"
                    compact_to_original[compact_cid] = cid
                    dict_map[compact_cid] = item_dict  # Add compact version too
                    profile_type_map[compact_cid] = profile_type  # Add profile_type for compact version

    print("Loaded dict entries:", len(compact_to_original) if compact_to_original else len(dict_map))

    with open(final_results_path) as results, open(final_path, "w") as final:
        for line in results:
            cid, text, error = final_batch_llm.parse_batch_line(json.loads(line))

            # Always convert compact format to original format if available
            if cid in compact_to_original:
                cid = compact_to_original[cid]
            
            it_dict = dict_map.get(cid)
            if it_dict is None:
                print(f"Warning: No dict entry found for custom_id: {cid}")
                continue
            
            # Extract style, occ_code, and profile_type from custom_id (original format: type::batch::occ::style::idx)
            cid_parts = cid.split("::")
            if len(cid_parts) == 5:
                occ_code = cid_parts[2]
                style = cid_parts[3]
                profile_type = cid_parts[0]  # 'jd' or 'cv' from custom_id
            else:
                occ_code = "unknown"
                style = "unknown"
                profile_type = profile_type_map.get(cid, "unknown")  # Fallback to stored value

            # Use stored profile_type if available, otherwise extract from custom_id
            profile_type = profile_type_map.get(cid, profile_type)

            item = {
                "custom_id": cid, 
                "occ_code": occ_code, 
                "batch_name": batch_name, 
                "style": style,
                "profile_type": profile_type,
                "profile_text": text, 
                "profile_dict": it_dict
            }

            final.write(json.dumps(item) + "\n")

def run_n_steps(
    batch_llms: list,   # list of llms to be used in each step batch llms[0] is used to read the first batch and generate the first input. It is also used to read the resulst of this step. Batch llm[1] is used to fomrat the new requests
    batch_name:str,
    first_batch_jsonl: str,
    dict_path: str,
    steps: int = 2,
    start_step: int = 1,
    batch_completion_window = "24h"
    ):

    batch_in = Path(first_batch_jsonl)
    batch_dir = Path(f"output/batch_jsonl/{batch_name}")
    log_path = batch_dir/ f"batch_{batch_name}_log.txt"

    print(f"running for batch {batch_name} stating at step {start_step}")


    for step in range(start_step, steps + 1):

        prev_results_path = batch_dir/f"batch_{batch_name}_step{step-1}_results.jsonl" 
        results_path = batch_dir / f"batch_{batch_name}_step{step}_results.jsonl" 
        input_path = batch_dir / f"batch_{batch_name}_step{step}_input.jsonl" 

        batch_llm_results = batch_llms[step-2]
        batch_llm_input= batch_llms[step-1]
        
      
        if start_step == 1 and step == 1:
            print(f"Copying first batch from {first_batch_jsonl}? batch will be saved at {input_path}. Continue? (yes/no)")
            answer = input()
            if answer != "yes":
                break
            
            copy_jsonl(first_batch_jsonl, input_path)


        else:
            print(f"Reading results from directly from: {prev_results_path}, and building new batch form it at {input_path} continue? (yes/no)")
            answer = input()
            if answer != "yes":
                break

            build_new_batch_from_results(
                batch_llm_results=batch_llm_results,
                batch_llm_input = batch_llm_input,
                results_file=str(prev_results_path),
                dict_file=dict_path,
                next_batch_file=str(input_path),

            )

        print(f"Ready to submit from {input_path}? (yes/no)")
        answer = input()
        if answer != "yes":
            break

        # 2) Submit batch
        batch_id = batch_llm_input.submit(input_path)
        print("Submitted:", batch_id)

        # 3) Wait + download results
        result = batch_llm_input.download_results(batch_id, wait=True, timeout_s=3600)

        if result.status == "completed" or result.status == "succeeded":
            # Write outputs
            batch_llm_input.write_jsonl(result.output_lines,  results_path)

            # Optional: write request-level errors too (some batches complete with partial failures)
            if result.error_lines:
                batch_llm_input.write_jsonl(result.error_lines, log_path)

            print("✅ Completed. Output lines:", len(result.output_lines))
            
            # Analyze failures
            print("\n" + "="*60)
            print("ANALYZING BATCH RESULTS")
            print("="*60)
            failures = analyze_batch_failures(results_path, batch_llm_input, input_file=input_path)
            
            total_failures = sum(len(v) for k, v in failures.items() if k != "success")
            print(f"✓ Successful: {failures['success']}")
            print(f"✗ Failed: {total_failures}")
            
            if total_failures > 0:
                print("\nFailure breakdown:")
                if failures["missing"]:
                    print(f"  - Missing results (no response): {len(failures['missing'])}")
                    for cid, err, _ in failures["missing"][:3]:  # Show first 3
                        print(f"    • {cid}")
                    if len(failures["missing"]) > 3:
                        print(f"    ... and {len(failures['missing']) - 3} more")
                
                if failures["token_limit"]:
                    print(f"  - Token limit exceeded: {len(failures['token_limit'])}")
                    for cid, err, _ in failures["token_limit"][:3]:  # Show first 3
                        print(f"    • {cid}: {err}")
                    if len(failures["token_limit"]) > 3:
                        print(f"    ... and {len(failures['token_limit']) - 3} more")
                
                if failures["content_filter"]:
                    print(f"  - Content filter: {len(failures['content_filter'])}")
                    for cid, err, _ in failures["content_filter"][:3]:
                        print(f"    • {cid}: {err}")
                    if len(failures["content_filter"]) > 3:
                        print(f"    ... and {len(failures['content_filter']) - 3} more")
                
                if failures["http_error"]:
                    print(f"  - HTTP errors: {len(failures['http_error'])}")
                    for cid, err, _ in failures["http_error"][:3]:
                        print(f"    • {cid}: {err}")
                
                if failures["other_error"]:
                    print(f"  - Other errors: {len(failures['other_error'])}")
                    for cid, err, _ in failures["other_error"][:3]:
                        print(f"    • {cid}: {err}")
                
                # Ask if user wants to retry
                print("\n" + "="*60)
                print(f"Do you want to retry the {total_failures} failed requests? (yes/no)")
                retry_answer = input()
                
                if retry_answer == "yes":
                    # Collect all failed items
                    all_failed = (
                        failures["missing"] +
                        failures["token_limit"] + 
                        failures["content_filter"] + 
                        failures["http_error"] + 
                        failures["other_error"]
                    )
                    
                    retry_batch_path = batch_dir / f"batch_{batch_name}_step{step}_retry_input.jsonl"
                    retry_results_path = batch_dir / f"batch_{batch_name}_step{step}_retry_results.jsonl"
                    
                    # Create retry batch
                    retry_batch_path = retry_failed_requests(all_failed, Path(dict_path), batch_llm_input, retry_batch_path, input_file=input_path)
                    
                    print(f"\nSubmitting retry batch with {len(all_failed)} requests...")
                    retry_batch_id = batch_llm_input.submit(retry_batch_path)
                    print(f"Retry batch submitted: {retry_batch_id}")
                    
                    # Wait for retry results
                    retry_result = batch_llm_input.download_results(retry_batch_id, wait=True, timeout_s=3600)
                    
                    if retry_result.status == "completed" or retry_result.status == "succeeded":
                        batch_llm_input.write_jsonl(retry_result.output_lines, retry_results_path)
                        print(f"✅ Retry completed: {len(retry_result.output_lines)} results")
                        
                        # Merge retry results back into main results
                        temp_results = batch_dir / f"batch_{batch_name}_step{step}_results_temp.jsonl"
                        merge_retry_results(results_path, retry_results_path, temp_results)
                        
                        # Replace original with merged
                        temp_results.replace(results_path)
                        print(f"✅ Merged retry results into {results_path}")
                        
                        # Re-analyze after retry
                        print("\nRe-analyzing after retry...")
                        failures_after = analyze_batch_failures(results_path, batch_llm_input)
                        total_failures_after = sum(len(v) for k, v in failures_after.items() if k != "success")
                        print(f"✓ Successful: {failures_after['success']}")
                        print(f"✗ Still failed: {total_failures_after}")
                    else:
                        print(f"❌ Retry batch failed")
            
            print("="*60 + "\n")
            
        else:
            # One log file with batch-level + request-level errors + raw batch JSON
            batch_llm_input.debug_first_error(batch_id)
            batch_llm_input.log_batch(batch_id, log_path)


DATA_OCCUPATIONS = read_data_occupations()
SKILLS = read_skills()
ABILITIES = read_abilities()
KNOWLEDGE = read_knowledge()
EXPERIENCE, EXPERIENCE_CAT = read_experience()
TECH_SKILLS = read_tech_skills()
WORK_STYLES = read_work_styles()
TASKS, TASKS_CAT = read_tasks()
CONTENT_REFERENCE = read_content_reference()


def build_data_main(n_it: int, steps: int, batch_name: str, start_step: Union[int, str] = 1, n_major = 20, n_detailed=5, reasoning_effort: str = "medium", use_multiple_styles: bool = False):
    """
    Main entry point for dataset generation pipeline.
    
    Coordinates the full workflow:
    1. Sample O*NET occupations
    2. Generate initial batch requests
    3. Execute multi-step processing
    4. Compile and save final dataset
    
    Special mode: start_step='complete_only' regenerates final output without API calls.
    
    Args:
        n_it: Competence items to sample per category
        steps: Total pipeline steps (default 3)
        batch_name: Unique identifier for this batch
        start_step: Step to start from (1-3) or 'complete_only'
        n_major: Number of major occupation groups
        n_detailed: Detailed occupations per major group
        reasoning_effort: LLM reasoning effort (low/medium/high)
        use_multiple_styles: Generate 4 style variations
    """

    batch_dir = Path(f"output/batch_jsonl/{batch_name}")
    dict_path = batch_dir / f"batch_{batch_name}_dict.jsonl"
    final_results_path = batch_dir / f"batch_{batch_name}_step{steps}_results.jsonl"
    final_jsonl_path = batch_dir / f"final_{batch_name}.jsonl"
    
    # Special mode: only compile final data from existing results
    if start_step == "complete_only":
        print("\n" + "="*60)
        print("COMPLETE_ONLY MODE: Regenerating final JSONL from existing results")
        print("="*60)
        
        if not final_results_path.exists():
            raise FileNotFoundError(f"Results file not found: {final_results_path}")
        if not dict_path.exists():
            raise FileNotFoundError(f"Dict file not found: {dict_path}")
        
        # Initialize only the final batch LLM for parsing
        anth_batch_llm_val = AnthropicBatchLLM(model="claude-sonnet-4-5-20250929", max_tokens=1024, temperature=0)
        
        # Compile the data
        compile_data(batch_name, anth_batch_llm_val, final_results_path, dict_path, final_jsonl_path)
        
        # Copy to final_data folder
        final_data_dir = Path("data/final_data")
        final_data_dir.mkdir(parents=True, exist_ok=True)
        destination = final_data_dir / f"final_{batch_name}.jsonl"
        shutil.copy2(final_jsonl_path, destination)
        
        print(f"\n✓ Final JSONL regenerated: {final_jsonl_path}")
        print(f"✓ Copied to: {destination}")
        print("="*60 + "\n")
        return
    
    # Normal processing mode
    occs = top_majors_with_top_detailed_report(DATA_OCCUPATIONS, n_major=n_major, n_detailed=n_detailed)

    print(occs)

    first_batch_jsonl = batch_dir/"first_batch.jsonl"

    # Step 1: generation with user-specified reasoning effort
    # Steps 2-3: validation with low reasoning effort
    gpt_batch_llm_gen = BatchLLM(model="gpt-5-mini-2025-08-07", reasoning_effort=reasoning_effort)
    gpt_batch_llm_val = BatchLLM(model="gpt-5-mini-2025-08-07", reasoning_effort="low")
    anth_batch_llm_val = AnthropicBatchLLM(model="claude-sonnet-4-5-20250929", max_tokens=1024, temperature=0)

    batch_llms = [gpt_batch_llm_gen, gpt_batch_llm_val, anth_batch_llm_val]

    batch_llms[2].list_models()

    if len(batch_llms) != steps:
        raise ValueError(f"The number of batch llms ({len(batch_llms)}) should equal the number of steps ({steps})")

    if start_step == 1: 
        build_batch_jsonl(batch_llms[0], occs, n_cv_per_occ=5, n_jd_per_occ=5, n_competence_items_per_cat=n_it, batch_path=first_batch_jsonl, dict_path=dict_path, batch_name=batch_name, use_multiple_styles=use_multiple_styles)

    
    run_n_steps(batch_llms, batch_name, first_batch_jsonl= first_batch_jsonl ,dict_path=dict_path, steps=steps, start_step=start_step)

    compile_data(batch_name, batch_llms[-1], final_results_path, dict_path, final_jsonl_path)
    
    # Copy to final_data folder
    final_data_dir = Path("data/final_data")
    final_data_dir.mkdir(parents=True, exist_ok=True)
    destination = final_data_dir / f"final_{batch_name}.jsonl"
    shutil.copy2(final_jsonl_path, destination)
    print(f"\n✓ Final data copied to: {destination}\n")