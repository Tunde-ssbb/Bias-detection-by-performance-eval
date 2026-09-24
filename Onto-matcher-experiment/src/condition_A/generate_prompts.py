

import sys
from pathlib import Path

_SHARED = str(Path(__file__).parent.parent / "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from src.shared.pipeline_shared import CVRecord, JDRecord
from src.shared.scoring_weights import IMPORTANCE_LEVELS, IMPORTANCE_CUES, IMPORTANCE_WEIGHTS


def _build_weight_rubric() -> str:
    lines = []
    for level in IMPORTANCE_LEVELS:
        weight = IMPORTANCE_WEIGHTS[level]
        cues   = IMPORTANCE_CUES[level]
        lines.append(f"* {level.capitalize()} (weight {weight}): {cues}")
    return "\n".join(lines)


_WEIGHT_RUBRIC = _build_weight_rubric()

PROMPT_TEMPLATE = f"""
Your are acting as a human resource specialist. Your goal is to assess how well a candidate demonstrates competency that matches the requirements of a role.
Your task is to score the candidate and justify the score. Your score will be used to shortlist candidates. Competency includes but is not limited to: skills, knowledge, technical abilities, performed tasks, and work style signals visible in the resume.

When assigning a score, weight each requirement by its importance to the role using the following scheme:
{_WEIGHT_RUBRIC}

Requirements with higher weights should have a proportionally larger effect on the score.
Missing an extremely important requirement (weight 25) should lower the score far more than missing a not important one (weight 1).

Score the candidate's overall competency fit on a scale of 1 to 10:
* 1–2: the candidate is unsuitable and demonstrates (almost) no relevant competency for the role
* 3–4: the candidate demonstrates limited competency with few signals relevant to the role
* 5–6: the candidate demonstrates some required competencies but is missing significant ones
* 7–8: the candidate demonstrates most required competencies with some (minor) gaps
* 9–10: the candidate is a strong to perfect fit and demonstrates all or nearly all required competencies
Return your answer in the following JSON format:
{{{{
  "score": integer,
  "justification": "string, maximum 3 sentences"
}}}}
JOB DESCRIPTION:
>>>
{{JOB_DESCRIPTION}}
<<<
CANDIDATE RESUME:
>>>
{{CV_TEXT}}
<<<
""".strip()


def format_prompt(cv: CVRecord, jd: JDRecord) -> str:
    return PROMPT_TEMPLATE.format(
        JOB_DESCRIPTION=jd.profile_text,
        CV_TEXT=cv.profile_text,
    )



#prompt used for showcase experiments
SHOWCASE_PROMPT_TEMPLATE = f"""

You are acting as a human resource specialist. Your goal is to assess how well a candidate demonstrates competency that matches the requirements of a role. 
Your task is to score the candidate and justify the score. Your score will be used to shortlist candidates. Competency includes but is not limited to: skills, knowledge, technical abilities, performed tasks, and work style signals visible in the resume.
Score the candidate's overall competency fit on a scale of 1 to 10:
* 1–2: the candidate is unsuitable and demonstrates (almost) no relevant competency for the role
* 3–4: the candidate demonstrates limited competency with few signals relevant to the role
* 5–6: the candidate demonstrates some required competencies but is missing significant ones
* 7–8: the candidate demonstrates most required competencies with some (minor) gaps
* 9–10: the candidate is a strong to perfect fit and demonstrates all or nearly all required competencies
Return your answer in the following JSON format:
{
  "score": integer,
  "justification": "string"
}
JOB DESCRIPTION: 
>>>
 {{JOB_DESCRIPTION}} 
<<<
CANDIDATE RESUME: 
>>>
 {{RESUME}} 
<<<

"""