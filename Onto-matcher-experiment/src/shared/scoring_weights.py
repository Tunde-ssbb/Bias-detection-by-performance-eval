"""
Central scoring weight definitions used by all conditions and ground-truth scoring.

Edit IMPORTANCE_WEIGHTS here to change how much each importance level contributes
across conditions B, C, D and the ground-truth evaluator.

The quadratic scheme (level² where level ∈ {1,2,3,4,5}) means a requirement
labelled "extremely important" (weight 25) counts 25× more than one labelled
"not important" (weight 1).
"""

# Ordered from highest to lowest importance (used to build prompts).
IMPORTANCE_LEVELS = [
    "extremely important",
    "very important",
    "important",
    "somewhat important",
    "not important",
]

# descriptions for importance levels to label jd requirements
IMPORTANCE_CUES = {
    "extremely important": "central, essential, critical, required, mandatory",
    "very important":      "strongly preferred, highly valued, very important, highly significant",
    "important":           "important, notable, meaningful",
    "somewhat important":  "beneficial, a plus, preferred, nice to have",
    "not important":       "optional, secondary, not required, not important",
}

# Linear weight multipliers: levels 1–5.
IMPORTANCE_WEIGHTS = {
    "not important":       1,
    "somewhat important":  2,
    "important":           3,
    "very important":      4,
    "extremely important": 5,
}

# Fixed importance assigned to education and experience requirements.
# Set to the maximum O*NET level (5) so weight = 
# equal to an "extremely important" labelled node.
ED_EXP_IMPORTANCE: float = 5.0
