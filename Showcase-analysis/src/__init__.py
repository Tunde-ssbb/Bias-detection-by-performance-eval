"""
Experiment Analysis Package
"""

from .parsing import parse_experiments, load_parsed
from .post_processing import enrich
from .metrics import ThresholdSpec, apply_threshold, compute_metrics, save_results, flatten_results
from .plots import plot_score_vs_gt, plot_score_distribution, plot_score_gt_overlap
from .latex import make_results_table
from .ground_truth_scoring import compute_ground_truth_score, normalize_score
from .explainability import explain, save_explain_results, load_explain_results, plot_explain_results

__version__ = "0.2.0"
