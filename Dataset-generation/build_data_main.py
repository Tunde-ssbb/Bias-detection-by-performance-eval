"""
Dataset Generation Pipeline - CLI Entry Point

Command-line interface for generating synthetic CV and job description datasets
using O*NET occupational data and LLM batch processing.

Usage:
    python build_data_main.py --batch-name test --n-items 5 --n-major 10
"""

from src.clean_build_dataset import build_data_main
import argparse


def parse_args():
    """Parse and validate command-line arguments for the dataset generation pipeline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=1, help="number of items to sample per category")
    ap.add_argument("--batch-name", required=True, help="name of the batch")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--start-step", default=1, help="step to start from (1, 2, 3, or 'complete_only' to only regenerate final JSONL)")
    ap.add_argument("--n-major", type=int, default = 20)
    ap.add_argument("--n-detailed", type=int, default = 5)
    ap.add_argument("--reasoning-effort", type=str, default="medium", choices=["low", "medium", "high"], help="reasoning effort for generation (validation steps use 'low')")
    ap.add_argument("--use-multiple-styles", action="store_true", help="use multiple writing styles (concise, detailed, formal, casual) instead of default realistic style")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    batch_name = args.batch_name
    n_items = args.n_items
    
    # Handle start_step: convert to int if numeric, keep as string if "complete_only"
    start_step = args.start_step
    if start_step != "complete_only":
        try:
            start_step = int(start_step)
        except ValueError:
            raise ValueError(f"--start-step must be an integer (1, 2, 3) or 'complete_only', got: {start_step}")
    
    build_data_main(n_it=n_items, steps=args.steps, n_major=args.n_major, n_detailed=args.n_detailed, start_step=start_step, batch_name=batch_name, reasoning_effort=args.reasoning_effort, use_multiple_styles=args.use_multiple_styles)