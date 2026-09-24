"""
Utility Functions

Helper functions for file I/O, data processing, and performance measurement.
Provides JSONL handling, type conversion, and timing decorators.
"""

from __future__ import annotations

from typing import Iterable, Dict, List, Literal, Callable, Any
from pathlib import Path
from dataclasses import asdict
import json
import time
import re
from functools import wraps
from termcolor import colored

# ============================================================================
# FILE I/O OPERATIONS
# ============================================================================

def save_jsonl(items: Iterable[Any], path: Path) -> None:
    """
    Save an iterable of items to a JSONL file (one JSON object per line).
    
    Automatically converts dataclasses to dicts. Creates parent directories if needed.
    
    Args:
        items: Iterable of dicts, dataclasses, or JSON-serializable objects
        path: Output file path
    
    Returns:
        String path to the created file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            # Dataclasses -> dict; dicts pass through.
            if hasattr(x, "__dataclass_fields__"):
                payload = asdict(x)
            elif isinstance(x, dict):
                payload = x
            else:
                payload = x  # best-effort
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Skips empty lines. Returns empty list if file doesn't exist.
    
    Args:
        path: Path to JSONL file
    
    Returns:
        List of parsed JSON objects as dicts
    """
    path = Path(path)
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out

# ============================================================================
# TYPE CONVERSION
# ============================================================================

def safe_float(x: Any, default: float = 0.0) -> float:
    """
    Convert value to float, returning default on failure.
    
    Args:
        x: Value to convert
        default: Value to return if conversion fails
    
    Returns:
        Float value or default
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def extract_json_from_str(text: str):
    """
    Extract and parse the first JSON object found in a string.
    
    Useful for parsing JSON from LLM responses that may include extra text.
    
    Args:
        text: String potentially containing JSON
    
    Returns:
        Parsed JSON object
    
    Raises:
        ValueError: If no JSON object is found
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))

# ============================================================================
# PERFORMANCE MEASUREMENT
# ============================================================================

def timeit(fn: Callable) -> Callable:
    """
    Decorator that measures and prints execution time of a function or method.
    
    Displays colored output with function name and elapsed time in seconds.
    
    Usage:
        @timeit
        def my_function():
            # ... code ...
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(colored(f"{fn.__qualname__} took {elapsed:.4f}s", 'cyan'))
    return wrapper

