# O*NET Dataset Generation Pipeline

A comprehensive system for generating synthetic CV and job description datasets using O*NET occupational data, with LLM-based generation, validation, and annotation capabilities.

## Overview

This pipeline generates realistic CVs and job descriptions by:
1. Sampling O*NET occupational profiles
2. Generating synthetic documents via LLM batch APIs (OpenAI & Anthropic)
3. Validating generated content through multi-step checking

## Project Structure

```
Dataset-generation/
├── build_data_main.py              # Main CLI entry point
├── data/
│   ├── Onet_text_files/           # O*NET source data not included in repo
│   └── final_data/                # Output: final processed datasets
├── src/
│   ├── clean_build_dataset.py     # Core pipeline logic
│   ├── llm/
│   │   ├── llm_client.py          # Batch API clients (OpenAI & Anthropic)
│   │   └── prompts.py             # Generation & validation prompts
│   └── util/
│       └── util.py                # Helper functions


```

## Quick Start

### Installation


### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Basic Usage

**Generate new dataset:**
```bash
python build_data_main.py \
  --batch-name my_batch \
  --n-items 5 \
  --n-major 10 \
  --n-detailed 3 \
  --reasoning-effort medium
```

**Resume from a specific step:**
```bash
python build_data_main.py --batch-name my_batch --start-step 2
```

**Regenerate final output only (no API calls):**
```bash
python build_data_main.py --batch-name my_batch --start-step complete_only
```

**Use multiple writing styles:**
```bash
python build_data_main.py \
  --batch-name styled_batch \
  --n-items 3 \
  --use-multiple-styles
```

## Pipeline Stages

### Stage 1: Generation (OpenAI gpt-5-mini)
- Samples O*NET occupations (major + detailed)
- Generates CVs and job descriptions with configurable reasoning effort
- Creates custom_id tracking: `{type}::{batch_name}::{occ_code}::{style}::{index}`
- Output: `batch_{name}_step1_results.jsonl`

### Stage 2: Validation (OpenAI gpt-5-mini, low effort)
- Checks generated text against required competence items
- Removes PII and placeholders
- Ensures all competencies are mentioned appropriately
- Output: `batch_{name}_step2_results.jsonl`

### Stage 3: Final Validation (Anthropic Claude Sonnet)
- Secondary validation with different model
- Final quality control
- Output: `batch_{name}_step3_results.jsonl`

### Final Compilation
- Merges results with source metadata
- Creates structured JSONL with profile_text and profile_dict
- Copies to `data/final_data/` for easy access

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-name` | str | *required* | Unique identifier for the batch |
| `--n-items` | int | 1 | Competence items to sample per category |
| `--n-major` | int | 20 | Number of major occupation groups |
| `--n-detailed` | int | 5 | Detailed occupations per major group |
| `--steps` | int | 3 | Total pipeline steps |
| `--start-step` | int/str | 1 | Start from step (1-3) or 'complete_only' |
| `--reasoning-effort` | str | "medium" | LLM reasoning effort: low/medium/high |
| `--use-multiple-styles` | flag | False | Generate 4 style variations per profile |


## Output Format

**Final JSONL structure:**
```json
{
  "custom_id": "cv::batch_name::29-1141.00::realistic::0",
  "occ_code": "29-1141.00",
  "batch_name": "my_batch",
  "style": "realistic",
  "profile_text": "Bachelor's Degree holder with...",
  "profile_dict": {
    "skill_0": {"kind": "skill", "name": "Active Listening", ...},
    "knowledge_0": {"kind": "knowledge", "name": "Medicine", ...},
    ...
  }
}
```

## Data Sources

O*NET database files from  https://www.onetcenter.org/database.html#individual-files Used under the CC BY 4.0 license. O*NET® is a trademark of USDOL/ETA

## API Models

- **Generation**: `gpt-5-mini-2025-08-07` with configurable reasoning effort
- **Validation**: `gpt-5-mini-2025-08-07` with low reasoning effort
- **Final Check**: `claude-sonnet-4-5-20250929` with temperature=0

