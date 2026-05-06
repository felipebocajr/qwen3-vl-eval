# Qwen3-VL MMMU Evaluation Pipeline

Evaluation pipeline for running **Qwen3-VL-2B-Instruct** locally on 100 samples from the **MMMU** (Massive Multi-discipline Multimodal Understanding) dataset.

## Features

- **100-sample cap**: The pipeline evaluates at most 100 unique MMMU samples per run.
- **Stratified sampling**: Samples are selected proportionally across subjects (e.g., ~25 from each of 4 subjects).
- **Resumable**: If interrupted, restarting skips already-completed samples and continues up to the cap.
- **Deterministic**: The same 100 stratified samples are selected on every fresh start, ensuring reproducible results.
- **Graceful failures**: Individual sample errors are logged without halting the pipeline.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Install dependencies with uv
uv sync
```

The pipeline uses **stratified sampling** to ensure balanced representation across MMMU subjects. For example, with 4 subjects and a 100-sample cap, the pipeline selects approximately 25 samples from each subject deterministically. Any shortfall in a subject (if it has fewer samples than its quota) is not backfilled from other subjects, so the total may be slightly below 100 in edge cases.

## Usage

```bash
# Full evaluation (up to 100 samples)
python -m src.main

# Results are written to:
#   results/trajectories.jsonl   — one JSON record per sample
#   results/summary.json          — aggregate metrics
```

### Resume a partial run

Simply rerun `python -m src.main`. The pipeline reads `results/trajectories.jsonl`, skips completed sample IDs, and evaluates the remaining samples up to the 100-sample cap.

## Project Structure

```bash
# Verify deterministic & stratified selection
python scripts/verify_determinism.py
```

- `src/config.py` — Cap value, paths, subject list, and quota helper
- `src/data.py` — MMMU dataset loading and **stratified** deterministic sample selection
- `src/model.py` — Qwen3-VL model loading and inference
- `src/parser.py` — Answer extraction from raw model responses
- `src/pipeline.py` — Per-sample evaluation and trajectory persistence
- `src/metrics.py` — Accuracy, per-subject stats, and summary generation
- `src/main.py` — Orchestration loop with resume logic

## Determinism Verification

```bash
python scripts/verify_determinism.py
```

The pipeline uses **stratified sampling** to ensure balanced representation across MMMU subjects. For example, with 4 subjects and a 100-sample cap, the pipeline selects approximately 25 samples from each subject deterministically. Any shortfall in a subject (if it has fewer samples than its quota) is not backfilled from other subjects, so the total may be slightly below 100 in edge cases.

## License

MIT
