# Implementation Plan: Limit Pipeline to 100 Samples

## Tech Stack
- Python 3.12
- datasets (HuggingFace)
- uv for dependency management

## Project Structure
```
src/
  config.py        # Configuration constants and quota helpers
  data.py          # MMMU dataset loading, stratified sampling
  model.py          # Qwen3-VL model setup & inference
  parser.py         # Answer extraction from raw model response
  pipeline.py       # Main loop with resume logic
  metrics.py        # Accuracy, per-subject stats
main.py            # Entry point
```

## Key Implementation Notes
- Stratified sampling across MMMU subjects with proportional quota allocation
- Resume state tracked in `results/trajectories.jsonl`
- Deterministic dataset selection via fixed alphabetical ordering and index-based slicing
