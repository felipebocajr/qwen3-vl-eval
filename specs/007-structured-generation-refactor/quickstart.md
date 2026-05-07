# Quickstart: Structured Generation Refactor

**Branch**: `006-structured-output-mandate`  
**Purpose**: One-page guide to running the evaluation pipeline after the structured generation refactor.

---

## Prerequisites

- Python 3.12+
- `uv` installed
- CUDA-capable GPU (recommended) or CPU with ≥ 16 GB RAM
- Internet access for initial model download (cached afterward)

---

## Setup

```bash
# 1. Sync dependencies (outlines, pydantic, and lm-format-enforcer will be installed)
uv sync

# 2. Verify environment
python -c "import torch, transformers, outlines, pydantic; print('OK')"
```

---

## Run the Pipeline

```bash
# Full pipeline (100 samples, resumable, structured generation)
python -m src.main
```

The pipeline now:
1. Loads `Qwen/Qwen3-VL-2B-Instruct` with Outlines structured generation wrapper.
2. Samples 100 MMMU validation questions across 4 subjects.
3. Runs **schema-constrained generation** — every response is valid JSON matching `{"reasoning": "...", "answer": "A|B|C|D"}`.
4. Validates answers using Pydantic (no regex parsing).
5. Writes per-sample records to `results/trajectories.jsonl`.
6. Writes aggregate summary to `results/summary.json`.

**To resume** after interruption:
```bash
python -m src.main
```

---

## Verify Structured Generation is Active

After a single sample completes, check the raw response:

```bash
head -n 1 results/trajectories.jsonl | python -m json.tool | grep -E "raw_model_response|extracted_answer|extraction_succeeded"
```

Expected:
- `raw_model_response` contains JSON with `reasoning` and `answer` fields.
- `extracted_answer` is a single letter (A/B/C/D).
- `extraction_succeeded` is `true`.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'outlines'` | Dependencies not synced | Run `uv sync` |
| `OutOfMemoryError` during inference | Reasoning chain + JSON too long for GPU | Lower `MAX_NEW_TOKENS` temporarily or use CPU; pipeline catches and continues |
| Structured output fails / invalid JSON | Outlines logits processor incompatible | Verify `outlines>=1.2.13` and `transformers>=4.40.0` |
| Parser returns `(None, False)` for all samples | Model not producing valid JSON | Check prompt in `src/pipeline.py` includes schema instruction |
| `RuntimeError: Expected all tensors to be on the same device` | Multi-device placement conflict | Verify `device_map="auto"` is used |

---

## Validation Checklist

After running the pipeline, confirm:

- [ ] Model downloaded from `Qwen/Qwen3-VL-2B-Instruct`.
- [ ] `src/parser.py` has zero `re.search` / `re.match` / `re.findall` / `re.compile` calls.
- [ ] `src/model.py` uses Outlines `Generator` with `output_type=ModelResponse` (not `model.generate()`).
- [ ] Raw responses contain `reasoning` and `answer` fields (CoT preserved).
- [ ] `MAX_NEW_TOKENS` is `2048` in `src/config.py`.
- [ ] Resumability works: interrupt at sample 20, rerun, verify it starts at 21.
- [ ] `summary.json` contains expected accuracy metrics.
