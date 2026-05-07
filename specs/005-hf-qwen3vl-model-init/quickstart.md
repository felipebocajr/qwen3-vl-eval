# Quickstart: HF Qwen3-VL Model Initialization with Preserved Structured Generation

**Branch**: `005-hf-qwen3vl-model-init`  
**Purpose**: One-page guide to running the evaluation pipeline after applying the initialization-layer change.

---

## Prerequisites

- Python 3.12+
- `uv` installed (dependency management)
- CUDA-capable GPU (recommended) or CPU with ≥ 16 GB RAM
- HuggingFace Hub token cached if model is gated (not required for `Qwen/Qwen3-VL-2B-Instruct`)
- Internet access for initial model download (cached afterward)

---

## Setup

```bash
# 1. Sync dependencies
uv sync

# 2. Verify environment
python -c "import torch, transformers, outlines; print('OK')"
```

---

## Run the Pipeline

```bash
# Full pipeline (100 samples, resumable)
python -m src.main
```

The pipeline automatically:
1. Loads `Qwen/Qwen3-VL-2B-Instruct` from HuggingFace Hub (or local cache).
2. Samples 100 MMMU validation questions across 4 subjects.
3. Runs structured generation via Outlines + Pydantic `ModelResponse`.
4. Writes per-sample records to `results/trajectories.jsonl`.
5. Writes aggregate summary to `results/summary.json`.

**To resume** after interruption:
```bash
python -m src.main
```
The pipeline reads `results/trajectories.jsonl`, skips completed IDs, and continues.

---

## Verify Structured Generation is Active

After a single sample completes, check the raw response in `results/trajectories.jsonl`:

```bash
head -n 1 results/trajectories.jsonl | python -m json.tool | grep -E "raw_model_response|extracted_answer|extraction_succeeded"
```

Expected:
- `raw_model_response` contains JSON-like text (not plain prose).
- `extracted_answer` is a single letter (A/B/C/D).
- `extraction_succeeded` is `true`.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: Expected all tensors to be on the same device` | Multi-device placement conflict | Verify `device_map="auto"` is used; check `accelerate` version |
| `OutOfMemoryError` during model load | GPU too small for bfloat16 | Reduce to `torch_dtype=torch.float16` or force CPU: `DEVICE="cpu"` |
| `OutOfMemoryError` during inference | `max_new_tokens=2048` on long prompts | This is expected occasionally; the pipeline catches and logs, then continues |
| `KeyError: 'Qwen3VLForConditionalGeneration'` | `transformers` version too old | Run `uv sync` to update dependencies |
| Structured output fails / invalid JSON | Outlines logits processor incompatible with model output | Verify `outlines>=1.2.13` and `transformers>=4.40.0`; check single-sample test first |

---

## Validation Checklist

After running the pipeline, confirm:

- [ ] Model downloaded from `Qwen/Qwen3-VL-2B-Instruct` (check cache or logs).
- [ ] No `re.search` / `re.match` / regex parsing introduced in `src/parser.py`.
- [ ] `src/model.py` does not accept an arbitrary `model_id` override.
- [ ] Raw responses contain `reasoning` and `answer` fields (CoT preserved).
- [ ] `MAX_NEW_TOKENS` is not reduced from `2048` (check `src/config.py`).
- [ ] Resumability works: interrupt at sample 20, rerun, verify it starts at 21.
- [ ] `summary.json` contains expected accuracy metrics.
