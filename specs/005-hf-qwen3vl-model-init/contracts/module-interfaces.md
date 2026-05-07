# Module Interface Contract: src/model.py ↔ src/pipeline.py

**Feature**: HF Qwen3-VL Model Initialization with Preserved Structured Generation  
**Date**: 2026-05-06  
**Status**: Stable (unchanged signatures)

---

## Overview

`src/model.py` exports two functions consumed by `src/pipeline.py`. This contract guarantees that the initialization-layer changes in `src/model.py` do not alter the interface surface area or observable behavior of these functions.

---

## Function: `load_model_and_processor`

### Signature (Python)

```python
def load_model_and_processor() -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    ...
```

**Note**: The parameter `model_id: str = MODEL_ID` is **removed**. Callers MUST NOT pass arguments. The function internally hardcodes the canonical identifier `Qwen/Qwen3-VL-2B-Instruct`.

### Pre-conditions

- Network connectivity to HuggingFace Hub OR model weights cached locally in `~/.cache/huggingface/hub`.
- Sufficient VRAM (≥ 6 GB recommended for 2B bfloat16) or CPU memory.
- `transformers>=4.40.0` and `torch>=2.2.0` installed.

### Post-conditions

- Returns a tuple `(model, processor)` where `model` is in `.eval()` mode.
- Returned `processor` is the `AutoProcessor` from the same checkpoint.
- Model weights are loaded onto device map (CUDA if available, CPU otherwise).
- No side effects on global state other than module-level `_generator_cache`.

### Error Contract

- **Failure mode**: On missing network, OOM, or corrupted weights, raises `RuntimeError` or `OSError`.
- **Handling**: Caller (`src/pipeline.py`) is expected to catch and log; this function does not swallow exceptions.

---

## Function: `run_inference`

### Signature (Python)

```python
def run_inference(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int = 2048,
) -> str:
    ...
```

### Parameters

| Name | Type | Description | Invariant |
|------|------|-------------|-----------|
| model | Qwen3VLForConditionalGeneration | Loaded model instance from `load_model_and_processor()` | Must be the same instance paired with the passed processor |
| processor | AutoProcessor | Processor paired with `model` | Must match the model's checkpoint |
| images | list[PIL.Image.Image] | List of RGB-converted PIL images (0–7 items) | Each image must be a valid PIL Image object |
| prompt | str | Text prompt with options and JSON schema instruction | Non-empty string |
| max_new_tokens | int | Maximum decoding steps | Must be ≥ 1; default 2048 (UNCHANGED per spec) |

### Return Value

- **Type**: `str`
- **Content**: Raw response string produced by Outlines `Generator`. Because `Generator` is configured with `output_type=ModelResponse`, the returned string is guaranteed to be JSON-compatible text that, when parsed and validated with `ModelResponse.model_validate()`, produces a valid `ModelResponse` instance.
- **Behavior**: The function applies a system message instructing JSON-only output and user message containing prompt + images. Logits are constrained by Outlines at every decoding step.

### Invariants

- The structured generation pipeline (Outlines + Pydantic) is the **only** generation path. No unconstrained `.generate()` or `.chat()` calls are made.
- The system message, user message construction, and image attachment pattern are preserved exactly.
- `max_new_tokens` default remains 2048; CoT reasoning is preserved by not lowering this budget.

### Error Contract

- On Outlines internal failure, raises an exception propagated to caller.
- On OOM during generation, `torch.cuda.OutOfMemoryError` is raised and propagated.
- Caller (`src/pipeline.py`) wraps inference in try/except and logs; `run_inference` does not swallow.

---

## Change Boundary

- `src/pipeline.py` MUST NOT be modified.
- `src/parser.py` MUST NOT be modified.
- `src/config.py` MUST NOT be modified.
- `src/schemas.py` MUST NOT be modified.
- `src/data.py` MUST NOT be modified.
- `src/metrics.py` MUST NOT be modified.
- `src/main.py` MUST NOT be modified.

The only permitted change is within `src/model.py`, specifically:
1. Hardcoding the model identifier inside `load_model_and_processor()`.
2. Any minor internal adapter logic strictly needed to make the Outlines wrapping succeed with that specific model class.

---

## Version History

| Date | Version | Change |
|------|---------|--------|
| 2026-05-06 | 1.0 | Initial contract; load_model_and_processor signature simplified (model_id param removed) |
