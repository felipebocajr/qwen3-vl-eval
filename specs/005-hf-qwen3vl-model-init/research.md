# Research: HF Qwen3-VL Model Initialization with Preserved Structured Generation

**Date**: 2026-05-06  
**Purpose**: Resolve unknowns about Outlines + Qwen3-VL-2B-Instruct integration and document initialization adaptation strategy.

---

## RQ1: Does Outlines `from_transformers` correctly wrap Qwen3VLForConditionalGeneration?

**Finding**: Outlines' `from_transformers` dispatches on model class. `Qwen3VLForConditionalGeneration` subclasses `GenerationMixin` and `PreTrainedModel`, both of which `from_transformers` supports. The wrapper builds an `OutlinesModel` that intercepts logits generation. For multimodal models, `from_transformers` accepts the model and processor pair and constructs tokenizers that respect the processor's `apply_chat_template` format.

**Implication**: The existing `outlines_model = from_transformers(model, processor)` call in `_get_cached_generator()` is architecturally sound for `Qwen3VLForConditionalGeneration`. No wrapper-level replacement is needed.

**Risk**: If `Qwen3-VL` introduces custom token types (e.g., vision tokens) that the logits processor does not recognize, structured JSON generation could still misfire. Mitigation: validate on a single sample before full run.

---

## RQ2: Are there specific AutoProcessor configuration nuances for Qwen3-VL?

**Finding**: The Qwen3-VL family uses `trust_remote_code=True` for both model and processor loading, which is already present in current code. The `AutoProcessor.from_pretrained` for this model returns a composite processor that handles text tokenization + image normalization. The `qwen-vl-utils` package provides helper functions for image preprocessing but is NOT required for basic processor instantiation.

**Key settings**:
- `torch_dtype=torch.bfloat16` — appropriate for 2B parameter model on modern GPUs.
- `device_map="auto"` — required for multi-device setups; falls back correctly on single GPU or CPU.

**Implication**: Current `load_model_and_processor()` settings are correct. No additional processor configuration changes are needed.

---

## RQ3: Does Outlines `inputs.Chat()` with `inputs.Image()` work with Qwen3-VL?

**Finding**: Outlines' multimodal input helpers (`inputs.Image`) are designed to work with any VL model that its `from_transformers` wrapper supports. The helper converts images into the processor's expected format before calling the model. The `chat.add_user_message(content_parts)` pattern appends both text strings and image objects, which the processor then converts into the model's native image-token representation.

**Implication**: Existing `run_inference()` logic using `inputs.Chat()`, `inputs.Image()`, and `generator(chat, ...)` is the correct pattern. No adaptation needed in the inference path.

---

## RQ4: What initialization changes are actually required?

**Finding**: The current `src/model.py` already:
- Uses the correct model ID `Qwen/Qwen3-VL-2B-Instruct`
- Loads with the correct model class (`Qwen3VLForConditionalGeneration`)
- Loads with the correct processor (`AutoProcessor`)
- Wraps with Outlines correctly (`from_transformers` → `Generator` with `ModelResponse`)

However, the `load_model_and_processor(model_id: str = MODEL_ID)` function accepts an override parameter. To satisfy **FR-001** (exclusive resolution to the canonical identifier), the function should either:
1. Hardcode the identifier internally and ignore any override, OR
2. Validate the passed identifier against the canonical ID and raise on mismatch.

Per the "exclusive" language in FR-001, **Option 1 (hardcode)** is the stronger interpretation.

---

## Conclusion

The required changes are minimal and confined to `src/model.py`:
1. **Model ID enforcement**: Remove the parameter override from `load_model_and_processor()` and hardcode `Qwen/Qwen3-VL-2B-Instruct` inside the function body.
2. **No wrapper or processor adaptation needed**: The existing Outlines wrapping and processor loading are already correct for this model.
3. **Verification**: Run a single-sample end-to-end test to confirm structured JSON outputs still validate against `ModelResponse`.

All other modules (`parser.py`, `pipeline.py`, `config.py`, `main.py`, `schemas.py`) remain unchanged.
