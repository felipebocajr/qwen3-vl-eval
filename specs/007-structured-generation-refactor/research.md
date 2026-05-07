# Research: Structured Generation Refactor for Evaluation Pipeline

**Date**: 2026-05-06  
**Purpose**: Resolve unknowns about Outlines + Qwen3-VL-2B-Instruct integration, JSON schema enforcement on VL models, and CoT output budget sizing.

---

## RQ1: Does Outlines `from_transformers` correctly wrap Qwen3VLForConditionalGeneration?

**Finding**: Outlines' `from_transformers` dispatches on model class. `Qwen3VLForConditionalGeneration` subclasses `GenerationMixin` and `PreTrainedModel`, both of which `from_transformers` supports. For multimodal models, `from_transformers` accepts the model and processor pair and constructs tokenizers that respect the processor's `apply_chat_template` format.

**Implication**: The call `outlines_model = from_transformers(model, processor)` is architecturally sound for `Qwen3VLForConditionalGeneration`. No wrapper-level replacement is needed.

**Risk**: If Qwen3-VL introduces custom token types (e.g., vision tokens) that the logits processor does not recognize, structured JSON generation could still misfire. Mitigation: validate on a single sample before full run.

---

## RQ2: Does `Generator(..., output_type=ModelResponse)` constrain only decoder text tokens?

**Finding**: Outlines' `Generator` applies a logits processor at every decoding step of the *language model head*. The vision encoder runs first, producing image embeddings that are fed into the decoder. The logits processor only constrains the probability distribution over the vocabulary tokens at each decoder step. This means the vision pathway is unaffected, and only the emitted text must conform to the Pydantic schema.

**Implication**: Using `Generator(outlines_model, output_type=ModelResponse)` with multimodal inputs is the correct pattern. The model can see images, reason about them, and then emit valid JSON structured output.

---

## RQ3: Is `max_new_tokens=2048` sufficient for CoT reasoning on MMMU questions?

**Finding**: MMMU questions vary in complexity. Simple questions may require 50–100 tokens of reasoning; complex multi-step questions may require 300–800 tokens. The JSON schema overhead (field names, braces, quotes) adds ~20–40 tokens. A budget of 2048 tokens provides substantial headroom for all but the most extreme reasoning chains.

**Implication**: 2048 is appropriate for the evaluation pipeline. OOM on very long sequences is a known risk for small GPUs; the existing `evaluate_sample` try/except catches `torch.cuda.OutOfMemoryError` and logs the failure, allowing the pipeline to continue.

---

## RQ4: What prompt format works best with Outlines + Qwen3-VL?

**Finding**: Outlines' `inputs.Chat()` helper constructs chat messages that the processor's `apply_chat_template` converts into the model's native format. For Qwen3-VL, the system message should explicitly instruct JSON-only output matching the schema. The user message contains the question, options, and images.

**Recommended prompt pattern**:
- System: "You are a helpful assistant. For multiple-choice questions, think step by step and explain your reasoning, then respond with ONLY valid JSON matching the provided schema. Do not include markdown, explanations, or any text outside the JSON."
- User: `[prompt text, images...]`

---

## Conclusion

The Outlines + Pydantic structured generation pipeline is technically sound for this model and use case:
1. `from_transformers(model, processor)` wraps the model correctly.
2. `Generator` with `output_type=ModelResponse` constrains only decoder outputs.
3. `max_new_tokens=2048` is sufficient for CoT + JSON on MMMU.
4. The `inputs.Chat()` + `inputs.Image()` pattern is the correct multimodal input approach.

All five file changes (`model.py`, `schemas.py`, `parser.py`, `config.py`, `pipeline.py`) are well-defined and scoped.
