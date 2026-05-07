# Data Model: Model Initialization Layer

**Feature**: HF Qwen3-VL Model Initialization with Preserved Structured Generation  
**Date**: 2026-05-06  
**Scope**: Entities impacted by the initialization-layer change in `src/model.py`.

---

## Entities

### Entity: ModelConfiguration

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| model_id | str | Canonical HuggingFace model identifier | Read-only constant: `Qwen/Qwen3-VL-2B-Instruct` |
| device | str | Target compute device | "cuda" if available, otherwise "cpu" |
| torch_dtype | torch.dtype | Tensor precision for model weights | `torch.bfloat16` |
| device_map | str | Device placement strategy | `"auto"` |
| trust_remote_code | bool | Allow custom model/processor code | `True` |

**Relationships**:
- Passed into `load_model_and_processor()` function (currently accepts override; will be hardcoded per FR-001).
- Produces: Loaded model weights + processor instance.

---

### Entity: StructuredOutputSchema (unchanged)

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| reasoning | str | Step-by-step chain of thought | Min length: 0 |
| answer | Literal["A","B","C","D"] | Extracted multiple-choice letter | Exactly one of A/B/C/D |

**Validation Rules**:
- `answer` must be uppercase single letter.
- `reasoning` is optional but encouraged (CoT enabled).

**Relationships**:
- Consumed by: Outlines `Generator` as `output_type`.
- Produced by: Model during inference.
- Parsed by: `src/parser.py` via `json.loads` + Pydantic `.model_validate()`.

---

### Entity: EvaluationRecord (unchanged)

| Field | Type | Description |
|-------|------|-------------|
| sample_id | str | MMMU sample identifier |
| subject | str | MMMU subject category |
| question | str | De-image-tagged question text |
| choices | List[str] | Formatted option strings |
| correct_answer | str | Ground truth letter |
| prompt_sent | str | Full prompt delivered to model |
| raw_model_response | str | Raw output from model (pre-parsing) |
| extracted_answer | str | Parsed answer letter (or null) |
| extraction_succeeded | bool | Whether parser validated successfully |
| is_correct | bool | Whether extracted_answer == correct_answer |
| inference_time_seconds | float | Wall-clock inference duration |
| error | str | Error message if inference/parsing failed |

**State Transitions**:
1. `Initialized` → `Prompt Built` (pipeline.py)
2. `Prompt Built` → `Response Ready` (model.py inference)
3. `Response Ready` → `Parsed` / `Parse Failed` (parser.py)
4. `Parsed` → `Correct` / `Incorrect` (metrics.py)
5. Any state → `Error` (on exception; logged, pipeline continues)

---

## Scope Boundary

- `ModelConfiguration.model_id` becomes immutable (hardcoded).
- No new entities introduced.
- No fields added to or removed from existing entities.
- Relationships between pipeline components remain unchanged.
