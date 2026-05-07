# Data Model: Structured Generation Refactor

**Feature**: Structured Generation Refactor for Evaluation Pipeline  
**Date**: 2026-05-06  
**Scope**: Entities impacted by the architectural refactor.

---

## Entities

### Entity: ModelResponse (NEW)

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| reasoning | str | Step-by-step chain of thought leading to the answer | Min length: 0 (allowed but discouraged) |
| answer | Literal["A","B","C","D"] | Final multiple-choice letter | Exactly one of A/B/C/D; validated at generation time by Outlines |

**Validation Rules**:
- `answer` must be uppercase single letter; enforced by Pydantic `Literal` and Outlines logits processor.
- `reasoning` is optional (empty string allowed) but the prompt encourages non-empty reasoning.

**Relationships**:
- Consumed by: Outlines `Generator` as `output_type`.
- Produced by: Model during inference (`src/model.py`).
- Parsed by: `src/parser.py` via `json.loads` + `ModelResponse.model_validate()`.

---

### Entity: ModelConfiguration (RESTORED/UPDATED)

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| model_id | str | Canonical HuggingFace model identifier | Constant: `Qwen/Qwen3-VL-2B-Instruct` |
| max_samples | int | Total evaluation sample cap | 100 |
| max_new_tokens | int | Maximum decoding steps per sample | 2048 |
| results_dir | str | Directory for output files | "results" |
| trajectories_path | str | JSONL trajectory file path | "results/trajectories.jsonl" |
| summary_path | str | Aggregate summary file path | "results/summary.json" |
| subjects | list[str] | MMMU subjects to evaluate | ["Accounting", "Architecture_and_Engineering", "Art", "Biology"] |

**Relationships**:
- Imported by: `src/main.py`, `src/data.py`, `src/pipeline.py` (via `evaluate_sample`), `src/metrics.py`.
- Defines: Global evaluation parameters.

---

### Entity: EvaluationRecord (UNCHANGED)

| Field | Type | Description |
|-------|------|-------------|
| sample_id | str | MMMU sample identifier |
| subject | str | MMMU subject category |
| question | str | De-image-tagged question text |
| choices | List[str] | Formatted option strings |
| correct_answer | str | Ground truth letter |
| prompt_sent | str | Full prompt delivered to model |
| raw_model_response | str | Raw output from model (now guaranteed to be JSON text) |
| extracted_answer | str | Parsed answer letter (or null) |
| extraction_succeeded | bool | Whether parser validated successfully |
| is_correct | bool | Whether extracted_answer == correct_answer |
| inference_time_seconds | float | Wall-clock inference duration |
| error | str | Error message if inference/parsing failed |

**State Transitions**:
1. `Initialized` → `Prompt Built` (pipeline.py)
2. `Prompt Built` → `Response Ready` (model.py inference with Outlines constraint)
3. `Response Ready` → `Parsed` / `Parse Failed` (parser.py via Pydantic validation)
4. `Parsed` → `Correct` / `Incorrect` (metrics.py)
5. Any state → `Error` (on exception; logged, pipeline continues)

---

## Scope Boundary

- `ModelResponse` is a new entity (previously absent).
- `ModelConfiguration` is restored/updated (previously missing from working tree; `__pycache__` indicated its former existence).
- `EvaluationRecord` format remains unchanged — backward compatible with existing `trajectories.jsonl`.
- No new fields added to `EvaluationRecord`.
- Relationships between pipeline components remain unchanged.
