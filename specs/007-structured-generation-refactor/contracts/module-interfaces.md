# Module Interface Contracts

**Feature**: Structured Generation Refactor for Evaluation Pipeline  
**Date**: 2026-05-06  
**Status**: Proposed (changes from baseline documented)

---

## Contract 1: `src/model.py` ↔ `src/pipeline.py`

### Function: `load_model_and_processor`

**Signature**:
```python
def load_model_and_processor() -> tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    ...
```

**Changes from baseline**:
- Parameter `model_id: str = MODEL_ID` removed (hardcoded internally per constitution).
- Returns unchanged: `(model, processor)` in eval mode.

**Post-conditions**:
- Model is loaded with `torch.bfloat16`, `device_map="auto"`, `trust_remote_code=True`.
- Processor is loaded from the same checkpoint.
- Outlines wrapper is NOT built here (built lazily in `run_inference` via `_get_cached_generator`).

---

### Function: `run_inference`

**Signature**:
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

**Changes from baseline**:
- `max_new_tokens` default increased from `512` to `2048`.
- Internal implementation replaces `model.generate()` + `processor.batch_decode()` with Outlines `Generator(..., output_type=ModelResponse)` + `inputs.Chat()` + `inputs.Image()`.
- Return value is still `str`, but now guaranteed to be valid JSON text matching `ModelResponse` schema.

**Pre-conditions**:
- `model` and `processor` are from `load_model_and_processor()`.
- `images` are RGB PIL images (0–7 items).
- `prompt` contains the evaluation question with JSON schema instruction.

**Post-conditions**:
- Returns a JSON string that validates against `ModelResponse`.
- No unconstrained `.generate()` calls are made.

---

## Contract 2: `src/schemas.py` ↔ `src/model.py`, `src/parser.py`

### Class: `ModelResponse`

**Definition**:
```python
class ModelResponse(BaseModel):
    reasoning: str = Field(default="", description="Step-by-step reasoning.")
    answer: Literal["A", "B", "C", "D"] = Field(..., description="Selected option letter.")
```

**Invariants**:
- `answer` is always one of A, B, C, D (enforced by Pydantic and Outlines logits processor).
- `reasoning` is always a string (may be empty).

**Consumers**:
- `src/model.py`: Passed to `Generator(..., output_type=ModelResponse)`.
- `src/parser.py`: Used for `json.loads` + `ModelResponse.model_validate()`.

---

## Contract 3: `src/parser.py` ↔ `src/pipeline.py`

### Function: `extract_answer`

**Signature**:
```python
def extract_answer(text: str, num_choices: int = 4) -> tuple[str | None, bool]:
    ...
```

**Changes from baseline**:
- **REMOVED**: All regex logic (`re.search`, `re.findall`, `re.match`, `re.compile`).
- **REMOVED**: All string-heuristic fallback strategies (first-character match, pattern matching, standalone letter search).
- **ADDED**: `json.loads` attempt on stripped text.
- **ADDED**: `ModelResponse.model_validate(parsed)` for schema validation.
- Returns `(validated.answer, True)` on success, `(None, False)` on any failure.

**Invariants**:
- No `re` module usage in the parser.
- Only `json` and `pydantic` modules are used for extraction.
- Malformed JSON or validation errors result in `(None, False)` — no guesswork.

---

## Contract 4: `src/config.py` ↔ All modules

### Module: `src.config`

**Exports**:
```python
MAX_SAMPLES = 100
MAX_NEW_TOKENS = 2048
RESULTS_DIR = "results"
TRAJECTORIES_PATH = "results/trajectories.jsonl"
SUMMARY_PATH = "results/summary.json"
SUBJECTS = ["Accounting", "Architecture_and_Engineering", "Art", "Biology"]

def get_subject_quotas(max_samples: int, num_subjects: int) -> list[int]: ...
```

**Changes from baseline**:
- `MAX_NEW_TOKENS` increased from `512` (previous default in `model.py`) to `2048`.
- File is created/restored (was missing from working tree but referenced by `main.py` and `data.py`).

**Consumers**:
- `src/main.py`: imports `config` for paths, cap, and resume logic.
- `src/data.py`: imports `config` for subjects and quotas.
- `src/pipeline.py`: imports `config` for `MAX_NEW_TOKENS` (passed to `run_inference`).

---

## Contract 5: `src/pipeline.py` ↔ `src/model.py`

### Function: `build_prompt`

**Signature**:
```python
def build_prompt(question: str, options: List[str]) -> str:
    ...
```

**Changes from baseline**:
- Final instruction line updated from:
  `"Answer with the option's letter from the given choices directly."`
  to:
  `"\nThink step by step, then respond with ONLY valid JSON matching this exact schema: {\"reasoning\": \"<your reasoning>\", \"answer\": \"<A|B|C|D>\"}."`

**Invariant**:
- Prompt must instruct the model to produce reasoning + structured JSON.
- Prompt must remain compatible with `inputs.Chat()` format (plain text, no markdown fences).

---

## Change Boundary Summary

| File | Change Type | Scope |
|------|-------------|-------|
| `src/model.py` | Refactor | Replace `generate()` with Outlines `Generator`; increase default `max_new_tokens` |
| `src/schemas.py` | Create | New `ModelResponse` Pydantic schema |
| `src/parser.py` | Refactor | Remove regex; add Pydantic JSON validation |
| `src/config.py` | Create/Restore | Add `MAX_NEW_TOKENS = 2048` and other constants |
| `src/pipeline.py` | Update | Update prompt text only |
| `pyproject.toml` | Update | Add `outlines`, `pydantic`, `lm-format-enforcer` dependencies |
| `uv.lock` | Regenerate | Reflect new dependency tree |

**Unchanged files**: `src/__init__.py`, `src/data.py`, `src/main.py`, `src/metrics.py`.
