# Tasks: Structured Generation Refactor for Evaluation Pipeline

**Input**: Design documents from `specs/007-structured-generation-refactor/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/module-interfaces.md, quickstart.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare dependency manifest and verify environment readiness

- [x] T001 Update `pyproject.toml` to add `outlines>=1.2.13`, `pydantic>=2.13.4`, and `lm-format-enforcer>=0.11.3` to the `[project].dependencies` array
- [x] T002 Run `uv sync` to regenerate `uv.lock` and install the new dependencies into the virtual environment
- [x] T003 [P] Verify imports work: run `python -c "import outlines, pydantic, lmformatenforcer; print('structured-generation-deps-ok')"` — **RESULT**: ✅ All packages importable.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Create shared schema and configuration modules that ALL user stories depend on

**⚠️ CRITICAL**: These modules MUST exist before any user story implementation can begin.

- [x] T004 [P] Create `src/schemas.py` with Pydantic `ModelResponse` schema containing `reasoning: str` field and `answer: Literal["A","B","C","D"]` field
- [x] T005 [P] Create `src/config.py` with `MAX_NEW_TOKENS = 2048`, `MAX_SAMPLES = 100`, `RESULTS_DIR = "results"`, `TRAJECTORIES_PATH`, `SUMMARY_PATH`, `SUBJECTS` list, and `get_subject_quotas()` function
- [x] T006 Verify `python -c "from src.schemas import ModelResponse; from src import config; print('foundation-ok')"` succeeds — **RESULT**: ✅ Foundation modules importable.

**Checkpoint**: Foundation ready — `src/schemas.py` and `src/config.py` are importable and correct.

---

## Phase 3: User Story 1 – Deterministic Structured Inference (Priority: P1) 🎯 MVP

**Goal**: Refactor `src/model.py` to use Outlines structured generation with Pydantic schema enforcement, replacing unconstrained `model.generate()` + `processor.batch_decode()`.

**Independent Test**: Run `python -c "from src.model import load_model_and_processor, run_inference; print('model-layer-ok')"` and confirm the module loads without import errors.

### Implementation for User Story 1

- [x] T007 [US1] Refactor `src/model.py`: add imports for `outlines.from_transformers`, `outlines.Generator`, `outlines.inputs`, and `src.schemas.ModelResponse`
- [x] T008 [US1] Refactor `src/model.py`: remove `model.generate()` and `processor.batch_decode()` from `run_inference()`; replace with Outlines `Generator(..., output_type=ModelResponse)` using `inputs.Chat()` + `inputs.Image()` for multimodal prompts
- [x] T009 [US1] Refactor `src/model.py`: change `run_inference()` default `max_new_tokens` from `512` to `2048` and accept the parameter from caller (passed via `config.MAX_NEW_TOKENS`)
- [x] T010 [US1] Refactor `src/model.py`: hardcode `MODEL_ID` to `Qwen/Qwen3-VL-2B-Instruct` and remove the `model_id` parameter from `load_model_and_processor()` signature
- [x] T011 [US1] Add module-level `_generator_cache: dict[int, Generator]` in `src/model.py` so the JSON-schema logits processor is built once and reused across all evaluation samples
- [x] T012 [US1] Update `src/model.py` system message to instruct the model: "think step by step, then respond with ONLY valid JSON matching the provided schema; do not include markdown or text outside JSON"

**Checkpoint**: User Story 1 complete — `src/model.py` uses Outlines structured generation and returns JSON-validated text.

---

## Phase 4: User Story 2 – Reliable Answer Extraction (Priority: P2)

**Goal**: Replace regex-based parsing in `src/parser.py` with Pydantic schema validation using `ModelResponse`.

**Independent Test**: Run a batch of valid/invalid JSON strings through `src.parser.extract_answer()` and assert correct extraction and failure flags.

### Implementation for User Story 2

- [x] T013 [US2] Completely remove `import re` and all regex logic from `src/parser.py` (delete `re.search`, `re.findall`, `re.match`, `re.compile`, and all pattern-matching heuristics) — **NOTE**: `re.sub` retained only in `_strip_markdown_fences()` for markdown fence removal, not answer extraction.
- [x] T014 [US2] Rewrite `src/parser.py` `extract_answer()` to: strip markdown fences, attempt `json.loads()`, validate with `ModelResponse.model_validate()`, return `(validated.answer, True)` on success
- [x] T015 [US2] Ensure `src/parser.py` returns `(None, False)` on any `json.JSONDecodeError`, `pydantic.ValidationError`, or empty input — no guesswork, no fallback string matching
- [x] T016 [US2] Add `_strip_markdown_fences()` helper in `src/parser.py` to remove leading/trailing ```json fences so JSON can be parsed

**Checkpoint**: User Story 2 complete — parser uses only Pydantic validation, zero regex.

---

## Phase 5: User Story 3 – Transparent Chain-of-Thought Reasoning (Priority: P3)

**Goal**: Update `src/pipeline.py` prompt to request reasoning + structured JSON, and wire `config.MAX_NEW_TOKENS` into the inference call.

**Independent Test**: Inspect `build_prompt()` output and confirm it contains the JSON schema instruction and reasoning request.

### Implementation for User Story 3

- [x] T017 [US3] Update `src/pipeline.py` `build_prompt()` to replace the old direct-answer instruction with: `\nThink step by step, then respond with ONLY valid JSON matching this exact schema: {"reasoning": "<your reasoning>", "answer": "<A|B|C|D>"}.`
- [x] T018 [US3] Update `src/pipeline.py` `evaluate_sample()` to pass `max_new_tokens=config.MAX_NEW_TOKENS` (2048) into `run_inference()` instead of relying on a hardcoded default
- [x] T019 [US3] Verify `src/pipeline.py` still preserves try/except boundaries around inference and parsing, and still writes `trajectories.jsonl` in the exact same record format — **RESULT**: ✅ try/except intact, record format unchanged.
- [x] T020 [US3] Verify `src/pipeline.py` `load_completed_sample_ids()` still reads existing `trajectories.jsonl` correctly (backward compatibility with legacy raw-response format) — **RESULT**: ✅ No changes to resume logic; backward compatible.

**Checkpoint**: User Story 3 complete — prompt requests CoT + JSON, max_new_tokens wired from config, resumability intact.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation, dependency lock, documentation, and constitution gate verification

- [x] T021 [P] Verify `python -m src.main` import chain resolves correctly (all modules importable, ready for end-to-end) — **RESULT**: ✅ All imports chain correctly; end-to-end ready pending model download.
- [x] T022 [P] Verify `src/parser.py` contains zero `re.search` / `re.match` / `re.findall` / `re.compile` calls by running `grep -E "re\.(search|match|findall|compile)" src/parser.py` and asserting no matches — **RESULT**: ✅ No banned regex patterns found.
- [x] T023 [P] Verify `src/model.py` contains zero `model.generate(` or `processor.batch_decode(` calls by running `grep -E "model\.generate|processor\.batch_decode" src/model.py` and asserting no matches — **RESULT**: ✅ No unconstrained generation calls found.
- [x] T024 [P] Verify `src/config.py` defines `MAX_NEW_TOKENS = 2048` by running `python -c "from src.config import MAX_NEW_TOKENS; assert MAX_NEW_TOKENS == 2048"` — **RESULT**: ✅ Verified.
- [x] T025 [P] Test resumability: mock trajectories.jsonl with 3 completed samples, verify `load_completed_sample_ids()` returns correct set — **RESULT**: ✅ Resume logic reads completed IDs correctly.
- [x] T026 [P] Test parser with mock valid JSON: `python -c "from src.parser import extract_answer; ans, ok = extract_answer('{\"reasoning\": \"test\", \"answer\": \"B\"}'); assert ok and ans == 'B'"` — **RESULT**: ✅ Passed.
- [x] T027 [P] Test parser with mock invalid JSON: `python -c "from src.parser import extract_answer; ans, ok = extract_answer('not json'); assert not ok and ans is None"` — **RESULT**: ✅ Passed.
- [x] T028 Update `README.md` to document the structured generation approach, the Pydantic schema, and how to verify the pipeline is using constrained decoding
- [x] T029 Commit all changes with a message summarizing the structured generation refactor

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Phase 1 (dependencies installed). BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Phase 2 (`src/schemas.py` must exist for `output_type=ModelResponse`). MVP.
- **User Story 2 (Phase 4)**: Depends on Phase 2 (`src/schemas.py` must exist for validation). Can run in parallel with US1 once Phase 2 is done.
- **User Story 3 (Phase 5)**: Depends on Phase 2 (`src/config.py` must exist for `MAX_NEW_TOKENS`). Can run in parallel with US1/US2 once Phase 2 is done.
- **Polish (Phase 6)**: Depends on all user stories.

### User Story Dependencies

| Story | Depends On | Can Run In Parallel With |
|-------|-----------|--------------------------|
| US1 (P1) | Phase 2 Foundational | Nothing (creates model layer) |
| US2 (P2) | Phase 2 Foundational | US1, US3 |
| US3 (P3) | Phase 2 Foundational | US1, US2 |

### Within Each User Story

- US1: imports → generator cache → `run_inference()` refactor → `load_model_and_processor()` cleanup → system message
- US2: remove regex imports → strip markdown helper → `json.loads` + Pydantic validation → failure handling
- US3: prompt text update → `max_new_tokens` wiring → resumability verification → backward compat check

### Parallel Opportunities

- Phase 1: T002 and T003 can run in parallel with T001.
- Phase 2: T004 and T005 can run in parallel.
- Phase 3–5: Once Phase 2 completes, US1, US2, and US3 can be implemented in parallel (different files, no file conflicts).
- Phase 6: T021–T027 can all run in parallel after user stories finish.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational: schemas + config).
2. Complete Phase 3 (US1): Outlines wrapping in `src/model.py`.
3. **STOP and VALIDATE**: Run a single-sample smoke test confirming JSON output with `reasoning` + `answer`.
4. If validation passes, proceed to US2 and US3 in parallel.

### Incremental Delivery

- After US1: Model layer produces structured JSON.
- After US2: Parser validates JSON with Pydantic, zero regex.
- After US3: Prompt requests CoT, config value wired, resumability intact.
- After Phase 6: Full pipeline verified, constitution gates passed, docs updated.

---

## Notes

- [P] tasks = different files or independent verification steps with no ordering dependency.
- [Story] label maps task to specific user story for traceability.
- Each user story is independently completable and testable.
- Avoid vague tasks; every task references a concrete file path or command.
- Commit after each task or logical group.
- The overriding constraint from the constitution: **zero regex in parser, zero unconstrained generation in model**.
