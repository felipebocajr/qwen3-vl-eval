# Tasks: HF Qwen3-VL Model Initialization with Preserved Structured Generation

**Input**: Design documents from `specs/005-hf-qwen3vl-model-init/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/module-interfaces.md, quickstart.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Verify environment readiness before making any code changes

- [x] T001 Verify `src/model.py`, `src/parser.py`, `src/config.py`, `src/schemas.py`, `src/pipeline.py`, and `src/main.py` exist and are importable in the current virtual environment — **RESULT**: `src/model.py` ✅, `src/parser.py` ✅, `src/pipeline.py` ✅, `src/main.py` ✅. `src/config.py` ❌ MISSING (was previously present — `__pycache__/config.cpython-312.pyc` exists with `MAX_NEW_TOKENS=2048`). `src/schemas.py` ❌ MISSING.
- [x] T002 [P] Run `python -c "import torch, transformers, outlines, pydantic; print('OK')"` to confirm all required packages are installed — **RESULT**: ✅ All packages importable.
- [x] T003 [P] Confirm `src/config.py` defines `MAX_NEW_TOKENS` with value `2048` (verify it has not been lowered) — **RESULT**: `src/config.py` is missing. From compiled bytecode (`src/__pycache__/config.cpython-312.pyc`), previous `MAX_NEW_TOKENS` was `2048`. Current `src/model.py` has `max_new_tokens: int = 512` which is LOWER than the spec baseline.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish baseline behavior of the existing Outlines/Pydantic pipeline before modifying initialization

**⚠️ CRITICAL**: These verification tasks MUST complete before ANY user-story implementation can begin.

- [ ] T004 Run a single-sample inference end-to-end using the current `src/main.py` and capture one raw response to confirm structured generation produces JSON matching `ModelResponse` schema
- [ ] T005 [P] Run `python -c "from src.schemas import ModelResponse; print(ModelResponse.model_json_schema())"` to confirm `ModelResponse` still enforces `reasoning` and `answer` fields
- [ ] T006 [P] Verify `src/parser.py` uses Pydantic validation (not regex/string parsing) by reading its imports: confirm it imports `ModelResponse` and validates with `.model_validate()`

**Checkpoint**: Baseline confirmed — structured generation pipeline is intact before initialization changes.

---

## Phase 3: User Story 1 – Stable Model Loading with Structured Generation (Priority: P1) 🎯 MVP

**Goal**: Hardcode the canonical HuggingFace model identifier in `src/model.py` and ensure the Outlines/Pydantic pipeline initializes correctly against it.

**Independent Test**: After completing these tasks, `python -c "from src.model import load_model_and_processor; m, p = load_model_and_processor()"` must load `Qwen/Qwen3-VL-2B-Instruct` from the canonical source without accepting arbitrary model overrides.

### Implementation for User Story 1

- [ ] T007 [US1] Remove the `model_id` parameter from `load_model_and_processor()` in `src/model.py` and hardcode `Qwen/Qwen3-VL-2B-Instruct` directly inside the function body
- [ ] T008 [US1] Update the docstring of `load_model_and_processor()` in `src/model.py` to document that the function loads **exclusively** the canonical identifier and accepts no overrides
- [ ] T009 [US1] Confirm `src/model.py` still passes the `model` and `processor` instances into the Outlines wrapper (`from_transformers` → `Generator` with `output_type=ModelResponse`) exactly as before; if any Outlines initialization error occurs with this specific model ID, adapt the wrapper call strictly inside `src/model.py` without leaking changes to any other module
- [ ] T010 [US1] Run a single-sample end-to-end inference after the initialization change and verify the raw model response is valid JSON that passes `ModelResponse.model_validate()`

**Checkpoint**: User Story 1 complete — model loads exclusively from the canonical source and structured generation continues to work.

---

## Phase 4: User Story 2 – Answer Extraction Continuity (Priority: P2)

**Goal**: Confirm answer extraction via Pydantic validation remains unchanged and reliable after the initialization-layer update.

**Independent Test**: Feed the same set of valid/invalid JSON strings to `src/parser.py` before and after the change; assert identical extracted answers and identical success/failure flags.

### Implementation for User Story 2

- [ ] T011 [US2] Verify `src/parser.py` has zero functional changes since baseline by comparing against a snapshot of the file before any model.py edits (no regex added, no string-matching heuristics introduced)
- [ ] T012 [US2] Run a batch of 10 labeled JSON responses through `src/parser.extract_answer()` and assert 100% extraction matches pre-change baseline letters and success flags
- [ ] T013 [US2] Run the full MMMU validation subset (100 samples) and assert overall extraction success rate is at or above the pre-change baseline

**Checkpoint**: User Story 2 complete — parser behavior identical to baseline.

---

## Phase 5: User Story 3 – Reasoning Chain Preservation (Priority: P3)

**Goal**: Confirm chain-of-thought reasoning is preserved and not truncated by any change to the initialization or generation path.

**Independent Test**: Inspect `raw_model_response` fields from a batch run; at least 95% contain non-empty `reasoning` text and `max_new_tokens` remains at 2048.

### Implementation for User Story 3

- [ ] T014 [US3] Verify `src/config.py` still defines `MAX_NEW_TOKENS = 2048`; confirm no commit has altered this value
- [ ] T015 [US3] Verify `src/model.py` still passes `max_new_tokens=2048` (or uses the config default) to the Outlines Generator; confirm no truncation-enforcing generation argument was added
- [ ] T016 [US3] Run a batch of 20 inference samples and programmatically inspect `raw_model_response` JSON to confirm ≥ 95% contain non-empty `reasoning` fields
- [ ] T017 [US3] Verify `src/pipeline.py` prompts still instruct the model to provide step-by-step reasoning in the system message or user prompt

**Checkpoint**: User Story 3 complete — CoT preserved, output budget unchanged.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final validation that downstream modules were not touched and the pipeline is fully functional

- [ ] T018 [P] Run `git diff --stat` and assert only `src/model.py` has modifications; all other files in `src/` must show zero changes
- [ ] T019 [P] Run the full 100-sample pipeline end-to-end and verify `results/trajectories.jsonl` and `results/summary.json` are generated correctly
- [ ] T020 [P] Interrupt the pipeline after sample 15, restart it, and confirm it resumes at sample 16 without re-evaluating completed IDs (resumability intact)
- [ ] T021 Update `specs/005-hf-qwen3vl-model-init/quickstart.md` with any new troubleshooting notes specific to the canonical model identifier

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Phase 1 completion. BLOCKS all user stories until baseline is confirmed.
- **User Story 1 (Phase 3)**: Depends on Phase 2. Model loading is the MVP; all other stories depend on it.
- **User Story 2 (Phase 4)**: Depends on Phase 3 (needs running model to test parser against real outputs). Can start as soon as US1 checkpoint passes.
- **User Story 3 (Phase 5)**: Depends on Phase 3 (needs running model to inspect reasoning). Can run in parallel with US2 if team capacity allows.
- **Polish (Phase 6)**: Depends on all user stories.

### User Story Dependencies

| Story | Depends On | Can Run In Parallel With |
|-------|-----------|--------------------------|
| US1 (P1) | Phase 2 Foundational | Nothing |
| US2 (P2) | US1 complete | US3 |
| US3 (P3) | US1 complete | US2 |

### Within Each User Story

- US1: code change → single-sample smoke test
- US2: parser unchanged check → batch parser validation → full-run validation
- US3: config value check → generation parameter check → batch reasoning inspection → prompt text check

### Parallel Opportunities

- Phase 1 tasks T001, T002, T003 can run in parallel.
- Phase 2 tasks T004, T005, T006 can run in parallel.
- Phase 6 tasks T018, T019, T020, T021 can run in parallel once US1–US3 are complete.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational baseline verification).
2. Complete Phase 3 (US1): Hardcode model ID, test single-sample inference.
3. **STOP and VALIDATE**: Confirm model loads from `Qwen/Qwen3-VL-2B-Instruct` and structured generation produces valid `ModelResponse` JSON.
4. If validation passes, proceed to US2 and US3 in parallel.

### Incremental Delivery

- After US1: Model loading is correct and structured generation works.
- After US2: Parser behavior confirmed identical to baseline.
- After US3: CoT reasoning and output budget preserved.
- After Phase 6: Full pipeline integrity verified, resumability intact, zero downstream modifications.

---

## Notes

- [P] tasks = different files or independent verification steps with no ordering dependency.
- [Story] label maps task to specific user story for traceability.
- Each user story is independently completable and testable.
- Avoid vague tasks; every task references a concrete file path or command.
- Commit after each task or logical group.
- The overriding constraint: **only `src/model.py` may be modified**; all other source files must remain untouched.
