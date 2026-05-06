# Tasks: Limit Pipeline to 100 Samples with Proportional Stratification

**Input**: Design documents from `specs/001-limit-100-samples/`
**Prerequisites**: plan.md, spec.md (required for user stories)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm project structure and dependencies are in place for certification

- [ ] T001 Verify `pyproject.toml` includes `datasets` dependency and `src/` is importable
- [ ] T002 Create `scripts/` directory if it does not exist

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Audit the existing proportional quota math and stratified selection logic **before** certifying any user story

**⚠️ CRITICAL**: No user story certification can proceed until this phase confirms the core algorithm is correct

- [ ] T003 Verify `src/config.py` `get_subject_quotas(100, 4)` returns `[25, 25, 25, 25]`, and for arbitrary `N` returns proportional split with remainder distributed to first `N` subjects
- [ ] T004 Verify `src/data.py` `select_stratified_samples()` applies `min(quota, len(ds))` per subject deterministically via `dataset.select(range(n))`, concatenates in fixed order, and **does not backfill** shortfalls
- [ ] T005 Verify `src/data.py` `get_evaluation_dataset()` iterates over `sorted(config.SUBJECTS)` for deterministic ordering and passes the resulting `subset_datasets` list to `select_stratified_samples()` with `max_samples` capped at `config.MAX_SAMPLES`

**Checkpoint**: `get_evaluation_dataset()` yields ≤100 samples with proportional subject quotas and deterministic index slicing

---

## Phase 3: User Story 1 – Fresh Run Fetches Exactly 100 Stratified Samples (Priority: P1) 🎯 MVP

**Goal**: Certify that a clean pipeline start selects exactly 100 MMMU samples with proportional allocation across subjects (e.g., 25 each from 4 configured subjects).

**Independent Test**: Delete `results/`, run `python -m src.main`, then assert `results/trajectories.jsonl` has exactly 100 records (or fewer if dataset is smaller) and per-subject counts match the proportional quotas computed by `config.get_subject_quotas()`.

### Implementation for User Story 1

- [ ] T006 [P] [US1] Create `scripts/certify_stratified.py` that imports `src.data.get_evaluation_dataset` and `src.config`, loads the full stratified dataset, prints per-subject counts, and asserts total samples equals the sum of applied quotas
- [ ] T007 [P] [US1] In `scripts/certify_stratified.py`, assert that for the current `config.SUBJECTS` list each subject receives `base + (1 if i < remainder else 0)` samples where `base = 100 // num_subjects` and `remainder = 100 % num_subjects`
- [ ] T008 [US1] Handle shortfall edge case in `scripts/certify_stratified.py`: if any subject has fewer samples than its quota, assert total is `< 100` and log the shortfall per subject without raising an error
- [ ] T009 [US1] Execute `scripts/certify_stratified.py` and confirm output shows **25 samples per subject × 4 subjects = 100 total** with deterministic alphabetical ordering

**Checkpoint**: At this point, `get_evaluation_dataset()` is empirically certified to return proportional stratified samples

---

## Phase 4: User Story 2 – Resumed Run Respects Stratified Cap (Priority: P2)

**Goal**: Certify that partial runs resumed after interruption correctly skip already-completed IDs and still respect the overall 100-sample cap, regardless of which subjects were partially covered.

**Independent Test**: Simulate 30 completed trajectories, run `python -m src.main`, and verify the combined unique sample count never exceeds 100 and completed IDs are skipped.

### Implementation for User Story 2

- [ ] T010 [US2] Verify `src/pipeline.py` `load_completed_sample_ids()` correctly reads `results/trajectories.jsonl` and returns a deduplicated set of sample IDs
- [ ] T011 [US2] Verify `src/main.py` checks `if n_completed >= config.MAX_SAMPLES` before loading the model or dataset, and exits with code `0` gracefully when the cap is already reached
- [ ] T012 [US2] Verify `src/main.py` computes `n_to_evaluate = min(len(dataset), config.MAX_SAMPLES - n_completed)` and stops the loop once `n_new >= n_to_evaluate`
- [ ] T013 [P] [US2] Add a dry-run test in `scripts/certify_stratified.py` (or a new `scripts/test_resume.py`) that appends 30 fake completed IDs to a temporary `trajectories.jsonl`, calls `load_completed_sample_ids()`, and asserts the cap math allows at most `100 - 30 = 70` new samples

**Checkpoint**: Resume logic does not break stratified quota enforcement and respects the 100-sample hard cap

---

## Phase 5: User Story 3 – Deterministic Stratified Selection (Priority: P3)

**Goal**: Certify that two fresh runs yield identical sample IDs and the same per-subject distribution because selection relies on fixed alphabetical ordering and deterministic index slicing.

**Independent Test**: Run `get_evaluation_dataset()` twice from a clean state and compare sample IDs AND per-subject counts; they must match exactly.

### Implementation for User Story 3

- [ ] T014 [P] [US3] Update `scripts/certify_stratified.py` to invoke `get_evaluation_dataset()` twice, collect `sample["id"]` sets, and assert both runs produce identical ordered lists of IDs
- [ ] T015 [P] [US3] In `scripts/certify_stratified.py`, assert per-subject counts are identical across both deterministic runs
- [ ] T016 [US3] Verify `src/data.py` `get_evaluation_dataset()` uses `sorted(config.SUBJECTS)` (not `config.SUBJECTS` unsorted) and `range(n)` (not random sampling) to guarantee determinism; if unsorted, fix it

**Checkpoint**: Determinism holds for both IDs and per-subject distribution across repeated executions

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Document the certified stratification algorithm and edge-case behavior in `README.md`

- [ ] T017 [P] Add a "Stratified Sampling" subsection to `README.md` documenting that samples are fetched proportionally across subjects: quota = `max_samples // N` with remainder distributed to the first subjects
- [ ] T018 [P] In `README.md`, note the current 4-subject configuration yields exactly 25 samples per subject, and explain the shortfall behavior when a subject has fewer samples than its quota
- [ ] T019 [P] Add a "Certification" subsection to `README.md` summarizing how to run `python scripts/certify_stratified.py` to verify proportional allocation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup — must confirm quota math is correct before certifying user stories
- **User Stories (Phase 3–5)**: All depend on Foundational completion
  - US1 (P1) → US2 (P2) → US3 (P3)
- **Polish (Phase 6)**: Depends on all certification work being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational — certifies fresh-run stratification
- **User Story 2 (P2)**: Depends on US1 — resume logic builds on top of the same stratified selection
- **User Story 3 (P3)**: Depends on US1 — determinism is an attribute of the same selection function certified in US1

### Within Each User Story

- Certification scripts must assert expected behavior before marking tasks complete
- If any assertion fails, the underlying code in `src/` must be fixed first

---

## Parallel Execution Examples

```bash
# Phase 1 Setup tasks can run in parallel:
Task: "Verify pyproject.toml includes datasets dependency"
Task: "Create scripts/ directory if it does not exist"

# Phase 2 Foundational audit tasks can run in parallel:
Task: "Verify src/config.py get_subject_quotas() correctness"
Task: "Verify src/data.py select_stratified_samples() determinism"
Task: "Verify src/data.py get_evaluation_dataset() ordering"

# US1 parallel tasks:
Task: "Create scripts/certify_stratified.py with per-subject assertions"
Task: "Create scripts/certify_stratified.py shortfall edge-case handler"

# US3 parallel tasks:
Task: "Update scripts/certify_stratified.py to assert identical IDs across two runs"
Task: "Update scripts/certify_stratified.py to assert identical per-subject counts across two runs"

# Polish tasks can all run in parallel:
Task: "Add Stratified Sampling subsection to README.md"
Task: "Add shortfall behavior note to README.md"
Task: "Add Certification subsection to README.md"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. T001–T002: Ensure setup is ready
2. T003–T005: Audit and confirm proportional quota math in `src/config.py` and `src/data.py`
3. T006–T009: Create and run `scripts/certify_stratified.py` to empirically prove 25 samples per subject × 4 subjects = 100 total
4. **STOP and VALIDATE**: If `certify_stratified.py` fails, fix `src/data.py` or `src/config.py` before proceeding
5. Only after US1 is certified → move to US2, then US3

### Incremental Delivery

1. Complete Setup + Foundational → confirm existing code is mathematically correct
2. Certify US1 → Prove proportional stratified sampling works on fresh runs
3. Certify US2 → Prove resume logic respects the stratified cap
4. Certify US3 → Prove determinism holds across repeated fresh runs
5. Polish → Document findings in README.md

---

## Notes

- The proportional stratification algorithm is: `base = max_samples // num_subjects`, `remainder = max_samples % num_subjects`. Subject at index `i` receives `base + (1 if i < remainder else 0)` samples.
- Subjects are iterated in **alphabetical order** (`sorted(config.SUBJECTS)`) for determinism.
- No randomness is used; all selection uses deterministic integer slicing (`dataset.select(range(n))`).
- If a subject has fewer samples than its quota, the pipeline does **not** backfill — total fetched samples may be slightly below 100. This is by design per spec.md edge-case definitions.
- The certification script in `scripts/certify_stratified.py` is the primary deliverable for proving the user's requirement: *"if 4 subjects is selected, and max_sample = 100, it must have 25 samples for each subject."*
