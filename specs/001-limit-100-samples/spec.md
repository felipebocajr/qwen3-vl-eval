# Feature Specification: Limit Pipeline to 100 Samples

**Feature Branch**: `001-limit-100-samples`
**Created**: 2026-05-05
**Status**: Draft
**Input**: User description: "make sure that the pipeline only fetcher 100 total samples from the dataset."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Fresh Run Fetches Exactly 100 Samples (Priority: P1)

A researcher starts the evaluation pipeline for the first time. The pipeline loads the MMMU dataset, selects 100 samples, and runs inference on each one. Once 100 samples are queued, no additional samples are fetched or processed.

**Why this priority**: This is the core requirement. Without the cap, the pipeline could run indefinitely or process an unbounded number of samples, making results incomparable and wasting compute.

**Independent Test**: Can be fully tested by deleting `trajectories.jsonl` and `summary.json`, running the pipeline, and verifying that `trajectories.jsonl` contains at most 100 unique sample IDs upon completion.

**Acceptance Scenarios**:

1. **Given** no `trajectories.jsonl` exists, **When** the pipeline is executed, **Then** it loads exactly 100 unique samples from the MMMU dataset and processes each one.
2. **Given** the MMMU dataset has at least 100 samples, **When** the pipeline completes a fresh run, **Then** `trajectories.jsonl` contains exactly 100 records with unique sample IDs.
3. **Given** the MMMU dataset has fewer than 100 samples, **When** the pipeline is executed, **Then** it loads and processes all available samples and stops.

---

### User Story 2 - Resumed Run Respects the 100-Sample Cap (Priority: P2)

A researcher interrupts the pipeline after 40 samples and restarts it. The pipeline reads the 40 already-completed samples from `trajectories.jsonl`, then fetches and processes the remaining 60 samples to reach the 100-sample cap. It does not exceed 100 total unique samples across the original and resumed runs combined.

**Why this priority**: Resume behavior is a stated design principle of the pipeline. The cap must work correctly with resume logic to avoid duplicate or excess work.

**Independent Test**: Can be fully tested by running the pipeline for a partial set of samples, interrupting it, resuming, and verifying that the total number of unique sample IDs in `trajectories.jsonl` never exceeds 100.

**Acceptance Scenarios**:

1. **Given** `trajectories.jsonl` contains 40 completed samples, **When** the pipeline is restarted, **Then** it processes at most 60 additional unique samples.
2. **Given** a resumed run completes successfully, **When** the total count is tallied, **Then** the combined unique sample IDs across the partial and resumed runs equal exactly 100 (or the original cap).
3. **Given** `trajectories.jsonl` already contains 100 or more unique sample IDs from a prior run, **When** the pipeline is restarted, **Then** it fetches zero new samples and exits gracefully.

---

### User Story 3 - Deterministic Sample Selection (Priority: P3)

A researcher reruns the evaluation from scratch after clearing output files. The same 100 samples are selected as in the previous run, ensuring reproducibility and consistent metrics.

**Why this priority**: Determinism ensures that results are comparable across runs and that debugging or reruns target the same data.

**Independent Test**: Can be fully tested by running the pipeline twice from a clean state and comparing the sample IDs in the two resulting `trajectories.jsonl` files.

**Acceptance Scenarios**:

1. **Given** two fresh runs with identical configuration, **When** sample selection is performed, **Then** the same 100 sample IDs are chosen in both runs.

---

### Edge Cases

- What happens if the MMMU dataset contains fewer than 100 samples?
  → The pipeline should process all available samples and stop without error.
- What happens if a subject has fewer samples than its proportional quota?
  → The pipeline takes all available samples from that subject and does not backfill from other subjects, so total may be slightly below 100 in that case.
- What happens if `trajectories.jsonl` from a prior buggy run contains more than 100 unique sample IDs?
  → The pipeline should detect the over-cap state, log a warning, and exit without fetching new samples.
- What happens if some of the 100 selected samples error out during inference?
  → The error is recorded in `trajectories.jsonl` as per existing graceful-failure logic, but the sample still counts toward the 100-sample cap (it was fetched and attempted).
- What happens if the user modifies `trajectories.jsonl` manually, removing some records?
  → The pipeline treats missing records as not-yet-completed and refills up to the cap.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST enforce a maximum of 100 unique samples fetched from the MMMU dataset across any execution (fresh or resumed).
- **FR-002**: On a fresh run (no prior `trajectories.jsonl`), the pipeline MUST select exactly 100 samples from the dataset using **stratified sampling across subjects** with proportional allocation, or fewer if the dataset contains fewer than 100 samples total.
- **FR-003**: On a resumed run, the pipeline MUST read the set of already-completed sample IDs from `trajectories.jsonl` and only fetch additional samples up to the 100-sample cap.
- **FR-004**: The sample selection logic MUST be deterministic so that the same 100 stratified samples are targeted on every fresh start with identical configuration.
- **FR-005**: The pipeline MUST write `summary.json` based only on the intended 100 (or fewer) samples, regardless of how many were successfully processed versus errored.
- **FR-006**: The pipeline MUST exit gracefully once the 100-sample cap is reached, without attempting to load or process additional samples.
- **FR-007**: If `trajectories.jsonl` already contains 100 or more unique sample IDs, the pipeline MUST log an informative message and exit immediately without fetching new samples.

### Key Entities *(include if feature involves data)*

- **Sample**: A single question instance from the MMMU dataset. Key attributes: `id`, `subject`, `question`, `choices`, `answer`.
- **Evaluation Run**: A single invocation of the pipeline. It may be fresh or resumed, and it is bounded by the 100-sample cap.
- **Completion Record**: A line in `trajectories.jsonl` representing a sample that has been fetched and attempted, whether successful or errored.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A fresh pipeline run processes no more than 100 unique samples (or the full dataset if smaller), with sample counts balanced proportionally across selected subjects.
- **SC-002**: A resumed pipeline run, combined with prior runs, processes no more than 100 unique samples in total.
- **SC-003**: The `trajectories.jsonl` file never contains more than 100 unique sample IDs for a given evaluation configuration after a compliant run.
- **SC-004**: Two fresh runs with identical configuration select the exact same set of 100 stratified sample IDs, and at least 90 % of subjects are represented in the selected set.
- **SC-005**: The pipeline completes and exits within 10 seconds of detecting the cap has been reached, with no additional data loading or inference attempted.

## Assumptions

- The 100-sample cap is a hard upper bound; stratification across MMMU subjects is applied **proportionally** within this bound (e.g. equal split per subject when counts allow).
- The pipeline already has resume logic that tracks completed samples in `trajectories.jsonl`; this feature only adds the cap enforcement layer.
- Existing records in `trajectories.jsonl` from prior runs are treated as completed samples toward the cap, regardless of their `is_correct` or `error` values.
- The dataset loader can return the full MMMU dataset, and a deterministic slice or index-based selection is sufficient to pick the 100 samples.
- The cap is configurable in principle, but the default and required behavior for this project is exactly 100.
