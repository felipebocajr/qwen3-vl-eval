# Feature Specification: HF Qwen3-VL Model Initialization with Preserved Structured Generation

**Feature Branch**: `[005-hf-qwen3vl-model-init]`
**Created**: 2026-05-06
**Status**: Draft
**Input**: User description: "Rewrite the implementation plan for 004-hf-qwen3vl-refactor. CRITICAL ARCHITECTURAL CONSTRAINTS: 1. DO NOT remove outlines, pydantic, or the ModelResponse schema. We are strictly keeping Structured Generation. 2. DO NOT revert to regex or string parsing in parser.py. 3. DO NOT lower MAX_NEW_TOKENS or disable Chain of Thought (CoT) reasoning. THE ONLY ALLOWED CHANGES: 1. Update the model initialization logic in model.py to point exclusively to the HuggingFace model ID: Qwen/Qwen3-VL-2B-Instruct. 2. Adapt any underlying initialization code (like processor loading or Outlines wrapping) strictly to make this specific model ID work with our existing Outlines/Pydantic pipeline. Do not touch the rest of the architecture."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stable Model Loading with Structured Generation (Priority: P1)

As a researcher running the MMMU evaluation pipeline, I want the model loading component to exclusively use the canonical HuggingFace Qwen3-VL-2B-Instruct identifier so that the pipeline fetches and loads the correct model from the canonical source, while preserving the existing structured generation and schema validation that constrains outputs to valid JSON with reasoning and answer fields.

**Why this priority**: Correct model identification is foundational; without it, the pipeline loads the wrong weights or fails entirely. Maintaining structured generation ensures answers remain parseable and reasoning stays inspectable.

**Independent Test**: Can be tested by verifying the model downloads from the correct source and produces a valid structured response on a single sample.

**Acceptance Scenarios**:

1. **Given** a fresh environment with network access, **When** the pipeline initializes, **Then** it resolves and loads the model exclusively from the canonical HuggingFace model identifier without ambiguity.
2. **Given** a multiple-choice sample, **When** inference runs, **Then** the response conforms to the established structured output schema containing both step-by-step reasoning and a single-letter answer.

---

### User Story 2 - Answer Extraction Continuity (Priority: P2)

As a researcher, I want the answer extraction component to continue operating exactly as it currently does, using schema validation to pull the answer from structured JSON, so that I do not lose parsing reliability or reintroduce fragile string-matching heuristics.

**Why this priority**: The existing parser is validated and reliable; changing it risks regressions in accuracy metrics due to extraction errors rather than model capability.

**Independent Test**: Can be tested by passing a suite of valid and malformed JSON responses through the parser and asserting identical behavior before and after the change.

**Acceptance Scenarios**:

1. **Given** a well-formed JSON response containing a valid answer letter, **When** the parser processes it, **Then** the extraction succeeds and returns the correct letter.
2. **Given** a malformed or empty response, **When** the parser processes it, **Then** it returns the same failure indication as the current implementation without introducing new error modes.

---

### User Story 3 - Reasoning Chain Preservation (Priority: P3)

As a researcher analyzing model behavior, I want the pipeline to continue generating step-by-step reasoning chains alongside final answers without reducing the output length budget, so that I can audit how the model arrived at its choice.

**Why this priority**: Reasoning transparency is a core evaluation requirement; suppressing it would change the task from chain-of-thought evaluation to direct-answer evaluation, invalidating comparisons with prior results.

**Independent Test**: Can be tested by inspecting the structured output field for reasoning text across a batch of samples and confirming its presence.

**Acceptance Scenarios**:

1. **Given** a multiple-choice question, **When** inference completes, **Then** the structured output contains a non-empty reasoning field in addition to the answer field.
2. **Given** a long or complex question, **When** inference runs, **Then** the output is not truncated due to an artificially reduced length budget.

### Edge Cases

- What happens when the canonical model identifier is temporarily unavailable? The pipeline should surface a clear error and halt rather than falling back to an alternative identifier.
- How does the system handle a structured generation failure? The pipeline should log the specific sample and continue with the remaining samples.
- What if the model initialization wrapping requires adjustments for compatibility? The system should isolate any adapter changes to the initialization layer without propagating to downstream components.
- What happens if the response JSON is truncated mid-stream by the generation framework? The parser should attempt partial extraction rather than marking the sample as completely unparseable.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The model loading component MUST exclusively resolve to the HuggingFace model identifier `Qwen/Qwen3-VL-2B-Instruct`.
- **FR-002**: Any initialization adaptations required for compatibility with the structured generation framework MUST be confined to the model initialization layer and MUST NOT alter downstream pipeline logic.
- **FR-003**: The structured output schema used for generation MUST be preserved unchanged, continuing to enforce both a reasoning field and a single-letter answer field.
- **FR-004**: The answer extraction component MUST continue to use the existing structured validation approach and MUST NOT revert to string-pattern or regular-expression matching.
- **FR-005**: The maximum output length configuration MUST NOT be reduced from its current value.
- **FR-006**: Step-by-step reasoning generation MUST remain enabled in the inference prompt and schema.
- **FR-007**: Existing pipeline behaviors—resumability via trajectory records, stratified sampling, per-sample structured output, and aggregate summary generation—MUST remain unaffected.
- **FR-008**: Processor loading logic MUST work in concert with the specified model identifier to correctly prepare multimodal inputs.

### Key Entities

- **StructuredOutputSchema**: The validation contract that enforces reasoning and answer fields in model responses.
- **ModelConfiguration**: Parameters governing which model identifier to load, including device placement and precision settings.
- **EvaluationRecord**: A per-sample trajectory containing metadata, prompt, raw response, extracted answer, correctness flag, and timing.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The pipeline resolves and loads the correct canonical model identifier on the first attempt in a clean environment.
- **SC-002**: Structured answer extraction success rate across 100 evaluation samples remains at or above the pre-change baseline.
- **SC-003**: At least 95% of valid responses contain a non-empty reasoning field.
- **SC-004**: Zero downstream pipeline modules require modification to accommodate the model initialization change.
- **SC-005**: Resumability functions correctly when the pipeline is interrupted and restarted after partial completion.

## Assumptions

- The structured generation framework supports the Qwen3-VL architecture without requiring version updates.
- Network connectivity to the HuggingFace Hub is available during initial model download.
- Target hardware has sufficient memory to load the Qwen3-VL-2B-Instruct weights.
- Existing dependency versions in the virtual environment are compatible with this model identifier.
- The current dependency manifest already includes the necessary core libraries; no additional packages are required.
- The model identifier is hardcoded to prevent accidental substitution during configuration.
