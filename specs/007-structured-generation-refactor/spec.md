# Feature Specification: Structured Generation Refactor for Evaluation Pipeline

**Feature Branch**: `[006-structured-output-mandate]`
**Created**: 2026-05-06
**Status**: Draft
**Input**: User description: "Refactor the evaluation pipeline to use Structured Generation with the HuggingFace model Qwen/Qwen3-VL-2B-Instruct directly on the current main branch. Implement architectural updates: model initialization with constrained generation wrapping, a structured response schema with reasoning and answer fields, parser cleanup to remove regex-based extraction, configuration for extended output length to support step-by-step reasoning, and prompt update to instruct the model to provide reasoning followed by a final structured answer."

## User Scenarios & Testing *(mandatory)*

### User Story 1 – Deterministic Structured Inference (Priority: P1)

As a researcher evaluating vision-language models on MMMU, I want the inference layer to produce outputs that are automatically validated against a fixed schema containing both reasoning and a single-letter answer, so that every response is guaranteed to be parseable and auditable without relying on fragile text-pattern matching.

**Why this priority**: Unstructured text generation produces inconsistent formatting that breaks downstream parsing and artificially deflates accuracy metrics. Schema-constrained generation eliminates entire classes of parsing failures.

**Independent Test**: Run a single evaluation sample and verify the raw response is a structured object containing a reasoning field and an answer field with value A, B, C, or D.

**Acceptance Scenarios**:

1. **Given** the evaluation pipeline is initialized, **When** a sample is processed, **Then** the model response is automatically constrained to a schema that requires a reasoning string and an answer letter.
2. **Given** a multiple-choice question with four options, **When** inference completes, **Then** the extracted answer is exactly one of A, B, C, or D and the reasoning field is non-empty.
3. **Given** the pipeline is restarted after interruption, **When** it resumes from the last completed sample, **Then** the structured generation behavior remains identical and no completed samples are re-evaluated.

---

### User Story 2 – Reliable Answer Extraction (Priority: P2)

As a researcher, I want the answer extraction component to validate model outputs using the same structured schema that generated them, so that parsing is deterministic and failures are explicit rather than silent misclassifications caused by regex heuristics.

**Why this priority**: Regex-based extraction is brittle across different phrasing, spacing, and formatting. Schema validation guarantees that only well-formed responses are accepted, making failure analysis straightforward.

**Independent Test**: Feed a suite of valid and malformed structured responses through the parser and assert correct extraction and explicit failure flags.

**Acceptance Scenarios**:

1. **Given** a well-formed structured response containing a valid answer letter, **When** the parser processes it, **Then** extraction succeeds and returns the correct letter with a success flag.
2. **Given** a malformed or incomplete response that does not conform to the schema, **When** the parser processes it, **Then** extraction fails with an explicit error indication and no guesswork is applied.
3. **Given** a response with an answer outside the valid set (e.g., "E"), **When** the parser processes it, **Then** validation rejects it and marks extraction as failed.

---

### User Story 3 – Transparent Chain-of-Thought Reasoning (Priority: P3)

As a researcher analyzing model behavior, I want the pipeline to allocate sufficient output capacity for the model to generate step-by-step reasoning before emitting its final answer, so that I can audit how the model arrived at its choice and compare reasoning quality across samples.

**Why this priority**: Reasoning transparency is essential for understanding model failures. Truncating the output budget would suppress the chain of thought and change the evaluation from a reasoning task to a direct-answer task, invalidating historical comparisons.

**Independent Test**: Inspect the structured output field for reasoning text across a batch of samples and confirm its presence and non-trivial length.

**Acceptance Scenarios**:

1. **Given** a complex multiple-choice question, **When** inference completes, **Then** the structured output contains a non-empty reasoning field explaining the model's step-by-step thinking.
2. **Given** a long or multi-step question, **When** inference runs, **Then** the output is not truncated before the reasoning and answer are both fully emitted.
3. **Given** the evaluation prompt, **When** it is constructed for any sample, **Then** it explicitly instructs the model to provide reasoning before the final answer.

### Edge Cases

- What happens if the structured generation library cannot wrap the target model? The pipeline should surface a clear initialization error and halt before attempting inference.
- How does the system handle a response that is structurally valid but contains an empty reasoning field? The parser should accept it (mark extraction as succeeded) but the researcher can filter on reasoning length during analysis.
- What if the model produces reasoning that exceeds the output budget before reaching the answer? The pipeline should log the truncation and continue; the answer may be missing, which the parser flags as a failure.
- What if a sample has no associated images (text-only question)? The prompt builder should gracefully omit image content from the structured input without breaking schema generation.
- How does the system handle resuming when the output format changes? Existing trajectory files with the old raw format must not cause crashes; the pipeline should skip already-completed samples regardless of their legacy format.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The inference layer MUST load the canonical vision-language model from the standard model repository using the exact identifier provided by the project constitution.
- **FR-002**: The inference layer MUST wrap the loaded model with a constrained generation mechanism that enforces every output to conform to a predefined structured schema before the token is emitted.
- **FR-003**: The structured response schema MUST contain exactly two fields: a reasoning field (free-form text for step-by-step thinking) and an answer field (constrained to exactly one of four valid letters).
- **FR-004**: The answer extraction component MUST validate raw model responses using the same structured schema used for generation, and MUST NOT rely on pattern matching, regular expressions, or string heuristics for answer extraction.
- **FR-005**: The maximum output length configuration MUST be set high enough to accommodate both multi-sentence reasoning chains and the final structured answer without truncation.
- **FR-006**: The evaluation prompt MUST instruct the model to generate reasoning followed by a structured answer object, and MUST remain compatible with the constrained generation wrapper.
- **FR-007**: The refactor MUST preserve existing pipeline behaviors: resumability via sample-level trajectory records, stratified sampling from the evaluation dataset, per-sample structured output files, and aggregate summary generation.
- **FR-008**: Processor loading logic MUST correctly prepare multimodal inputs for the target model so that both text and images are passed into the constrained generation path.

### Key Entities

- **StructuredResponse**: The validation contract that enforces a reasoning text field and a single-letter answer field in every model response.
- **ModelConfiguration**: Parameters governing which model to load, including device placement, precision settings, and the constrained generation wrapper.
- **EvaluationRecord**: A per-sample trajectory containing metadata, prompt, raw structured response, extracted answer, correctness flag, inference time, and any error message.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: At least 95% of model raw responses across 100 evaluation samples are parseable into the structured schema on the first parsing attempt.
- **SC-002**: Parse failure rate across the full 100-sample run does not exceed 5%, and zero failures are caused by regex or string-heuristic misclassification.
- **SC-003**: At least 90% of valid responses contain a non-empty reasoning field with at least 20 characters of text.
- **SC-004**: Zero samples are re-evaluated when the pipeline is restarted after completing N samples; resumability must remain intact.
- **SC-005**: Average inference time per sample does not increase by more than 20% compared to the pre-refactor baseline (measured on the same hardware).

## Assumptions

- The constrained generation framework is compatible with the target vision-language model architecture and its multimodal input format.
- Network connectivity to the model repository is available during initial download; weights are cached afterward.
- Target hardware has sufficient resources to load the model in reduced precision with automatic device placement (fallback to CPU is acceptable).
- The evaluation dataset format remains unchanged (fields: identifier, question, options, answer, and up to seven associated images).
- Images are provided as standard image objects from the datasets library and require color-space conversion before being passed to the model.
- The pipeline is run locally with no remote API or cloud inference.
- The maximum sample count (100) and stratified sampling strategy are unchanged.
