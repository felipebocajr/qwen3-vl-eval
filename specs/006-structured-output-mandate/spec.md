# Feature Specification: Constitution Structured Output Mandate

**Feature Branch**: `006-structured-output-mandate`  
**Created**: 2026-05-06  
**Status**: Draft  
**Input**: User description: "I need to update our project constitution (constitution.md). Please add a new strict architectural rule under the appropriate section: All AI model generation MUST utilize Structured Outputs (e.g., Pydantic schemas via Outlines). Native Hugging Face string generation and Regex-based text parsing are strictly forbidden. Please update the Constitution Check gates so that any future plans that attempt to introduce regex or remove Pydantic will automatically fail the gate."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Enforce structured generation in existing pipeline (Priority: P1)

As a developer maintaining the evaluation pipeline, I want all model inference to be constrained by Pydantic schemas so that answers are deterministic, type-safe, and extractable without regex or brittle string parsing.

**Why this priority**: This eliminates an entire class of parsing bugs and makes the pipeline auditable. The current codebase already uses Outlines; the constitution must codify this as a non-negotiable rule so that future refactors do not inadvertently revert to raw string generation.

**Independent Test**: A code-review script or CI gate scans `src/` for any call to `model.generate()` or `tokenizer.batch_decode()` followed by `re.search()` or `re.match()`. If found, the gate fails.

**Acceptance Scenarios**:

1. **Given** a PR that adds `re.search(r'Answer: ([A-E])', raw_output)` in any module, **When** the Constitution Check gate runs, **Then** the gate fails and the PR is blocked.
2. **Given** a PR that replaces the Outlines generator with native `model.generate()` for answer extraction, **When** the Constitution Check gate runs, **Then** the gate fails and the PR is blocked.
3. **Given** a PR that adds new Pydantic schemas and uses them with Outlines, **When** the Constitution Check gate runs, **Then** the gate passes.

---

### User Story 2 - Document the mandate for new contributors (Priority: P2)

As a new engineer joining the project, I want the constitution to explicitly state the structured-output rule so that I do not accidentally write regex-based parsing code.

**Why this priority**: Prevents future regressions without requiring tribal knowledge. Written rules are enforceable in review checklists.

**Independent Test**: A new contributor reads `constitution.md` and can state the rule without asking a senior engineer.

**Acceptance Scenarios**:

1. **Given** `constitution.md` after the amendment, **When** a contributor reads section 3 (Architecture & Modularity) or 6 (Constitution Check Gates), **Then** they find an explicit prohibition on regex-based answer extraction and raw HF string generation.

---

### User Story 3 - Automated gate blocks non-compliant specs (Priority: P2)

As a tech lead, I want the Constitution Check gate to scan incoming `spec.md` and `plan.md` files for forbidden keywords so that non-compliant features are rejected before implementation begins.

**Why this priority**: Shifting compliance left to the specification phase is cheaper than catching violations during code review or in production.

**Independent Test**: A spec that proposes "use regex to extract the final answer" is flagged by the Constitution Check gate during `/speckit.plan` or `/speckit.checklist`.

**Acceptance Scenarios**:

1. **Given** a new spec containing the phrase "implement regex-based parsing", **When** the Constitution Check gate runs, **Then** the gate fails with a message citing the structured-output rule.
2. **Given** a new spec containing the phrase "remove Pydantic schemas" or "drop Outlines dependency", **When** the Constitution Check gate runs, **Then** the gate fails with a message citing the structured-output rule.

### Edge Cases

- What happens when a plan uses `re` for a non-answer-extraction purpose (e.g., sanitizing filenames)? The gate should permit `re` usage outside of model-output parsing; the prohibition targets answer extraction and generation pathways specifically.
- How does the gate handle plans that mention "fallback to regex if structured generation fails"? This should be treated as a violation because it introduces regex as an answer-extraction path.
- What if a plan proposes adding `outlines` but also keeping a legacy regex parser for backward compatibility? The gate must flag this as non-compliant; backward compatibility exceptions must be approved via a constitution amendment, not a feature plan.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The constitution MUST contain an explicit architectural rule stating that all AI model generation MUST use structured outputs (Pydantic schemas via Outlines or equivalent constrained generation).
- **FR-002**: The constitution MUST explicitly forbid native Hugging Face string generation and regex-based text parsing for answer extraction from model outputs.
- **FR-003**: The constitution MUST include a "Constitution Check Gates" section that defines automated validation criteria.
- **FR-004**: The Constitution Check gate MUST scan incoming specs and plans for forbidden keywords related to regex-based parsing and removal of structured-output dependencies.
- **FR-005**: The Constitution Check gate MUST output a clear failure message citing the specific constitutional rule that was violated.
- **FR-006**: The Constitution Check gate MUST permit `re` usage for non-answer-extraction purposes (e.g., input sanitization, path manipulation) provided it does not touch model-output parsing or generation pathways.
- **FR-007**: The constitution amendment MUST be versioned and dated.

### Key Entities

- **Constitution**: The living document at `.specify/memory/constitution.md` that governs all project architecture decisions.
- **Constitution Check Gate**: An automated script (`scripts/constitution-check.sh` or equivalent) that validates specs, plans, and code against constitutional rules.
- **Forbidden Pattern Registry**: A documented list of regexes and keywords (e.g., `re.search`, `re.match`, `batch_decode`, `remove.*outlines`, `drop.*pydantic`) that the gate uses for scanning.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of new specs and plans created after the amendment pass through the Constitution Check gate; zero non-compliant specs reach the implementation phase.
- **SC-002**: Any PR introducing regex-based answer extraction or raw HF string generation is blocked by the gate within 30 seconds of CI start.
- **SC-003**: New contributors can identify the structured-output rule by reading `constitution.md` in under 60 seconds.
- **SC-004**: Existing `src/` codebase scan shows zero violations of the structured-output rule at the time of amendment ratification.

## Assumptions

- The project already uses Outlines with Pydantic schemas (`src/schemas.py`, `src/model.py`), so the new rule formalizes current practice rather than introducing a new technology.
- The Constitution Check gate will be implemented as a shell script or Python script that can be run in CI or locally by maintainers.
- The speckit workflow (`/speckit.specify`, `/speckit.plan`, `/speckit.tasks`) is the primary vehicle for creating new features, so the gate integrates at the spec/plan stage.
- The `re` standard library module is permitted for non-model-parsing use cases; the prohibition is scoped to answer extraction and generation pathways.
