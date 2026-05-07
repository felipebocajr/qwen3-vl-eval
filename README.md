# Qwen3-VL MMMU Evaluation Pipeline

Evaluation pipeline for running **Qwen3-VL-2B-Instruct** locally on 100 samples from the **MMMU** (Massive Multi-discipline Multimodal Understanding) dataset.

## Features

- **Structured Generation**: Every model response is constrained at decode time by a Pydantic schema (`{"reasoning": "...", "answer": "A|B|C|D"}`) using the Outlines library, guaranteeing parseable JSON output.
- **100-sample cap**: The pipeline evaluates at most 100 unique MMMU samples per run.
- **Stratified sampling**: Samples are selected proportionally across subjects (e.g., ~25 from each of 4 subjects).
- **Resumable**: If interrupted, restarting skips already-completed samples and continues up to the cap.
- **Deterministic**: The same 100 stratified samples are selected on every fresh start, ensuring reproducible results.
- **Graceful failures**: Individual sample errors are logged without halting the pipeline.
- **Chain-of-Thought reasoning**: The prompt instructs the model to provide step-by-step reasoning before emitting the final structured answer, with `MAX_NEW_TOKENS = 2048` to prevent truncation.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Install dependencies with uv
uv sync
```

The pipeline uses **stratified sampling** to ensure balanced representation across MMMU subjects. For example, with 4 subjects and a 100-sample cap, the pipeline selects approximately 25 samples from each subject deterministically. Any shortfall in a subject (if it has fewer samples than its quota) is not backfilled from other subjects, so the total may be slightly below 100 in edge cases.

## Usage

```bash
# Full evaluation (up to 100 samples)
python -m src.main

# Results are written to:
#   results/trajectories.jsonl   — one JSON record per sample
#   results/summary.json          — aggregate metrics
```

### Resume a partial run

Simply rerun `python -m src.main`. The pipeline reads `results/trajectories.jsonl`, skips completed sample IDs, and evaluates the remaining samples up to the 100-sample cap.

## Project Structure

```bash
# Verify deterministic & stratified selection
python scripts/verify_determinism.py
```

- `src/config.py` — Cap value, paths, subject list, quota helper, and `MAX_NEW_TOKENS = 2048`
- `src/data.py` — MMMU dataset loading and **stratified** deterministic sample selection
- `src/model.py` — Qwen3-VL model loading and **Outlines structured generation** inference
- `src/parser.py` — Answer extraction via **Pydantic JSON validation** (no regex)
- `src/pipeline.py` — Per-sample evaluation and trajectory persistence
- `src/metrics.py` — Accuracy, per-subject stats, and summary generation
- `src/main.py` — Orchestration loop with resume logic
- `src/schemas.py` — Pydantic `ModelResponse` schema with `reasoning` and `answer` fields

## Architecture

The diagram below shows the full data flow from MMMU dataset loading through structured generation to `trajectories.jsonl`.

```mermaid
flowchart TB
    subgraph Entry["Entry Point"]
        MAIN["main.py"]
    end

    subgraph Data["src/data.py — Data Loading"]
        LOAD_DS["load_dataset MMMU/MMMU"]
        SUB1["Accounting (25)"]
        SUB2["Architecture and Engineering (25)"]
        SUB3["Art (25)"]
        SUB4["Biology (25)"]
        CONCAT["concatenate_datasets"]
        EVAL_DS["Evaluation Dataset (100 samples)"]

        LOAD_DS --> SUB1 --> CONCAT
        LOAD_DS --> SUB2 --> CONCAT
        LOAD_DS --> SUB3 --> CONCAT
        LOAD_DS --> SUB4 --> CONCAT
        CONCAT --> EVAL_DS
    end

    subgraph Model["src/model.py — Model Initialization"]
        LOAD_MODEL["from_pretrained Qwen3-VL-2B-Instruct"]
        PROC["AutoProcessor"]
        DEVICE["device_map='auto' (cuda or cpu)"]
        CACHED["_generator_cache: Outlines Generator + ModelResponse schema"]

        LOAD_MODEL --> PROC --> DEVICE --> CACHED
    end

    subgraph Pipeline["src/pipeline.py — Evaluation Loop"]
        direction TB
        FOR["for sample in EVAL_DS"]
        EX_IMG["extract_images: PIL images from image_1..7"]
        FIX_FMT["Preserve PIL format after RGB convert"]
        STRIP["strip_image_tags: remove image_N placeholders"]
        PARSE_OPT["parse_options: ast.literal_eval"]
        BUILD["build_prompt: Question + choices + JSON schema hint"]
        INFER_SUBJ["infer_subject: from sample_id"]
        INIT_REC["Initialize record dict with defaults"]
        TRY["try block"]
        CALL_INF["run_inference"]
        TIMER["time.perf_counter"]
        PARSE["extract_answer"]
        SAVE["save_trajectory"]
        INDIV["Write results/sample_id.json"]
        JSONL["Append to trajectories.jsonl"]
        CATCH["except Exception"]
        ERR_LOG["record.error = str(exc)"]

        FOR --> EX_IMG --> FIX_FMT --> STRIP --> PARSE_OPT --> BUILD
        BUILD --> INFER_SUBJ --> INIT_REC --> TRY
        TRY --> CALL_INF
        CALL_INF --> TIMER --> PARSE --> SAVE
        SAVE --> INDIV --> JSONL
        TRY -.-> CATCH --> ERR_LOG --> SAVE
    end

    subgraph Inference["src/model.py — Structured Generation"]
        direction TB
        GET_GEN["_get_cached_generator"]
        CHAT["inputs.Chat"]
        SYS_MSG["add_system_message: JSON schema instruction"]
        USER_MSG["add_user_message: images + prompt"]
        GEN_CALL["generator: max_new_tokens, do_sample=False"]
        RAW_JSON["Raw JSON string (ModelResponse)"]

        GET_GEN --> CHAT --> SYS_MSG --> USER_MSG --> GEN_CALL --> RAW_JSON
    end

    subgraph Parser["src/parser.py — Answer Extraction"]
        direction TB
        STRIP_MD["_strip_markdown_fences"]
        JSON_LOAD["json.loads"]
        PYDANTIC["ModelResponse.model_validate"]
        EXTRACT["Extract answer field A/B/C/D"]
        FALLBACK["Best-effort fallback heuristics"]
        FAIL["Return (None, False)"]

        STRIP_MD --> JSON_LOAD --> PYDANTIC --> EXTRACT
        JSON_LOAD -.-> FALLBACK -.-> FAIL
    end

    subgraph Output["results/ — Output Files"]
        TRAJ_FILE["trajectories.jsonl (100 lines)"]
        IND_FILES["sample_id.json (100 files)"]
    end

    subgraph Metrics["src/metrics.py — Aggregation (unimplemented)"]
        STUB["Empty stub (0 lines)"]
        SUMM["summary.json (planned)"]
    end

    MAIN --> Data
    MAIN --> Model
    EVAL_DS -->|yield| Pipeline
    CACHED -->|reuse| Inference
    CALL_INF -.->|calls| Inference
    Inference --> RAW_JSON
    RAW_JSON --> Parser
    Parser -->|extracted_answer| Pipeline
    SAVE -->|writes| Output
    TRAJ_FILE -.->|future| Metrics
    Metrics -.-> SUMM
```

## Structured Generation

The pipeline uses **Outlines** to enforce a Pydantic schema at every decoding step. This means the model is physically incapable of emitting invalid JSON or answers outside the allowed set (A, B, C, D).

### How it works

1. **Schema definition** (`src/schemas.py`): A `ModelResponse` Pydantic model defines two fields:
   - `reasoning`: free-form text for step-by-step chain-of-thought
   - `answer`: constrained to exactly one of `A`, `B`, `C`, `D`

2. **Constrained generation** (`src/model.py`): The Outlines `Generator` wraps the Qwen3-VL model with a logits processor that only allows tokens conforming to the schema at each decoding step.

3. **Deterministic parsing** (`src/parser.py`): Raw responses are parsed with `json.loads()` + `ModelResponse.model_validate()`. There is **zero regex-based extraction** — either the response validates against the schema or it is flagged as a parse failure.

4. **Prompt design** (`src/pipeline.py`): The evaluation prompt explicitly instructs the model to think step by step and then emit ONLY valid JSON matching the schema.

### Verification

To confirm the pipeline is using structured generation:

```bash
# Check parser has no regex answer extraction
grep -E "re\.(search|match|findall|compile)" src/parser.py
# Expected: no output

# Check model uses Outlines Generator
grep "Generator" src/model.py
# Expected: lines referencing Generator with output_type=ModelResponse

# Check prompt requests reasoning + JSON
grep "Think step by step" src/pipeline.py
# Expected: the prompt line
```

```bash
python scripts/verify_determinism.py
```

The pipeline uses **stratified sampling** to ensure balanced representation across MMMU subjects. For example, with 4 subjects and a 100-sample cap, the pipeline selects approximately 25 samples from each subject deterministically. Any shortfall in a subject (if it has fewer samples than its quota) is not backfilled from other subjects, so the total may be slightly below 100 in edge cases.

## License

MIT
