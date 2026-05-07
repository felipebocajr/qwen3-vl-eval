# Qwen3-VL MMMU Evaluation Pipeline

A local evaluation pipeline that loads **Qwen3-VL-2B-Instruct**, runs structured inference on **100 stratified samples** from the **MMMU** (Massive Multi-discipline Multimodal Understanding) dataset, and produces per-sample trajectories, aggregate metrics, and visualization charts.

## Features

- **Structured Generation**: Every model response is constrained at decode time by a Pydantic schema (`{"reasoning": "...", "answer": "A|B|C|D"}`) using the Outlines library, guaranteeing parseable JSON output.
- **Model Adapter Architecture**: The inference backend is decoupled behind a `BaseVLMAdapter` interface with a decorator-based registry. Swapping models (e.g., ChatGPT, Claude, another local VLM) requires only a new adapter file + one import + one string change in `main.py`.
- **Three-Stage Extraction Cascade**: If the primary Pydantic/JSON extraction fails, the pipeline retries inference with **2× the token budget**. If that also fails, a regex fallback attempts recovery as a last resort.
- **100-sample stratified cap**: Samples are selected proportionally across subjects (e.g., 25 from each of 4 subjects).
- **Resumable**: If interrupted, restarting skips already-completed samples by reading `trajectories.jsonl` and continues from where it stopped.
- **Deterministic**: The same 100 stratified samples are selected on every fresh start, ensuring reproducible results.
- **Graceful degradation**: Individual sample errors are logged without halting the pipeline. Every attempted sample generates a trajectory entry.
- **Memory management**: PIL images are explicitly closed after each sample, and PyTorch CUDA cache is purged after every generation to prevent OOM.

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Install dependencies with uv
uv sync
```

**Requirements**: Python 3.12, CUDA-capable GPU recommended (CPU fallback works but is slow).

## Usage

```bash
# Full evaluation (up to 100 samples)
uv run python -m src.main

# Results are written to results/:
#   trajectories.jsonl      — one JSON record per sample
#   summary.json             — aggregate metrics
#   accuracy_chart.png       — overall & per-subject accuracy bar chart
#   runtime_metrics.png      — runtime & quality dashboard
#   fallback_analysis.png    — extraction cascade breakdown
```

### Resume a partial run

Simply rerun the pipeline. It reads `results/trajectories.jsonl`, skips completed sample IDs, and evaluates the remaining samples up to the cap.

## Project Structure

```
src/
├── __init__.py                # Package marker
├── config.py                  # Constants: cap, paths, DEFAULT_MODEL_ID, MAX_NEW_TOKENS
├── data.py                    # Stratified MMMU dataset loading
├── schemas.py                 # Pydantic schemas for structured output (Outlines)
├── parser.py                  # Answer extraction (Pydantic JSON + regex fallback)
├── pipeline.py                # Per-sample evaluation, resume logic, extraction cascade
├── main.py                    # Orchestration & entrypoint
├── metrics.py                 # Aggregate summary computation
├── visualization.py           # Matplotlib charts (accuracy, runtime, fallback)
├── adapters/
│   ├── __init__.py            # Factory: get_model_adapter(provider_name)
│   ├── base.py                # BaseVLMAdapter ABC + MODEL_REGISTRY + @register_adapter
│   ├── qwen.py                # QwenAdapter (Qwen3-VL-2B-Instruct, Outlines)
│   └── _template.py           # Skeleton for adding new model adapters
└── ...
```

## Architecture

### Adapter & Registry Pattern

Models are loaded through a self-registering adapter system:

```python
from src.adapters import get_model_adapter

adapter = get_model_adapter("qwen_local")       # loads Qwen3-VL-2B-Instruct
answer = adapter.generate_answer(prompt, images, max_new_tokens=512)
```

Adding a new model (e.g., Claude, ChatGPT) only requires creating a new adapter file with a `@register_adapter("name")` decorator, importing it in `__init__.py`, and changing the provider name in `main.py`. The pipeline loop and parser are untouched.

### Swapping the Model

The pipeline uses a provider name string (`"qwen_local"`) to select the adapter. To use a different model, follow the three steps below. A ready-to-fill template is available at `src/adapters/_template.py`.

#### Step 1 — Create the adapter file

Copy the template and fill in model-specific logic. Below is a complete example for **OpenAI (ChatGPT)**:

```python
# src/adapters/openai.py
from openai import OpenAI
from PIL import Image

from src.adapters.base import BaseVLMAdapter, register_adapter


@register_adapter("openai_api")
class OpenAIAdapter(BaseVLMAdapter):
    """ChatGPT adapter via OpenAI API."""

    def __init__(self, model_id: str | None = None, **kwargs):
        super().__init__()
        self._model_id = model_id or "gpt-4o"
        self._client = OpenAI()  # reads OPENAI_API_KEY from env

    def generate_answer(self, prompt: str, images: list[Image.Image], **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=self._model_id,
            max_tokens=kwargs.get("max_new_tokens", 512),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

For a **local Hugging Face model**, follow the same pattern as `src/adapters/qwen.py` — use `from_pretrained()` to load the model and processor, optionally build an Outlines Generator for structured output.

#### Step 2 — Import the adapter

Add one line to `src/adapters/__init__.py`:

```python
from src.adapters import qwen    # noqa: F401   ← existing
from src.adapters import openai  # noqa: F401   ← add this
```

The import triggers the `@register_adapter` decorator, registering `"openai_api"` in the registry automatically.

#### Step 3 — Switch the provider name

Change one string in `src/main.py`:

```python
# adapter = get_model_adapter("qwen_local")   ← before
adapter = get_model_adapter("openai_api")       ← after
```

That's it. Re-run `uv run python -m src.main`. No changes to `pipeline.py`, `parser.py`, or any other file are needed.


### Extraction Cascade

```
Step 1: First inference (max_new_tokens = 512)
        └── extract_answer(fallback=False) — JSON/Pydantic only
            ├── succeeded → done
            └── failed
                  │
Step 2: Retry inference (max_new_tokens = 1024)
                  └── extract_answer(fallback=False) — JSON/Pydantic only
                      ├── succeeded → done (with retry_used=true)
                      └── failed
                            │
Step 3: Regex fallback on the richer response
                            └── extract_answer(fallback=True)
                                ├── succeeded → done (with used_fallback_extraction=true)
                                └── failed → extraction_succeeded=false
```

### Output Files

| File | Content |
|------|---------|
| `trajectories.jsonl` | Per-sample records with prompt, raw response, extracted answer, success flags, timing, fallback indicators |
| `summary.json` | Overall accuracy, per-subject accuracy, parse failure rate, error count, runtime |
| `accuracy_chart.png` | Bar chart: overall + per-subject accuracy (%) |
| `runtime_metrics.png` | 2×3 dashboard: parse failure rate, inference errors, completion, avg inference time, total runtime, overall accuracy |
| `fallback_analysis.png` | Two-panel: extraction cascade stacked bar + per-subject retry & regex fallback rates |

### Trajectory Record Schema

```json
{
  "sample_id": "validation_Accounting_1",
  "subject": "Accounting",
  "question": "...",
  "choices": ["A: ...", "B: ...", "C: ...", "D: ..."],
  "correct_answer": "B",
  "prompt_sent": "...",
  "raw_model_response": "{\"reasoning\": \"...\", \"answer\": \"B\"}",
  "extracted_answer": "B",
  "extraction_succeeded": true,
  "is_correct": true,
  "inference_time_seconds": 10.15,
  "error": null,
  "used_fallback_extraction": false,
  "retry_used": false,
  "retry_inference_time_seconds": null
}
```

## Structured Generation

The pipeline uses **Outlines** to enforce a Pydantic schema at every decoding step:

1. **Schema definition** (`src/schemas.py`): A `ModelResponse` model with free-form `reasoning` and an enumerated `answer` field (A–J).
2. **Constrained generation** (`src/adapters/qwen.py`): The Outlines `Generator` wraps Qwen3-VL with logits filtering that only allows tokens conforming to the schema.
3. **Deterministic parsing** (`src/parser.py`): Responses are parsed with `json.loads()` + `ModelResponse.model_validate()`. The regex fallback only activates when both JSON attempts fail.
4. **Prompt design** (`src/pipeline.py`): The prompt instructs the model to think step by step and emit ONLY valid JSON.

## Project Considerations

### Results

- Fetches 100 samples from the HF MMMU dataset using stratified fetching across subjects, meaning all subjects have the same number of samples evaluated.
- Resumability logic implemented by tracking the `sample_id`, ensuring that the pipeline can continue exactly from where it stopped if anything interrupts it in the middle of an evaluation.
- Structured generation via Outlines ensures every model output is valid, parseable JSON — eliminating brittle regex-only extraction from the primary path.

### Main Issues Encountered

- **5% parse failure rate with 512 max_new_tokens**, mainly due to truncated model outputs. Switching to 1024 max_new_tokens via the retry fallback achieved a 0% parse failure rate (every sample was answered by the model). Dealing with the truncation of the model's reasoning was a significant challenge.

  **Approach**: Implemented a three-stage extraction cascade. If the first inference attempt fails to produce parseable JSON, the pipeline retries the same sample with 2× the token budget. If that also fails, a regex-based fallback attempts recovery as a last resort.

- **Memory explosion (OOM)**: The system was not releasing consumed RAM during new sample evaluations, resulting in multiple crashes.

  **Approach**: 1) Explicitly closing PIL images after each sample for immediate garbage collection; 2) Forcing PyTorch to release cached GPU memory via `torch.cuda.empty_cache()` after every generation.

## Verification Scripts

```bash
# Check parser follows structured output mandate
bash scripts/constitution-check.sh

# Verify deterministic sample selection
python scripts/verify_determinism.py

# Test stratified sampling distribution
python scripts/test_stratified.py
```

## License

MIT
