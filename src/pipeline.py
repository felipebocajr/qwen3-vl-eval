"""Evaluation pipeline loop with resume support and trajectory persistence.

Orchestrates per-sample evaluation: image extraction, prompt construction,
model inference, answer parsing, and result recording. Supports resuming from
a partially complete ``trajectories.jsonl`` file.
"""

import ast
import json
import os
import re
import time
from typing import List

from PIL import Image

from src import config
from src.parser import extract_answer


def extract_images(sample: dict) -> List[Image.Image]:
    """Extract up to 7 images from an MMMU sample dict.

    Converts non-RGB images to RGB and preserves the original image format
    metadata when available.
    """
    images = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            # Preserve format metadata across mode conversion.
            original_format = getattr(img, "format", None)
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.format is None and original_format:
                img.format = original_format
            images.append(img)
    return images

def parse_options(options_str: str) -> List[str]:
    """Parse the MMMU options string into a list of individual choices.

    Tries ``ast.literal_eval`` for safe structured parsing first. Falls back
    to comma-splitting on malformed strings.
    """
    try:
        return ast.literal_eval(options_str)
    except Exception:
        return [opt.strip() for opt in options_str.strip("[]").split(",")]


def strip_image_tags(question: str) -> str:
    """Remove <image N> placeholders from question text."""
    return re.sub(r"<image\s+\d+>", "", question).strip()


def format_choices(options: List[str]) -> List[str]:
    """Prefix each option string with its letter (A, B, C, …)."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    return [f"{letter}: {opt}" for letter, opt in zip(letters, options)]


def build_prompt(question: str, options: List[str]) -> str:
    """Build the text prompt for a multiple-choice question.

    Formats the question and options, then appends an instruction to respond
    with valid JSON (including a schema hint showing the allowed answer
    letters).
    """
    lines = [f"Question: {question}"]
    if options:
        lines.append("Options:")
        for choice in format_choices(options):
            lines.append(choice)
    # Build a human-readable schema hint matching the actual option count.
    if options:
        letters = "|".join(chr(ord("A") + i) for i in range(len(options)))
        schema_hint = f'{{"reasoning": "<your reasoning>", "answer": "<{letters}>"}}'
    else:
        schema_hint = '{"reasoning": "<your reasoning>", "answer": "<text answer>"}'
    lines.append(
        f"\nThink step by step, then respond with ONLY valid JSON matching this "
        f"exact schema: {schema_hint}."
    )
    return "\n".join(lines)


def infer_subject(sample_id: str) -> str:
    """Infer the MMMU subject name from a sample ID string.

    Example: ``"validation_Accounting_1"`` → ``"Accounting"``.
    """
    parts = sample_id.split("_")
    if len(parts) > 2 and parts[0] == "validation":
        return "_".join(parts[1:-1])
    return "unknown"


def evaluate_sample(adapter, sample) -> dict:
    """Run the full evaluation pipeline on a single dataset sample.

    Performs image extraction, prompt construction, model inference (with
    automatic retry and regex fallback on extraction failure), answer
    parsing, and correctness judging.

    Args:
        adapter: A ``BaseVLMAdapter`` instance that provides
            ``generate_answer``.
        sample: An MMMU dataset row dict with keys ``id``, ``question``,
            ``options``, ``answer``, and ``image_1`` … ``image_7``.

    Returns:
        A dict ready for JSONL persistence containing the sample metadata,
        model response, extraction result, correctness flag, and timing
        information.
    """
    sample_id = sample["id"]
    images = extract_images(sample)
    question = strip_image_tags(sample["question"])
    options = parse_options(sample["options"])
    prompt = build_prompt(question, options)
    choices = format_choices(options)
    correct_answer = sample["answer"]
    subject = infer_subject(sample_id)

    record = {
        "sample_id": sample_id,
        "subject": subject,
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "prompt_sent": prompt,
        "model_used": adapter.model_name,
        "first_try": None,
        "retry_2x": None,
        "regex_fallback": None,
        "raw_model_response": None,
        "extracted_answer": None,
        "extraction_succeeded": False,
        "is_correct": False,
        "inference_time_seconds": None,
        "error": None,
        "used_fallback_extraction": False,
        "retry_used": False,
        "retry_inference_time_seconds": None,
    }

    try:
        # Step 1: first inference (JSON extraction only, no regex yet).
        start = time.perf_counter()
        raw_answer = adapter.generate_answer(
            prompt,
            images,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )
        elapsed = time.perf_counter() - start

        num_opts = len(options)
        extracted, succeeded = extract_answer(raw_answer, num_choices=num_opts, fallback=False)

        record["first_try"] = {
            "raw_response": raw_answer,
            "extracted_answer": extracted,
            "succeeded": succeeded,
            "inference_time_seconds": round(elapsed, 2),
        }

        retry_time = 0.0
        retry_raw: str | None = None

        # Step 2: retry with 2× token budget (JSON only, still no regex).
        if not succeeded:
            print(
                f"    🔄 JSON extraction failed for {sample_id} — "
                f"retrying with {config.MAX_NEW_TOKENS * 2} max_new_tokens"
            )
            retry_start = time.perf_counter()
            retry_raw = adapter.generate_answer(
                prompt,
                images,
                max_new_tokens=config.MAX_NEW_TOKENS * 2,
            )
            retry_time = time.perf_counter() - retry_start
            elapsed += retry_time

            retry_extracted, retry_succeeded = extract_answer(
                retry_raw, num_choices=num_opts, fallback=False
            )
            record["retry_2x"] = {
                "raw_response": retry_raw,
                "extracted_answer": retry_extracted,
                "succeeded": retry_succeeded,
                "inference_time_seconds": round(retry_time, 2),
            }
            record["retry_used"] = True
            record["retry_inference_time_seconds"] = round(retry_time, 2)
            raw_answer = retry_raw  # keep the richer response for regex fallback
            extracted, succeeded = retry_extracted, retry_succeeded

        # Step 3: regex fallback (last resort). Uses whichever response
        # has more context — the retry response (2× tokens) if available.
        if not succeeded:
            best_text = retry_raw if retry_raw is not None else raw_answer
            regex_extracted, regex_succeeded = extract_answer(
                best_text, num_choices=num_opts, fallback=True
            )
            record["regex_fallback"] = {
                "raw_response": best_text,
                "extracted_answer": regex_extracted,
                "succeeded": regex_succeeded,
            }
            if regex_succeeded:
                record["used_fallback_extraction"] = True
                raw_answer = best_text
                extracted, succeeded = regex_extracted, regex_succeeded
                print(f"    ⚠️  Regex fallback recovered answer for {sample_id}")
            else:
                print(f"    ❌ All extraction attempts failed for {sample_id}")

        # Finalise record.
        record["raw_model_response"] = raw_answer
        record["extracted_answer"] = extracted
        record["extraction_succeeded"] = succeeded
        record["inference_time_seconds"] = round(elapsed, 2)
        record["is_correct"] = succeeded and (extracted == correct_answer)

        # Release PIL image references to prevent memory creep across samples.
        for img in images:
            img.close()
        images.clear()
    except Exception as exc:
        record["error"] = str(exc)

    return record


def load_completed_sample_ids(jsonl_path: str) -> set[str]:
    """Return the set of sample IDs already present in the trajectories file.

    Also scans the ``results/samples/`` directory as a backup source of truth,
    so that if ``trajectories.jsonl`` is truncated or lost, the per-sample
    JSON files still count as completed.

    Returns an empty set if neither source has any records.
    """
    completed = set()

    # Primary source: trajectories.jsonl
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed.add(record["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    # Backup source: per-sample JSON files under results/samples/
    samples_dir = os.path.join("results", "samples")
    if os.path.isdir(samples_dir):
        for fname in os.listdir(samples_dir):
            if fname.endswith(".json"):
                sample_id = fname[:-5]  # strip .json suffix
                completed.add(sample_id)

    return completed


def save_trajectory(record: dict, out_dir: str = "results") -> None:
    """Persist a trajectory record to disk.

    Writes both a per-sample JSON file under ``<out_dir>/samples/`` and
    appends one line to the aggregate ``trajectories.jsonl``.
    """
    os.makedirs(out_dir, exist_ok=True)

    sample_id = record["sample_id"]
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    json_path = os.path.join(samples_dir, f"{sample_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Saved record to {json_path}")

    jsonl_path = os.path.join(out_dir, "trajectories.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Appended record to {jsonl_path}")


def rebuild_jsonl_from_samples(out_dir: str = "results") -> int:
    """Rebuild ``trajectories.jsonl`` from per-sample JSON files.

    Reads every ``.json`` file under ``<out_dir>/samples/`` and writes them
    as one JSONL line each into ``<out_dir>/trajectories.jsonl``, sorted by
    filename. This ensures the JSONL stays in sync with the individual sample
    records even if the JSONL was truncated or lost.

    Returns:
        The number of records written.
    """
    samples_dir = os.path.join(out_dir, "samples")
    if not os.path.isdir(samples_dir):
        return 0

    records = []
    for fname in sorted(os.listdir(samples_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(samples_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

    jsonl_path = os.path.join(out_dir, "trajectories.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)
