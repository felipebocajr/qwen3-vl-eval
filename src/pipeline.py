import ast
import json
import os
import re
import time
from typing import List

from PIL import Image

from src import config
from src.model import run_inference
from src.parser import extract_answer


def extract_images(sample: dict) -> List[Image.Image]:
    """Extract non-None PIL images from an MMMU sample."""
    images = []
    for i in range(1, 8):
        img = sample.get(f"image_{i}")
        if img is not None:
            original_format = getattr(img, "format", None)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # convert() strips the .format attribute which Outlines' inputs.Image requires
            if getattr(img, "format", None) is None and original_format:
                img.format = original_format
            images.append(img)
    return images


def parse_options(options_str: str) -> List[str]:
    """Parse MMMU options string into a Python list."""
    try:
        return ast.literal_eval(options_str)
    except Exception:
        return [opt.strip() for opt in options_str.strip("[]").split(",")]


def strip_image_tags(question: str) -> str:
    """Remove <image N> placeholders from question text."""
    return re.sub(r"<image\s+\d+>", "", question).strip()


def format_choices(options: List[str]) -> List[str]:
    """Format options with letter prefixes."""
    letters = [chr(ord("A") + i) for i in range(len(options))]
    return [f"{letter}: {opt}" for letter, opt in zip(letters, options)]


def build_prompt(question: str, options: List[str]) -> str:
    """Build the text prompt for a multiple-choice question."""
    lines = [f"Question: {question}"]
    if options:
        lines.append("Options:")
        for choice in format_choices(options):
            lines.append(choice)
    # Build a human-readable schema hint that matches the actual option count
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
    """Infer subject from sample id like 'validation_Accounting_1'."""
    parts = sample_id.split("_")
    if len(parts) > 2 and parts[0] == "validation":
        return "_".join(parts[1:-1])
    return "unknown"


def evaluate_sample(model, processor, sample) -> dict:
    """
    Run the full evaluation pipeline on a single dataset sample.

    Returns a structured record dict ready for persistence.
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
        "raw_model_response": None,
        "extracted_answer": None,
        "extraction_succeeded": False,
        "is_correct": False,
        "inference_time_seconds": None,
        "error": None,
    }

    try:
        start = time.perf_counter()
        raw_answer = run_inference(
            model,
            processor,
            images,
            prompt,
            num_options=len(options),
            max_new_tokens=config.MAX_NEW_TOKENS,
        )
        elapsed = time.perf_counter() - start
        record["inference_time_seconds"] = round(elapsed, 2)
        record["raw_model_response"] = raw_answer

        extracted, succeeded = extract_answer(raw_answer, num_choices=len(options))
        record["extracted_answer"] = extracted
        record["extraction_succeeded"] = succeeded
        record["is_correct"] = succeeded and (extracted == correct_answer)
    except Exception as exc:
        record["error"] = str(exc)

    return record


def load_completed_sample_ids(jsonl_path: str) -> set[str]:
    """Read trajectories.jsonl and return a set of already-completed sample IDs."""
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed

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

    return completed


def save_trajectory(record: dict, out_dir: str = "results"):
    """Save trajectory record as individual JSON and append to JSONL."""
    os.makedirs(out_dir, exist_ok=True)

    sample_id = record["sample_id"]
    json_path = os.path.join(out_dir, f"{sample_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"Saved record to {json_path}")

    jsonl_path = os.path.join(out_dir, "trajectories.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Appended record to {jsonl_path}")
