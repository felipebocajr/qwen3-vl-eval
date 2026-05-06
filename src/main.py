import json
import sys

from src import config
from src.data import get_evaluation_dataset
from src.metrics import compute_summary, save_summary
from src.model import load_model_and_processor
from src.pipeline import evaluate_sample, load_completed_sample_ids, save_trajectory


def main():
    # 1. Resume state
    completed_ids = load_completed_sample_ids(config.TRAJECTORIES_PATH)
    n_completed = len(completed_ids)
    print(f"Found {n_completed} completed sample(s) in {config.TRAJECTORIES_PATH}.")

    # 2. Over-cap guard
    if n_completed >= config.MAX_SAMPLES:
        print(
            f"Cap already reached ({n_completed} samples >= {config.MAX_SAMPLES}). "
            "Exiting without loading model or dataset."
        )
        sys.exit(0)

    # 3. Load model & processor once
    model, processor = load_model_and_processor()

    # 4. Load evaluation dataset (stratified + deterministically capped)
    dataset = get_evaluation_dataset(max_samples=config.MAX_SAMPLES)

    # Subject distribution logging
    from collections import Counter
    from src.pipeline import infer_subject
    subject_counts = Counter(infer_subject(s["id"]) for s in dataset)
    print(f"Subject distribution: {dict(subject_counts)}")

    n_to_evaluate = min(len(dataset), config.MAX_SAMPLES - n_completed)
    print(
        f"Loaded {len(dataset)} evaluation sample(s). "
        f"Will evaluate up to {n_to_evaluate} new sample(s) "
        f"(cap={config.MAX_SAMPLES}, already_done={n_completed}).\n"
    )

    if n_to_evaluate <= 0:
        print("Nothing new to evaluate. Exiting.")
        return

    # 5. Iterate and evaluate
    n_new = 0
    for idx, sample in enumerate(dataset):
        sample_id = sample["id"]

        # Skip already completed
        if sample_id in completed_ids:
            continue

        # Stop once we've reached the cap
        if n_new >= n_to_evaluate:
            break

        print(f"[{n_new + 1}/{n_to_evaluate}] Processing {sample_id} ...")

        record = evaluate_sample(model, processor, sample)

        status = "✅" if record["is_correct"] else ("⚠️" if record["extraction_succeeded"] else "❌")
        print(
            f"    {status} extracted={record['extracted_answer']}  "
            f"correct={record['correct_answer']}  "
            f"time={record['inference_time_seconds']}s  error={record['error']}\n"
        )

        save_trajectory(record)
        completed_ids.add(sample_id)
        n_new += 1

    # 6. Summary
    total_unique = len(completed_ids)
    print(f"\nRun complete. Evaluated {n_new} new sample(s). Total unique: {total_unique}.")

    summary = compute_summary(config.TRAJECTORIES_PATH, max_samples=config.MAX_SAMPLES)
    save_summary(summary, config.SUMMARY_PATH)
    print(f"Summary saved to {config.SUMMARY_PATH}")


if __name__ == "__main__":
    main()
