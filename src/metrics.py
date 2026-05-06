import json
import os
from collections import defaultdict


def compute_summary(jsonl_path: str, max_samples: int = 100) -> dict:
    """Compute aggregate metrics from trajectories.jsonl.

    Only the first ``max_samples`` unique records (in file order) are
    considered, matching the intended evaluation cap.
    """
    if not os.path.exists(jsonl_path):
        return {
            "overall_accuracy": 0.0,
            "per_subject_accuracy": {},
            "parse_failure_rate": 0.0,
            "inference_error_count": 0,
            "average_inference_time_sec": 0.0,
            "total_runtime_sec": 0.0,
            "total_samples_evaluated": 0,
            "max_samples": max_samples,
        }

    records = []
    seen_ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sid = record.get("sample_id")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            records.append(record)
            if len(records) >= max_samples:
                break

    total = len(records)
    if total == 0:
        return {
            "overall_accuracy": 0.0,
            "per_subject_accuracy": {},
            "parse_failure_rate": 0.0,
            "inference_error_count": 0,
            "average_inference_time_sec": 0.0,
            "total_runtime_sec": 0.0,
            "total_samples_evaluated": 0,
            "max_samples": max_samples,
        }

    correct = 0
    parse_failures = 0
    inference_errors = 0
    total_inference_time = 0.0
    subject_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in records:
        succeeded = rec.get("extraction_succeeded", False)
        is_correct = rec.get("is_correct", False)
        error = rec.get("error")
        time_sec = rec.get("inference_time_seconds")
        subject = rec.get("subject", "unknown")

        if error:
            inference_errors += 1

        if not succeeded:
            parse_failures += 1
        elif is_correct:
            correct += 1
            subject_counts[subject]["correct"] += 1

        subject_counts[subject]["total"] += 1

        if time_sec is not None:
            total_inference_time += time_sec

    overall_accuracy = correct / total if total > 0 else 0.0
    parse_failure_rate = parse_failures / total if total > 0 else 0.0
    avg_inference_time = total_inference_time / total if total > 0 else 0.0

    per_subject_accuracy = {
        subj: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for subj, stats in subject_counts.items()
    }

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "per_subject_accuracy": per_subject_accuracy,
        "parse_failure_rate": round(parse_failure_rate, 4),
        "inference_error_count": inference_errors,
        "average_inference_time_sec": round(avg_inference_time, 2),
        "total_runtime_sec": round(total_inference_time, 2),
        "total_samples_evaluated": total,
        "max_samples": max_samples,
    }


def save_summary(summary: dict, path: str):
    """Write summary dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
