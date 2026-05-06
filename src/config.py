"""Pipeline configuration constants."""

MAX_SAMPLES = 100
RESULTS_DIR = "results"
TRAJECTORIES_PATH = f"{RESULTS_DIR}/trajectories.jsonl"
SUMMARY_PATH = f"{RESULTS_DIR}/summary.json"

# Subjects to load from MMMU, listed alphabetically for deterministic ordering
SUBJECTS = [
    "Accounting",
    "Architecture_and_Engineering",
    "Art",
    "Biology",
]


def get_subject_quotas(max_samples: int, num_subjects: int) -> list[int]:
    """Return a list of per-subject sample quotas that sum to at most ``max_samples``."""
    base = max_samples // num_subjects
    remainder = max_samples % num_subjects
    return [base + (1 if i < remainder else 0) for i in range(num_subjects)]

