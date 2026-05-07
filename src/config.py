"""Central configuration constants for the Qwen3-VL MMMU evaluation pipeline.

Defines generation parameters, output paths, subject selection, and
sample-quota allocation logic.
"""

MAX_SAMPLES = 100

# Canonical model checkpoint ID. Changing this value propagates across
# the entire pipeline (adapter, processor, and related components).
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Output directory and file paths.
RESULTS_DIR = "results"
TRAJECTORIES_PATH = f"{RESULTS_DIR}/trajectories.jsonl"
SUMMARY_PATH = f"{RESULTS_DIR}/summary.json"

# Token budget per generation pass.
MAX_NEW_TOKENS = 1024

# MMMU subjects to load, ordered alphabetically for deterministic runs.
SUBJECTS = [
    "Accounting",
    "Architecture_and_Engineering",
    "Art",
    "Biology",
]


def get_subject_quotas(max_samples: int, num_subjects: int) -> list[int]:
    """Return per-subject sample quotas that sum to at most *max_samples*.

    Distributes samples evenly across subjects, with any remainder
    assigned to the first subjects in order.
    """
    base = max_samples // num_subjects
    remainder = max_samples % num_subjects
    return [base + (1 if i < remainder else 0) for i in range(num_subjects)]
