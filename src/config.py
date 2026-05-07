"""Central configuration constants for the Qwen3-VL MMMU evaluation pipeline.

Defines generation parameters, output paths, subject selection, and
sample-quota allocation logic.
"""

MAX_SAMPLES = 100

# Canonical model checkpoint ID.  Changing this value updates model loading
# across the entire pipeline (adapter, processor, and related components)
# without requiring edits to any other file.
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

RESULTS_DIR = "results"
TRAJECTORIES_PATH = f"{RESULTS_DIR}/trajectories.jsonl"
SUMMARY_PATH = f"{RESULTS_DIR}/summary.json"

# Generation parameters
# Reduced from 4096: the structured JSON output (reasoning + single letter)
# never needs more than ~500 tokens.  A smaller cap also shrinks the KV cache
# footprint during generation, reducing memory pressure.
MAX_NEW_TOKENS = 512

# Subjects to load from MMMU, listed alphabetically for deterministic ordering
SUBJECTS = [
    "Accounting",
    "Architecture_and_Engineering",
    "Art",
    "Biology",
]


def get_subject_quotas(max_samples: int, num_subjects: int) -> list[int]:
    """Return per-subject sample quotas that sum to at most *max_samples*.

    Distributes samples evenly across subjects, with any remainder assigned
to the first subjects in order.
    """
    base = max_samples // num_subjects
    remainder = max_samples % num_subjects
    return [base + (1 if i < remainder else 0) for i in range(num_subjects)]
