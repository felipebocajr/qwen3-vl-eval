"""Quick smoke test that dataset loading returns ~25 samples per subject."""

from collections import Counter

from src.data import get_evaluation_dataset
from src.pipeline import infer_subject


def main():
    ds = get_evaluation_dataset(max_samples=100)
    counts = Counter(infer_subject(s["id"]) for s in ds)

    print(f"Total samples: {len(ds)}")
    print(f"Subject distribution: {dict(counts)}")

    # Assert roughly balanced (allow ±3 for edge cases where dataset is small)
    for subj, cnt in counts.items():
        assert 20 <= cnt <= 30, f"Subject {subj} has {cnt} samples, expected 20–30"

    print("✅ Stratified distribution looks balanced.")


if __name__ == "__main__":
    main()
