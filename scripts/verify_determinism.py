"""Verify that get_evaluation_dataset returns identical sample IDs AND
per-subject distribution across two runs."""

from collections import Counter

from src.data import get_evaluation_dataset
from src.pipeline import infer_subject


def main():
    ds1 = get_evaluation_dataset(max_samples=100)
    ids1 = [s["id"] for s in ds1]
    counts1 = Counter(infer_subject(s["id"]) for s in ds1)

    ds2 = get_evaluation_dataset(max_samples=100)
    ids2 = [s["id"] for s in ds2]
    counts2 = Counter(infer_subject(s["id"]) for s in ds2)

    assert ids1 == ids2, f"Sample IDs differ!\nFirst:  {ids1}\nSecond: {ids2}"
    assert counts1 == counts2, f"Subject counts differ!\nFirst:  {counts1}\nSecond: {counts2}"

    print(f"✅ Determinism verified: {len(ids1)} identical sample IDs.")
    print(f"   Subject distribution: {dict(counts1)}")


if __name__ == "__main__":
    main()
