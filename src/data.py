from datasets import load_dataset, concatenate_datasets

from src import config


def select_stratified_samples(subset_datasets: list, max_samples: int):
    """Return a deterministically stratified subset.

    Each dataset in ``subset_datasets`` corresponds to one subject.
    Quotas are assigned proportionally (equal split by default) in the
    order the datasets are given.  If a dataset is smaller than its quota
    the shortfall is *not* backfilled from other subjects so the total
    may be slightly below ``max_samples``.
    """
    num_subjects = len(subset_datasets)
    quotas = config.get_subject_quotas(max_samples, num_subjects)

    selected = []
    for ds, quota in zip(subset_datasets, quotas):
        n = min(len(ds), quota)
        if n > 0:
            selected.append(ds.select(range(n)))

    if not selected:
        # create an empty dataset with the same features as the first input
        return subset_datasets[0].select([])

    return concatenate_datasets(selected)


def get_evaluation_dataset(max_samples: int = config.MAX_SAMPLES):
    """Load the MMMU validation subset for selected subjects.

    Subjects are loaded in a fixed alphabetical order, and samples are
    selected **proportionally/stratified** across subjects so that each
    subject contributes roughly ``max_samples / num_subjects`` items.
    """
    subset_datasets = []
    for subject in sorted(config.SUBJECTS):
        ds = load_dataset(
            "MMMU/MMMU",
            name=subject,
            split="validation",
        )
        subset_datasets.append(ds)

    return select_stratified_samples(subset_datasets, max_samples)
