from datasets import load_dataset, concatenate_datasets


def get_evaluation_dataset():
    """Load the MMMU validation subset for selected subjects."""
    subjects = [
        "Accounting",
        "Architecture_and_Engineering",
        "Art",
        "Biology",
    ]

    subset_datasets = []
    for subject in subjects:
        ds = load_dataset(
            "MMMU/MMMU",
            name=subject,
            split="validation[:25]",
        )
        subset_datasets.append(ds)

    return concatenate_datasets(subset_datasets)
