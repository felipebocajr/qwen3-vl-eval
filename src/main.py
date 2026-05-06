from src.data import get_evaluation_dataset
from src.model import load_model_and_processor
from src.pipeline import evaluate_sample, save_trajectory


def main():
    # 1. Load model & processor once
    model, processor = load_model_and_processor()

    # 2. Load evaluation dataset
    dataset = get_evaluation_dataset()
    print(f"Loaded {len(dataset)} evaluation samples.\n")

    # 3. Iterate and evaluate
    for idx, sample in enumerate(dataset):
        sample_id = sample["id"]
        print(f"[{idx + 1}/{len(dataset)}] Processing {sample_id} ...")

        record = evaluate_sample(model, processor, sample)

        status = "✅" if record["is_correct"] else ("⚠️" if record["extraction_succeeded"] else "❌")
        print(f"    {status} extracted={record['extracted_answer']}  correct={record['correct_answer']}  "
              f"time={record['inference_time_seconds']}s  error={record['error']}\n")

        save_trajectory(record)


if __name__ == "__main__":
    main()
