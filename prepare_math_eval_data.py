"""Build AIME24 / AIME25 eval parquets in the raw-data format read by
mas_r1_reasoner.trainer.utils.helper.create_raw_data_datasets
(columns: prompt, reward_model.ground_truth, data_source)."""
import os

import pandas as pd
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def to_rows(problems, answers, data_source):
    return [
        {
            "data_source": data_source,
            "prompt": str(problem).strip(),
            "reward_model": {"style": "rule", "ground_truth": str(answer).strip()},
            "extra_info": {"index": i},
        }
        for i, (problem, answer) in enumerate(zip(problems, answers))
    ]


def save(rows, name):
    path = os.path.join(OUT_DIR, name, "test.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)
    print(f"{name}: {len(rows)} problems -> {path}")


def main():
    aime24 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    save(to_rows(aime24["problem"], aime24["answer"], "aime24"), "aime24")

    aime25 = load_dataset("math-ai/aime25", split="test")
    save(to_rows(aime25["problem"], aime25["answer"], "aime25"), "aime25")


if __name__ == "__main__":
    main()
