"""
[ToolOrchestra step 4] Download DeepScaleR-Preview (40k math problems) into the raw-data parquet format
used by data/aime24/test.parquet: data_source, prompt (question string), reward_model{style, ground_truth}, extra_info{index}.

Usage: python scripts/prepare_deepscaler.py --out data/deepscaler/train.parquet
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--out", default="data/deepscaler/train.parquet")
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split="train")
    rows = [
        {
            "data_source": "deepscaler",
            "prompt": ex["problem"],
            "reward_model": {"style": "rule", "ground_truth": str(ex["answer"])},
            "extra_info": {"index": i},
        }
        for i, ex in enumerate(ds)
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"Wrote {len(rows)} problems to {args.out}")


if __name__ == "__main__":
    main()
