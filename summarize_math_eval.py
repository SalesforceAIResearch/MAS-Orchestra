"""Per-benchmark Avg@k / Pass@k from the responses saved by an eval_math.sh run."""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

DATA_FILES = {
    "aime24": "data/aime24/test.parquet",
    "aime25": "data/aime25/test.parquet",
}


def main(save_path):
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    question_to_source = {}
    for name, rel_path in DATA_FILES.items():
        df = pd.read_parquet(os.path.join(repo_dir, rel_path))
        for prompt in df["prompt"]:
            question_to_source[prompt.strip()] = name

    resp_dir = os.path.join(save_path, "responses_and_ground_truth")
    step_files = sorted(f for f in os.listdir(resp_dir) if f.startswith("step_") and f.endswith("_responses.json"))
    with open(os.path.join(resp_dir, step_files[-1])) as f:
        samples = json.load(f)["responses_and_ground_truth"]

    per_question = defaultdict(list)
    for s in samples:
        source = question_to_source.get(s["question"].strip(), "unknown")
        per_question[(source, s["question"])].append(float(s["answer_correct"]))

    per_source = defaultdict(lambda: {"avg": [], "pass": []})
    for (source, _), scores in per_question.items():
        per_source[source]["avg"].append(sum(scores) / len(scores))
        per_source[source]["pass"].append(float(max(scores) > 0))

    summary = {}
    for source, vals in sorted(per_source.items()):
        n_q = len(vals["avg"])
        summary[source] = {
            "num_questions": n_q,
            "samples_per_question": len(samples) // max(len(per_question), 1),
            "avg@k": 100 * sum(vals["avg"]) / n_q,
            "pass@k": 100 * sum(vals["pass"]) / n_q,
        }
        print(f"{source}: Avg@k = {summary[source]['avg@k']:.2f}  Pass@k = {summary[source]['pass@k']:.2f}  ({n_q} questions)")

    with open(os.path.join(save_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
