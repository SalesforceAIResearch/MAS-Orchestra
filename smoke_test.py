"""GPU smoke test: load the merged orchestrator in vLLM, build the real low-DoM (harmony_minimal)
prompt for a few AIME problems, generate, and parse the proposed MAS with the repo's harmony parser.
Sub-agents are not executed, so no API key is needed."""
import os
import sys

import pandas as pd
from omegaconf import OmegaConf

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)
os.environ.setdefault("TOGETHER_API_KEY", "dummy")


def main(model_path, num_questions=3, n=2):
    config = OmegaConf.load(os.path.join(REPO_DIR, "mas_r1_reasoner/configs/grpo_trainer.yaml"))
    config.azr.problem_type = "harmony_minimal"
    config.azr.mas_r1.multiply_processes = 0
    config.azr.mas_r1.max_ray_workers = 1
    config.azr.mas_r1.agent.model_name = "gpt-oss-120b"
    config.azr.mas_r1.agent.reasoning_effort = "low"

    from mas_r1_reasoner.data_precessor.MathDatasetProcessor import MathDatasetProcessor
    from mas_r1_reasoner.agents.common import get_prompt
    from mas_r1_reasoner.agents.code_sanity import validate_python_code
    from mas_r1_reasoner.rewards.utils.harmony_parser import extract_harmony_code_from_response

    MathDatasetProcessor(None).setup_global_variables(config.azr.mas_r1.agent, config.azr.mas_r1, config)
    system_prompt, user_prompt, developer_prompt = get_prompt("PLACEHOLDER_QUESTION")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    questions = pd.read_parquet(os.path.join(REPO_DIR, "data/aime24/test.parquet"))["prompt"].tolist()[:num_questions]
    prompts = []
    for q in questions:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.replace("PLACEHOLDER_QUESTION", q)},
            {"role": "user", "content": developer_prompt},
        ]
        prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    print(f"Prompt length (tokens): {len(tokenizer.encode(prompts[0]))}")

    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=24000, gpu_memory_utilization=0.8, enforce_eager=True)
    outputs = llm.generate(prompts, SamplingParams(n=n, temperature=0.5, top_p=1.0, max_tokens=8192))

    parsed_ok = 0
    for q, out in zip(questions, outputs):
        for i, completion in enumerate(out.outputs):
            text = completion.text
            code, name, _ = extract_harmony_code_from_response(text, validate_python_code, None)
            ok = code is not None and validate_python_code(code, None)
            parsed_ok += int(ok)
            print("=" * 100)
            print(f"Q: {q[:120]}...  | sample {i} | {len(completion.token_ids)} tokens | parsed MAS: {ok} | name: {name}")
            print(text[:1500])
    total = len(questions) * n
    print("=" * 100)
    print(f"SMOKE TEST: {parsed_ok}/{total} responses parsed into a valid MAS")


if __name__ == "__main__":
    main(sys.argv[1])
