# MAS-Orchestra: AIME24 / AIME25 Reproduction

Reproduction of the MAS-Orchestra math results (low DoM setting) reported in the paper / README, using the released 7B orchestrator checkpoint.

**Outcome:** the paper's numbers reproduce. AIME25 matches exactly; AIME24 is 0.42 points lower, which is one sample out of 240 and within sampling noise.

---

## 1. Headline results (Avg@8, %)

| Benchmark | Paper (MAS-Orchestra) | This run | Δ | This run Pass@8 |
|---|---|---|---|---|
| AIME24 | 66.25 | **65.83** | −0.42 | 83.33 |
| AIME25 | 61.25 | **61.25** | 0.00 | 80.00 |
| Mean of both | 63.75 | 63.54 | −0.21 | 81.67 |

- **Avg@8**: for each problem, the fraction of its 8 samples that are correct, averaged over the 30 problems. This is the metric reported in the paper.
- **Pass@8**: fraction of problems where at least one of the 8 samples is correct. Not reported in the paper.
- Code execution success: **100%** (480/480 generated MAS programs ran without error), so no score was lost to invalid MAS code.

### Comparison with the paper's baselines (from the README table)

| Method | AIME24 | AIME25 |
|---|---|---|
| CoTAgent | 50.00 | 45.00 |
| SCAgent | 57.50 | 51.67 |
| DebateAgent | 62.08 | 57.50 |
| ReflexionAgent | 60.83 | 50.42 |
| AFlow | 62.50 | 53.33 |
| MaAS | 32.50 | 40.83 |
| MAS-GPT | 58.75 | 43.33 |
| ToolOrchestra | 23.33 | 11.25 |
| GPT-5 (as orchestrator) | 55.00 | 47.72 |
| Claude-Sonnet-4.5 (as orchestrator) | 45.56 | 35.00 |
| MAS-Orchestra (paper) | 66.25 | 61.25 |
| **MAS-Orchestra (this run)** | **65.83** | **61.25** |

---

## 2. Experimental setup

### Model and task

| Item | Value |
|---|---|
| Orchestrator checkpoint | `harmony-grpo-7b-global-step-180` ([Hugging Face](https://huggingface.co/ZixuanKe/harmony-grpo-7b-global-step-180)), FSDP shards merged to HF format in `checkpoints/harmony-grpo-7b-global-step-180/hf_model` |
| Orchestrator size | 7B |
| Problem type | `harmony_minimal` (low degree of multi-agent-ness, "Low DoM") |
| Building blocks available to orchestrator | `COT`, `COT_SC`, `Reflexion`, `LLM_debate` |
| Sub-agent model | `gpt-oss-120b` (`openai/gpt-oss-120b` on Together AI) |
| Sub-agent reasoning effort | `low` |
| Sub-agent temperature | 0.5 |
| Sub-agent max tokens | not set (provider default) |
| Debate rounds / self-correction attempts | `max_round=5`, `max_sc=5` (config defaults) |

### Data

| Benchmark | File | Problems | Samples per problem | Total runs |
|---|---|---|---|---|
| AIME24 | `data/aime24/test.parquet` | 30 | 8 | 240 |
| AIME25 | `data/aime25/test.parquet` | 30 | 8 | 240 |

### Orchestrator decoding (vLLM, validation sampling)

| Parameter | Value |
|---|---|
| Samples per prompt (`val_kwargs.n`) | 8 |
| Temperature | 0.5 |
| top_p | 1.0 |
| top_k | −1 (disabled) |
| Max prompt length | 15,000 tokens |
| Max response length | 8,192 tokens |
| Tensor parallel size | 1 |
| vLLM GPU memory utilization | 0.7 |
| Eager mode | `enforce_eager=True` |

### Scoring

| Item | Value |
|---|---|
| Reward | final-answer correctness only (`final_answer_weight=1.0`, `execution_success_weight=0.0`) |
| LLM judge | disabled (`use_llm_judge=False`) |
| Answer check | exact match of the MAS final answer against the AIME ground truth |
| Mode | validation only (`val_before_train=True`, `total_training_steps=0`), no training updates |

### Execution

| Item | Value |
|---|---|
| Executor | async agent system (`multiply_processes=0`), 1 Ray agent worker with up to 128 concurrent tasks |
| Script | `eval_math.sh` |
| Slurm job | 1070006 |
| Node / hardware | `dgx040`, 1× NVIDIA H200, 16 CPU cores, 192 GB RAM |
| Start → end | 2026-11-17 07:21:33 → 07:37:24 (15 min 51 s wall clock) |
| MAS execution phase | 480 tasks in 12 min 33 s |

### Software

| Package | Version |
|---|---|
| Python | 3.10.22 |
| torch | 2.6.0 |
| vllm | 0.8.5.post1 |
| transformers | 4.51.1 |
| flash_attn | 2.7.4.post1 |
| ray | 2.49.2 |
| verl | 0.3.1.dev0 (bundled in `verl/`) |
| together | 3.1.1 |
| math-verify | 0.9.0 |
| opentelemetry-sdk | 1.26.0 |
| opentelemetry-exporter-prometheus | 0.47b0 |
| CUDA module | `CUDA/12.9.1` |
| Repo commit | `0527c70` |

### Differences from the original setup

- **1 GPU instead of 8.** The GPU only hosts the 7B orchestrator; sub-agents run through the API. Tensor parallelism was reduced from 2 to 1 and `n_gpus_per_node` follows the Slurm allocation. Model, sampling, sub-agents and scoring are unchanged.
- **Environment fix.** `opentelemetry-exporter-prometheus` was downgraded from 0.66b2 to 0.47b0 to match `opentelemetry-sdk` 1.26.0 (required by vLLM). The mismatch crashed Ray at startup in an earlier attempt (job 1065270).

---

## 3. Per-problem results

"Correct" is the number of the 8 samples that matched the ground truth. "Most common wrong answer" is the most frequent incorrect final answer among the failed samples.

### Distribution of correct samples per problem

| Correct out of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| AIME24 problems | 5 | 2 | 2 | 3 | 0 | 0 | 1 | 17 |
| AIME25 problems | 6 | 3 | 1 | 0 | 2 | 3 | 1 | 14 |

### AIME24

| # | Ground truth | Correct | Most common wrong answer |
|---|---|---|---|
| 1 | 204 | 8/8 | — |
| 2 | 113 | 1/8 | 59 |
| 3 | 371 | 1/8 | 139 |
| 4 | 385 | 0/8 | 9 |
| 5 | 110 | 3/8 | 134 |
| 6 | 104 | 0/8 | 115 |
| 7 | 721 | 8/8 | — |
| 8 | 025 | 8/8 | — |
| 9 | 809 | 8/8 | — |
| 10 | 116 | 8/8 | — |
| 11 | 104 | 7/8 | 110 |
| 12 | 294 | 8/8 | — |
| 13 | 540 | 8/8 | — |
| 14 | 197 | 2/8 | 389 |
| 15 | 480 | 8/8 | — |
| 16 | 073 | 8/8 | — |
| 17 | 468 | 3/8 | 156 |
| 18 | 601 | 3/8 | 7 |
| 19 | 023 | 8/8 | — |
| 20 | 321 | 8/8 | — |
| 21 | 211 | 8/8 | — |
| 22 | 315 | 0/8 | 15 |
| 23 | 236 | 8/8 | — |
| 24 | 045 | 8/8 | — |
| 25 | 033 | 8/8 | — |
| 26 | 080 | 0/8 | 100 |
| 27 | 055 | 8/8 | — |
| 28 | 699 | 8/8 | — |
| 29 | 127 | 2/8 | 17 |
| 30 | 902 | 0/8 | 1024 |

### AIME25

| # | Ground truth | Correct | Most common wrong answer |
|---|---|---|---|
| 1 | 70 | 8/8 | — |
| 2 | 588 | 7/8 | 661 |
| 3 | 16 | 8/8 | — |
| 4 | 117 | 8/8 | — |
| 5 | 279 | 8/8 | — |
| 6 | 504 | 8/8 | — |
| 7 | 821 | 1/8 | 271 |
| 8 | 77 | 8/8 | — |
| 9 | 62 | 1/8 | 79 |
| 10 | 81 | 0/8 | 62 |
| 11 | 259 | 8/8 | — |
| 12 | 510 | 2/8 | 508 |
| 13 | 204 | 0/8 | 487/3 |
| 14 | 60 | 0/8 | 87 |
| 15 | 735 | 0/8 | 147 |
| 16 | 468 | 8/8 | — |
| 17 | 49 | 8/8 | — |
| 18 | 82 | 4/8 | 90 |
| 19 | 106 | 8/8 | — |
| 20 | 336 | 0/8 | 516 |
| 21 | 293 | 8/8 | — |
| 22 | 237 | 4/8 | 60671 |
| 23 | 610 | 5/8 | 600 |
| 24 | 149 | 5/8 | 150 |
| 25 | 907 | 8/8 | — |
| 26 | 113 | 5/8 | 4464 |
| 27 | 19 | 8/8 | — |
| 28 | 248 | 0/8 | 296 |
| 29 | 104 | 8/8 | — |
| 30 | 240 | 1/8 | 188 |

---

## 4. Reproducing this run

```bash
# one-time: store the Together API key (file readable only by you)
read -rsp "Paste your Together API key: " KEY && echo && printf '%s\n' "$KEY" > ~/.together_api_key && chmod 600 ~/.together_api_key && unset KEY

cd /scratch/user/debajoym98_tamu.edu/MAS-Orchestra
sbatch eval_math.sh            # defaults to harmony-grpo-7b-global-step-180
```

Results are written to `eval_results/eval_math_<checkpoint>_<timestamp>/`, and the per-benchmark Avg@8 / Pass@8 summary is printed at the end of `logs/slurm_eval_math_<jobid>.log`.

## 5. Artifacts from this run

| File | Contents |
|---|---|
| `eval_results/eval_math_harmony-grpo-7b-global-step-180_20261117_072138/summary.json` | Per-benchmark Avg@8 / Pass@8 |
| `eval_results/eval_math_harmony-grpo-7b-global-step-180_20261117_072138/responses_and_ground_truth/step_0_responses.json` | All 480 samples: question, MAS final answer, ground truth, execution status, correctness |
| `eval_results/eval_math_harmony-grpo-7b-global-step-180_20261117_072138/run.log` | Full trainer / MAS execution log |
| `logs/slurm_eval_math_1070006.log` | Slurm job log, including the final summary |

## 6. Limitations

- **Single run.** Avg@8 at temperature 0.5 has run-to-run variance; differences under about one or two points between runs should not be read as meaningful.
- **Building-block choice not recorded.** The saved responses contain each run's final answer but not the MAS code the orchestrator generated, so this run cannot confirm the paper's observation that the orchestrator delegates to Reflexion or Debate on AIME.
- **API-served sub-agent.** `gpt-oss-120b` is served by Together; provider-side model or serving changes could shift results over time.
