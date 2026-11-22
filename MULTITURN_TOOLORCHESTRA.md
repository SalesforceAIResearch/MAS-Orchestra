# ToolOrchestra branch: multi-turn (dynamic plan-execute) MAS-Orchestra

Branch: `ToolOrchestra`. Every change is tagged in the code with a `[ToolOrchestra step N]` comment
(search the repo for `ToolOrchestra step` to jump to all of them).

**What changed in one paragraph.** Single-shot MAS-Orchestra lets Qwen2.5-7B write a whole MAS plan
(XML `<agent>` blocks) in one response; the plan is executed afterwards inside the reward manager.
On this branch the orchestrator instead works turn by turn: it thinks, calls one or more of the same
four agents (CoTAgent, SCAgent, ReflexionAgent, DebateAgent, run by gpt-oss-120b) as tools, reads their
results, and decides what to do next, up to 6 turns, until it writes a final `<answer>`. The loop runs
inside verl's SGLang multi-turn rollout, agent results are masked out of the policy loss, and the
reward is the correctness of the final `<answer>`. Single-shot training is untouched and still runs
with `--config-name=grpo_trainer`.

```
for turn in 1..max_turns (6):
    orchestrator (Qwen2.5-7B) -> <thinking> + <tool_call>{"name": "<Agent>", "arguments": {...}}</tool_call>   loss_mask = 1
    no tool call?  -> episode ends; this turn holds the final <answer>
    each tool call -> MAS-Orchestra agent on gpt-oss-120b (parallel within a turn)
    agent results  -> appended as <tool_response> messages                                                 loss_mask = 0
reward = final <answer> vs ground truth   (GRPO, n = 8 episodes per question)
```

---

## Step 1: branch

`git checkout -b ToolOrchestra` (created from `main`).

## Step 2: multi-turn logic and rollout loop

| What | Where |
|---|---|
| Rollout dispatch: multi-turn when `multi_turn.enable=True`, otherwise the original single-shot path | [`rollout_generation`](mas_r1_reasoner/trainer/utils/rollout.py#L28) |
| Multi-turn rollout driver (train: `rollout.n` episodes per prompt; validation: prompts already repeated `val_kwargs.n` times) | [`multi_turn_rollout_generation`](mas_r1_reasoner/trainer/utils/rollout_multiturn.py#L66) |
| Builds `raw_prompt` (chat messages) and `tools_kwargs` (original question for each agent tool) | [`attach_multiturn_inputs`](mas_r1_reasoner/trainer/utils/rollout_multiturn.py#L52) |
| Hook that calls it when the generation batch is built | [`_prepare_raw_data_batch_for_generation`](mas_r1_reasoner/trainer/utils/helper.py#L345) |
| Episode metrics: turns, agent calls, fraction of episodes with no agent, per-agent usage | [`summarize_episodes`](mas_r1_reasoner/trainer/utils/rollout_multiturn.py#L101) |
| Mode switch / tool names | [`is_multi_turn_enabled`](mas_r1_reasoner/trainer/utils/rollout_multiturn.py#L30), [`get_tool_names`](mas_r1_reasoner/trainer/utils/rollout_multiturn.py#L35) |
| The per-turn loop itself (verl, unchanged): generate, parse tool calls, run tools in parallel, append results, stop on no tool call / max turns / length | [`AsyncSGLangRollout._async_rollout_a_request`](verl/verl/workers/rollout/sglang_rollout/async_sglang_rollout.py#L460), [`generate_sequences_with_tools`](verl/verl/workers/rollout/sglang_rollout/async_sglang_rollout.py#L611) |

**The environment: agents as tools.** [`mas_r1_reasoner/tools/mas_agent_tools.py`](mas_r1_reasoner/tools/mas_agent_tools.py#L4)

| What | Where |
|---|---|
| Base tool (verl `BaseTool`): per-episode state, concurrency limit, timeout, error-to-text | [`MASAgentTool`](mas_r1_reasoner/tools/mas_agent_tools.py#L128), [`execute`](mas_r1_reasoner/tools/mas_agent_tools.py#L172) |
| Empty `agent_input` = whole question; otherwise the same sub-task template as `harmony_minimal` | [`_build_task_info`](mas_r1_reasoner/tools/mas_agent_tools.py#L147), [`SUB_TASK_TEMPLATE`](mas_r1_reasoner/tools/mas_agent_tools.py#L30) |
| Observation returned to the orchestrator (final answer + truncated reasoning) | [`_format_observation`](mas_r1_reasoner/tools/mas_agent_tools.py#L162) |
| Rebuilds the agent globals (sampler map, FORMAT_INST, node model, debate roles...) inside the rollout worker, where the driver's globals do not exist | [`_ensure_agent_runtime`](mas_r1_reasoner/tools/mas_agent_tools.py#L42) |
| The four agents | [`CoTAgentTool`](mas_r1_reasoner/tools/mas_agent_tools.py#L196), [`SCAgentTool`](mas_r1_reasoner/tools/mas_agent_tools.py#L200), [`ReflexionAgentTool`](mas_r1_reasoner/tools/mas_agent_tools.py#L204), [`DebateAgentTool`](mas_r1_reasoner/tools/mas_agent_tools.py#L208) (validates `debate_roles`, [`_parse_debate_roles`](mas_r1_reasoner/tools/mas_agent_tools.py#L104)) |

**Reward and training changes.**

| What | Where |
|---|---|
| Parser: last `<answer>` written by the orchestrator, ignoring text inside `<tool_response>` / `<tool_call>` | [`multiturn.py`](mas_r1_reasoner/rewards/utils/harmony_parser/multiturn.py#L26) ([`extract_harmony_code_from_response`](mas_r1_reasoner/rewards/utils/harmony_parser/multiturn.py#L33)) |
| Parser router branch for `harmony_multiturn` | [`harmony_parser/__init__.py`](mas_r1_reasoner/rewards/utils/harmony_parser/__init__.py#L42) |
| Reward manager skips MAS code execution (agents already ran during rollout); episodes are scored as direct answers | [`HarmonyRewardManager.__call__`](mas_r1_reasoner/rewards/harmony_reward_manager.py#L158) |
| GRPO advantage uses `loss_mask` (`multi_turn=True`) | [`fit` → `compute_advantage`](mas_r1_reasoner/trainer/mas_r1_trainer.py#L984) |
| Actor loss uses `loss_mask` (`meta_info['multi_turn']`) | [`fit` → `update_actor`](mas_r1_reasoner/trainer/mas_r1_trainer.py#L1003) |
| Multi-turn metrics logged for training and validation | [training](mas_r1_reasoner/trainer/mas_r1_trainer.py#L800), [validation](mas_r1_reasoner/trainer/mas_r1_trainer.py#L344) |

## Step 3: system prompt for the plan-execute protocol

| What | Where |
|---|---|
| System prompt: per-turn protocol (think, then either call agents or finish), Hermes `<tool_call>` format, parallel calls, revise after results, `<answer>` only in the final turn without tool calls, turn limit | [`system_prompt/harmony_multiturn.py`](mas_r1_reasoner/agents/prompts/system_prompt/harmony_multiturn.py#L1) |
| Developer prompt: gpt-oss-120b sub-agent description, tool argument semantics (`agent_input`, `debate_roles`), agent list | [`developer_prompt/harmony_multiturn.py`](mas_r1_reasoner/agents/prompts/developer_prompt/harmony_multiturn.py#L1) |
| User prompt with three worked episodes: solve directly; delegate then verify; parallel sub-tasks | [`user_prompt/propose_harmony_multiturn.py`](mas_r1_reasoner/agents/prompts/user_prompt/propose_harmony_multiturn.py#L1) |
| `get_prompt` wiring: prompt selection and placeholder filling. Agent cards keep name/description/arguments and drop the implementation source (the tool environment runs the agents) | [user prompt branch](mas_r1_reasoner/agents/common.py#L245), [system/developer branch](mas_r1_reasoner/agents/common.py#L344) |
| `harmony_multiturn` accepted as a harmony prompt type when building the batch | [`helper.py`](mas_r1_reasoner/trainer/utils/helper.py#L187) |

The full rendered prompt (system + user + developer + tool schemas, Qwen chat template) is **2,195 tokens**.

## Step 4: verl configs for the multi-turn architecture

| What | Where |
|---|---|
| Multi-turn training config (inherits `grpo_trainer.yaml`) | [`grpo_trainer_multiturn.yaml`](mas_r1_reasoner/configs/grpo_trainer_multiturn.yaml#L1) |
| Note at the (still disabled) `multi_turn` block of the single-shot config | [`grpo_trainer.yaml`](mas_r1_reasoner/configs/grpo_trainer.yaml#L144) |
| Tool registry: 4 agents, shared runtime block (gpt-oss-120b, reasoning effort low, 64 concurrent calls, 900 s timeout, 6000-char observations) | [`tool_config/mas_agents.yaml`](mas_r1_reasoner/configs/tool_config/mas_agents.yaml#L6), [tools](mas_r1_reasoner/configs/tool_config/mas_agents.yaml#L21) |
| Mock tool registry (no API calls) for smoke tests | [`tool_config/mas_agents_mock.yaml`](mas_r1_reasoner/configs/tool_config/mas_agents_mock.yaml#L1) |
| 8-GPU Slurm training script | [`train_multiturn.sh`](train_multiturn.sh#L12) |
| 2-GPU smoke test (mock agents, 2 steps) | [`smoke_test_multiturn.sh`](smoke_test_multiturn.sh#L12) |
| DeepScaleR-Preview download into the raw-data parquet format | [`scripts/prepare_deepscaler.py`](scripts/prepare_deepscaler.py#L2) |

Key values in `grpo_trainer_multiturn.yaml`:

| Setting | Value | Why |
|---|---|---|
| `rollout.name` | `sglang_async` | verl's rollout with tool calling |
| `multi_turn.enable / max_turns / format` | `True / 6 / qwen` | Qwen2.5 tool-response format |
| `multi_turn.tool_config_path` | `configs/tool_config/mas_agents.yaml` | launch script passes an absolute path |
| `data.max_prompt_length / max_response_length` | 8192 / 16384 | the response holds every turn plus every agent result; total fits the 32k context |
| `rollout.n`, `val_kwargs.n` | 8, 8 | GRPO group size; Avg@8 validation |
| `data.train_batch_size`, `ppo_mini_batch_size` | 32, 32 | one policy update per step |
| `use_dynamic_bsz`, `ppo_max_token_len_per_gpu` | True, 32768 | long multi-turn sequences |
| `rollout.tensor_model_parallel_size` | 2 | 4 SGLang engines on 8 GPUs |
| `azr.problem_type` | `harmony_multiturn` | selects step-3 prompts and the step-2 parser |
| reward weights | answer 1.0, execution 0.0 | no code execution in multi-turn mode |

---

## ToolOrchestra reward, W&B and checkpoint location

Reward (ToolOrchestra paper, Sec. 3; same formula as their `generation_quick3.py`): for each episode,
\(M = [\text{calls per agent}, r_\text{outcome}, -\$\text{cost}, -\text{latency}]\), min-max normalised over the
8 episodes of the same question; \(R = P \cdot M_\text{norm}\) if the answer is correct, else 0.
Default \(P\): accuracy 1.0, cost 0.1, latency 0.1, per-agent 0 (`azr.mas_r1.toolorchestra_reward`).
Cost = OpenRouter `usage.cost` of every agent call + orchestrator tokens at \$0.2/M; latency = wall-clock time
agents were running (parallel calls in one turn counted once). Validation keeps the 0/1 outcome reward.

| What | Where |
|---|---|
| Reward + metrics (`toolorchestra/*`) | [`toolorchestra_reward.py`](mas_r1_reasoner/rewards/utils/toolorchestra_reward.py) |
| Applied after outcome scoring | [`HarmonyRewardManager.__call__`](mas_r1_reasoner/rewards/harmony_reward_manager.py) |
| Per-call cost/latency returned by every agent tool | [`MASAgentTool.execute`](mas_r1_reasoner/tools/mas_agent_tools.py) |
| Per-call OpenRouter cost accumulator | [`call_usage`](mas_r1_reasoner/agents/sampler/openrouter_completion_sampler.py) |
| verl rollout exports tool metrics (`non_tensor_batch["tool_metrics"]`) | [`async_sglang_rollout.py`](verl/verl/workers/rollout/sglang_rollout/async_sglang_rollout.py) |
| Config (preference, prices, W&B project, checkpoint root) | [`grpo_trainer_multiturn.yaml`](mas_r1_reasoner/configs/grpo_trainer_multiturn.yaml) |

Plots: W&B project [`multi-agent-experiments`](https://wandb.ai/debajoym98-texas-a-m-university/multi-agent-experiments).
Outputs of run `<RUN_NAME>`: `/scratch/project/prj-02-llm-reasoning-shakkottai/Multi_agent_systems/<RUN_NAME>/`
(`checkpoints/`, `val/`, `hydra/`, `wandb/`, `run.log`).

Not adopted from ToolOrchestra: per-example preference vectors with a natural-language preference in the prompt
(DeepScaleR has none; one fixed \(P\) is used), and their filtering of groups with reward std ≤ 0.1.

## OpenRouter setup (gpt-oss-120b sub-agents)

| What | Where |
|---|---|
| Sampler: OpenAI client on `https://openrouter.ai/api/v1`, `reasoning: {effort}`, provider routing, per-call token/cost accounting (`OpenRouterCompletionSampler.usage`) | [`openrouter_completion_sampler.py`](mas_r1_reasoner/agents/sampler/openrouter_completion_sampler.py) |
| Shared retry loop now calls an overridable `_create_completion` | [`together_completion_sampler.py`](mas_r1_reasoner/agents/sampler/together_completion_sampler.py) |
| Sampler type registration (`type: OpenRouterCompletionSampler`, optional `provider`) | [`create_model_sampler_map`](mas_r1_reasoner/data_precessor/BaseDatasetProcessor.py#L21) |
| Agents use it in the multi-turn run (provider order: Together first, fallbacks allowed) | [`mas_agents.yaml`](mas_r1_reasoner/configs/tool_config/mas_agents.yaml#L6), [`grpo_trainer_multiturn.yaml`](mas_r1_reasoner/configs/grpo_trainer_multiturn.yaml) |
| One-call check (also run automatically by `train_multiturn.sh` before training) | [`scripts/check_openrouter.py`](scripts/check_openrouter.py) |
| `eval_math.sh` switches to this sampler when only `OPENROUTER_API_KEY` is set | [`eval_math.sh`](eval_math.sh) |

## How to run

```bash
# API key (stored once, never pasted in chat)
read -rsp "OpenRouter key: " KEY && echo && printf '%s\n' "$KEY" > ~/.openrouter_api_key && chmod 600 ~/.openrouter_api_key && unset KEY
OPENROUTER_API_KEY=$(cat ~/.openrouter_api_key) /scratch/user/debajoym98_tamu.edu/conda_envs/mas-orchestra/bin/python scripts/check_openrouter.py

sbatch smoke_test_multiturn.sh     # 2 GPUs, mock agents, 2 training steps: checks the whole loop for free
sbatch train_multiturn.sh          # 8 GPUs, real gpt-oss-120b agents; downloads DeepScaleR on first run
sbatch train_multiturn.sh trainer.total_training_steps=100   # any Hydra override can be appended
```

Watch `multi_turn/train_turns_mean`, `multi_turn/train_agent_calls_mean`, `multi_turn/train_no_agent_fraction`
and `multi_turn/train_usage_<Agent>` to see how the orchestration policy evolves.

## What was verified (CPU only)

- All changed Python files compile; both shell scripts pass `bash -n`.
- Hydra composes `grpo_trainer_multiturn` with the values above, and the tool registry resolves to the four agents.
- `get_prompt("...")` for `harmony_multiturn` returns 3 prompts with no unfilled placeholders and no leaked implementation code; rendered with tools it is 2,195 tokens.
- Mock tool calls: CoTAgent (whole question), SCAgent (sub-task template), DebateAgent (custom roles) return observations; invalid debate roles return an error message instead of crashing.
- Parser returns the orchestrator's final `<answer>` (204) and ignores an agent's `<answer>` (99) inside `<tool_response>`.

**GPU smoke test (`smoke_test_multiturn.sh`, Slurm job 1114836, 2 GPUs, mock agents): passed.**
Rollout → reward → GRPO advantage → actor update → validation all ran end to end. The untrained
Qwen2.5-7B-Instruct already uses the protocol: 2.5 turns and 1.75 agent calls per episode
(CoTAgent 79%, SCAgent 14%, DebateAgent 7%), no truncated responses, prompt ≈ 2.3k tokens.
Two fixes were needed: an event loop for SGLang under uvloop
([`AsyncSGLangRollout.__init__`](verl/verl/workers/rollout/sglang_rollout/async_sglang_rollout.py#L105)), and
`val_kwargs.n ≥ 2`, because MAS-Orchestra's single-response validation path (`val_kwargs.n=1`) has a key-indexing bug
that also affects single-shot runs.

## Known limitations and risks

- **verl 0.3.1 multi-turn code is early.** The SGLang multi-turn rollout in this bundled verl is the first version of that feature; the smoke test is the way to find integration bugs before a long run.
- **Agents run on the TP-rank-0 worker of each SGLang engine.** All tool calls for that engine's episodes go through one process; `max_concurrent_calls` (64) bounds API concurrency per engine (4 engines → up to 256 concurrent calls).
- **Prompt truncation.** Prompts longer than `max_prompt_length` are truncated by the rollout with a warning; the current prompt is ~2.2k tokens, so this only matters for very long questions.
- **Responses reordering (observation).** In training, the reward reformatting picks the `responses` tensor by per-question index while other tensors (including `loss_mask` and `input_ids`) are reordered by question. The actor computes log-probs from `input_ids`, so training is unaffected, but anything reading `responses` directly after reformatting should keep this in mind.
- **Agent stdout is verbose.** The MAS-Orchestra agents print every sub-agent message; expect large logs.

## Full-run schedule

| Setting | Value | Where |
|---|---|---|
| Training steps | 180 (`trainer.total_training_steps`) | [`grpo_trainer_multiturn.yaml`](mas_r1_reasoner/configs/grpo_trainer_multiturn.yaml#L88) |
| Validation | step 0, then every 25 steps on AIME24, Avg@8 (240 episodes) | [`grpo_trainer_multiturn.yaml`](mas_r1_reasoner/configs/grpo_trainer_multiturn.yaml#L90), `VAL_FILES` in [`train_multiturn.sh`](train_multiturn.sh#L27) |
| Checkpoints | every 50 steps, only the latest kept (`max_actor_ckpt_to_keep: 1`), plus the final step | [`grpo_trainer_multiturn.yaml`](mas_r1_reasoner/configs/grpo_trainer_multiturn.yaml#L91) |

## Next steps

1. Run `smoke_test_multiturn.sh` and fix any runtime issues.
2. Prepare DeepScaleR (done automatically by `train_multiturn.sh`).
3. OpenRouter: done (see "OpenRouter setup"). Compute nodes need outbound HTTPS to `openrouter.ai`.
4. Budget-matched comparison with single-shot MAS-Orchestra: log sub-agent tokens/cost per step for both runs and compare at equal API spend.
