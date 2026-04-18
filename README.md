<div align="center">

<h1 align="center">
  <!-- <img src="./figures/mas_zero.png" alt="MAS-Orchestra Icon" height="45" style="vertical-align: middle;"> -->
  MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks
</h1>
<!-- #  Absolute Zero:  Reinforced Self-play Reasoning with Zero Data -->

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.14652)    [![Project Page](https://img.shields.io/badge/Project%20Page-blue?style=for-the-badge&logo=snowflake&logoColor=white&labelColor=black)](https://vincent950129.github.io/mas-design/mas_r1/)    [![Live Demo](https://img.shields.io/badge/Live%20Demo-ff6f61?style=for-the-badge&logo=rocket&logoColor=white)](https://vincent950129.github.io/mas-design/mas_r1/demo)    [![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/SalesforceAIResearch/MAS-Orchestra)    [![MAS Bench](https://img.shields.io/badge/MAS_Bench-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/Salesforce/MASBench)

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">🔗 Links</a> •
    <a href="#demo" style="text-decoration: none; font-weight: bold;">🎬 Demo</a> •
    <a href="#todo" style="text-decoration: none; font-weight: bold;">📝 Conceptual Overview</a> •
    <!-- <a href="#algorithm-flow" style="text-decoration: none; font-weight: bold;">⚙️ Algorithm Flow</a> • -->
    <a href="#results" style="text-decoration: none; font-weight: bold;">📊 Results</a>
  </p>
  <p>
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#training" style="text-decoration: none; font-weight: bold;">🏋️ MAS-Orchestra</a> •
    <a href="#masbench" style="text-decoration: none; font-weight: bold;">📐 MASBench</a> •
    <a href="#case-inspection" style="text-decoration: none; font-weight: bold;">🔍 Case Inspection</a>
  </p>
  <p>
    <a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a> •
    <a href="#contact" style="text-decoration: none; font-weight: bold;">📧 Contact</a>
    <!-- <a href="#star-history" style="text-decoration: none; font-weight: bold;">📈 Star History</a> -->
  </p>
</div>

</div>

![Conceptual Overview](figures/overview.png)

<!-- ============================================== -->

- **[04/16/2026]** We present the Live Demo! [[Live Demo](https://vincent950129.github.io/mas-design/mas_r1/demo)]
- **[01/29/2026]** We present the **<span style="font-variant: small-caps;">MAS-Orchestra</span>** [[Project Page](https://vincent950129.github.io/mas-design/mas_r1/) | [Paper](https://arxiv.org/abs/2601.14652) | [Code](https://github.com/SalesforceAIResearch/MAS-Orchestra)]
 <!-- | [Model(s)](https://huggingface.co/collections/andrewzh/absolute-zero-reasoner-68139b2bca82afb00bc69e5b) | [Logs](https://wandb.ai/andrewzhao112/AbsoluteZeroReasoner)]. -->

<!-- ============================================== -->
<div align="left">
  <h1 id="links">🔗 Links</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

- 🏠 [[Project Page]](https://vincent950129.github.io/mas-design/mas_r1/)
- 📜 [[Paper]](https://arxiv.org/abs/2601.14652)
- 💻 [[Code]](https://github.com/SalesforceAIResearch/MAS-Orchestra)
- 🚀 [[Demo]](https://vincent950129.github.io/mas-design/mas_r1/demo)
<!-- ============================================== -->
<div align="left">
  <h1 id="demo">🎬 Demo</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

<p align="center">
  <a href="https://vincent950129.github.io/mas-design/mas_r1/demo">
    <img src="https://img.shields.io/badge/Try%20Live%20Demo-ff6f61?style=for-the-badge&logo=rocket&logoColor=white" alt="Try Live Demo">
  </a>
</p>

<p align="center">
  <a href="https://vincent950129.github.io/mas-design/mas_r1/demo">
    <img src="figures/demo_teaser.png" alt="MAS-Orchestra Demo Video" width="80%">
  </a>
</p>

<!-- ============================================== -->
<div align="left">
  <h1 id="results">📊 Results</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>


<p align="center">
  <img src="figures/pareto_front.png" alt="Pareto Front: Accuracy vs. Cost" width="560">
</p>

<p align="center"><em>Accuracy vs. cost Pareto front. MAS-Orchestra achieves Pareto-optimal performance with the highest accuracy at low cost.</em></p>

MAS-Orchestra achieves state-of-the-art performance across both IID and OOD benchmarks while maintaining Pareto-optimal cost efficiency.

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4" align="center">IID Tasks</th>
      <th rowspan="2">OOD Task<br>GPQA</th>
    </tr>
    <tr>
      <th>AIME24</th>
      <th>AIME25</th>
      <th>HotpotQA</th>
      <th>BrowseComp+</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="6" align="center"><strong>Standalone Agents</strong></td></tr>
    <tr><td>CoTAgent</td><td>50.00</td><td>45.00</td><td>33.56</td><td>1.12</td><td>60.54</td></tr>
    <tr><td>SCAgent</td><td>57.50</td><td>51.67</td><td>35.50</td><td>0.75</td><td>62.88</td></tr>
    <tr><td>DebateAgent</td><td>62.08</td><td>57.50</td><td>36.88</td><td>0.81</td><td>64.14</td></tr>
    <tr><td>ReflexionAgent</td><td>60.83</td><td>50.42</td><td>36.63</td><td>1.00</td><td>62.37</td></tr>
    <tr><td>DeepResearchAgent</td><td>—</td><td>—</td><td>46.44</td><td>8.56</td><td>—</td></tr>
    <tr><td colspan="6" align="center"><strong>SoTA Inference-time Orchestration</strong></td></tr>
    <tr><td>AFlow</td><td>62.50</td><td>53.33</td><td>—</td><td>—</td><td>65.43</td></tr>
    <tr><td>MaAS</td><td>32.50</td><td>40.83</td><td>—</td><td>—</td><td>40.78</td></tr>
    <tr><td>MAS-Zero</td><td colspan="5" align="center">No valid MAS generated with 7B orchestrator</td></tr>
    <tr><td colspan="6" align="center"><strong>SoTA Public Training-time Orchestration</strong></td></tr>
    <tr><td>MAS-GPT</td><td>58.75</td><td>43.33</td><td>—</td><td>—</td><td>63.51</td></tr>
    <tr><td>ToolOrchestra</td><td>23.33</td><td>11.25</td><td>37.44</td><td>1.38</td><td>29.80</td></tr>
    <tr><td colspan="6" align="center"><strong>SoTA LLM as Orchestrator</strong></td></tr>
    <tr><td>GPT-5</td><td>55.00</td><td>47.72</td><td>25.87</td><td>0.50</td><td>59.01</td></tr>
    <tr><td>Claude-Sonnet-4.5</td><td>45.56</td><td>35.00</td><td>38.00</td><td>0.50</td><td>21.72</td></tr>
    <tr><td colspan="6" align="center"><strong>Ours</strong></td></tr>
    <tr><td><strong>MAS-Orchestra</strong></td><td><strong>66.25</strong></td><td><strong>61.25</strong></td><td><strong>49.00</strong></td><td><strong>11.00</strong></td><td><strong>65.21</strong></td></tr>
  </tbody>
</table>

<p align="center"><em>Performance comparison across IID and OOD benchmarks. MAS-Orchestra achieves the best results on all tasks.</em></p>


<!-- ============================================== -->
<div align="left">
  <h1 id="getting-started">✨ Getting Started</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

## 🎄 Environment Setup
```bash
conda create -n mas-orchestra python=3.10
conda activate mas-orchestra

apt update && apt install -y wget curl

cd ./verl
./install.sh
pip install --no-deps -e .
pip install ray==2.49.2 --force-reinstall
pip install protobuf==4.25.8 --force-reinstall
pip install together
pip install math-verify[antlr4_13_2]
pip install antlr4-python3-runtime==4.9.3

pip install langchain-core langchain-together langchain-community duckduckgo-search tavily-python pydantic ddgs langchain_brightdata bs4
pip install pyserini faiss-gpu
pip install git+https://github.com/texttron/tevatron.git

```

<!-- ## 💾 Data Processing
### Process evaluation data on CruxEval / LiveCodeBench Execution during AZR Self-play
```bash
python -m absolute_zero_reasoner.data_construction.process_code_reasoning_data
``` -->

## 📦 (Optional) Download Trained Orchestrators

| Task | Model |
|------|-------|
| Math (AIME) | [harmony-grpo-7b-global-step-180](https://huggingface.co/ZixuanKe/harmony-grpo-7b-global-step-180) |
| HotpotQA | [harmony-medium-grpo-7b-hotpot-global-step-250](https://huggingface.co/ZixuanKe/harmony-medium-grpo-7b-hotpot-global-step-250) |
| BrowseComp+ | [harmony-medium-grpo-7b-browse-comp-plus-global-step-140](https://huggingface.co/ZixuanKe/harmony-medium-grpo-7b-browse-comp-plus-global-step-140) |

## 🔌 Use via API

The trained orchestrators can also be called via API without downloading. Check out the notebook below for usage:

<p align="center">
  <a href="https://colab.research.google.com/drive/1tOqZ7J3dA8mQPDdYT6M7y0pJqiX11eKO?usp=sharing">
    <img src="https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open in Colab">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscern-stroller-recycling.ngrok-free.dev%2Fcounts&query=%24.models.math&label=%F0%9F%93%8A%20Math%20(AIME)&color=blue&style=for-the-badge" alt="Math API Calls">
  <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscern-stroller-recycling.ngrok-free.dev%2Fcounts&query=%24.models.hotpotqa&label=%F0%9F%93%8A%20HotpotQA&color=blue&style=for-the-badge" alt="HotpotQA API Calls">
  <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscern-stroller-recycling.ngrok-free.dev%2Fcounts&query=%24.models.browsecomp&label=%F0%9F%93%8A%20BrowseComp%2B&color=blue&style=for-the-badge" alt="BrowseComp+ API Calls">
  <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscern-stroller-recycling.ngrok-free.dev%2Fcounts&query=%24.total&label=%F0%9F%93%8A%20Total%20Calls&color=brightgreen&style=for-the-badge" alt="Total API Calls">
</p>

<p align="center">
  <a href="https://discern-stroller-recycling.ngrok-free.dev/charts">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://discern-stroller-recycling.ngrok-free.dev/chart.svg?theme=dark&legend=top-left" />
      <source media="(prefers-color-scheme: light)" srcset="https://discern-stroller-recycling.ngrok-free.dev/chart.svg?theme=light&legend=top-left" />
      <img alt="Call History Chart" src="https://discern-stroller-recycling.ngrok-free.dev/chart.svg?theme=light&legend=top-left" width="80%" />
    </picture>
  </a>
</p>

<!-- ============================================== -->
<div align="left">
  <h1 id="training">🏋️ MAS-Orchestra</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

## ♟️ Example Training Script
<!-- 3b models need 2 X 80gb GPUs, 7/8b models need 4 X 80gb, 14b requires 8 X 80gb
```bash
bash scripts/selfplay/<7b|14b|coder3b|coder7b|coder14b|llama>.sh
``` -->
```bash
export OPENAI_API_KEY={YourKey}
export TOGETHER_API_KEY={YourKey}
export WANDB_API_KEY={YourKey}
LOG_FILE={YourLogFile}

python -u -m mas_r1_reasoner.main_mas_r1 \
    --config-path=configs \
    --config-name=grpo_trainer \
    data.max_prompt_length=15000 \
    data.max_validation_prompt_length=15000 \
    data.val_files=data/browse_comp/test_subset_200.parquet \
    data.train_files=data/browse_comp/train_subset_1066.parquet \
    azr.mas_r1.use_llm_judge=True \
    data.raw_data=True \
    data.train_batch_size=64 \
    actor_rollout_ref.rollout.n=32 \
    azr.mas_r1.execution_success_weight=0.0 \
    azr.mas_r1.final_answer_weight=1.0 \
    azr.mas_r1.agent.model_name=gpt-oss-120b\
    azr.mas_r1.multiply_processes=0 \
    azr.mas_r1.max_ray_workers=1 \
    azr.problem_type=harmony_medium \
    azr.mas_r1.agent.init_archive=['COT','COT_SC','Reflexion','LLM_debate','WebSearch'] \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    trainer.experiment_name=harmony_medium_grpo_7b_gpt_oss_120b_browse_comp_plus \
    $@ 2>&1 | tee -a "$LOG_FILE"
```

<!-- ============================================== -->
<div align="left">
  <h1 id="masbench">📐 MASBench</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

[MASBench](https://huggingface.co/datasets/Salesforce/MASBench) is a controlled benchmark that characterizes tasks along **five structural axes** to rigorously study when and why multi-agent systems outperform single-agent systems.

### A Five-Axis Evaluation Framework

<p align="center">
  <img src="figures/five_axis.png" alt="Five-Axis Evaluation Framework" width="80%">
</p>

| Axis | Definition |
|------|-----------|
| **Depth** | Length of the longest dependency chain |
| **Horizon** | Number of intermediate sub-tasks whose answers are needed |
| **Breadth** | Maximum in-degree, i.e., maximum dependencies of a sub-task |
| **Parallel** | Number of independent sub-task components in the task |
| **Robustness** | Number of sub-tasks with adversarial attacks |

### Benchmark Statistics

<p align="center">
  <img src="figures/masbench_stats.png" alt="MASBench Statistics" width="70%">
</p>

The benchmark covers all five axes with axis values ranging from 2 to 12, and provides axis-specific training and test splits. The dataset is available on [Hugging Face](https://huggingface.co/datasets/Salesforce/MASBench).

<!-- ============================================== -->
<div align="left">
  <h1 id="case-inspection">🔍 Case Inspection</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

Browse real, generated multi-agent designs produced by MAS-Orchestra. Each example shows the full orchestration trace — how the orchestrator decomposes a task, selects sub-agents, and aggregates their outputs.

<p align="center">
  <img src="figures/case_teaser.png" alt="Case Teaser" width="80%">
</p>

<p align="center">
  <a href="https://vincent950129.github.io/mas-design/mas_r1/#generated-mas-examples">
    <img src="https://img.shields.io/badge/Browse_More_Examples-on_Project_Page-5755A3?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Browse Examples">
  </a>
</p>

**Highlights from the case studies:**

- **AIME24 (Low DoM):** MAS-Orchestra learns to delegate entirely to a single strong sub-agent (100% delegation after 20 training steps), dynamically selecting ReflexionAgent or DebateAgent — the best-performing standalone baselines.
- **BrowseComp+ (High DoM):** MAS-Orchestra generates substantially more sub-agents, invoking SearchAgent with 3–4 parallel search processes per question.
- **General Pattern:** MAS-Orchestra dynamically adapts to each task by proposing MAS designs that align with the underlying sub-task structure and delegating execution to the most effective agent configurations.

<!-- ============================================== -->
<div align="left">
  <h1 id="mas-series">🤖 Check Out Our MAS Series</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

- **[MAS-Zero](https://github.com/SalesforceAIResearch/MAS-Zero)**: Designing Multi-Agent Systems with Zero Supervision — an inference-time self-refinement framework for automatic MAS design.
- **[MAS-ProVe](https://github.com/Wang-ML-Lab/MAS-ProVe)**: Understanding the Process Verification of Multi-Agent Systems — analysis of process verification for multi-agent systems.
- **[SkillOrchestra](https://github.com/jiayuww/SkillOrchestra)**: Learning to Route Agents via Skill Transfer — skill-based agent routing.
- **[LLM Reasoning Survey](https://llm-reasoning-ai.github.io/)**: A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems.

<!-- ============================================== -->
<div align="left">
  <h1 id="citation">🎈 Citation</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

If you find MAS-Orchestra helpful, please consider starring this repo and citing our work. We would be very grateful!

```bibtex
@misc{Ke2026MASOrchestra,
        title        = {MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks},
        author       = {Zixuan Ke and Yifei Ming and Austin Xu and Ryan Chin and Xuan-Phi Nguyen and Prathyusha Jwalapuram and Semih Yavuz and Caiming Xiong and Shafiq Joty},
        year         = {2026},
        eprint       = {2601.14652},
        archivePrefix= {arXiv},
        primaryClass = {cs.AI},
        note         = {Preprint; Work in Progress},
      }
```
<!-- ============================================== -->
<div align="left">
  <h1 id="acknowledgement">🌻 Acknowledgement</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>
<!-- This project received help from many researchers at Salesforce AI Research. The code is adapted from the [ADAS](https://github.com/ShengranHu/ADAS). During development, we also referred to [simple-evals](https://github.com/openai/simple-evals), [MaAS](https://github.com/bingreeky/MaAS), and [AFlow](https://github.com/FoundationAgents/AFlow).   -->

This project received help from many researchers at Salesforce AI Research. We also thank the authors of [verl](https://github.com/volcengine/verl) for their excellent contributions to the community!

<!-- ============================================== -->
<div align="left">
  <h1 id="contact">📧 Contact</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

Feel free to contact Zixuan Ke via email: zixuan.ke@salesforce.com

<!-- ============================================== -->
<!-- <div align="left">
  <h1 id="star-history">📈 Star History</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div> -->

<!-- [![Star History Chart](https://api.star-history.com/svg?repos=LeapLabTHU/Absolute-Zero-Reasoner&type=Date)](https://www.star-history.com/#LeapLabTHU/Absolute-Zero-Reasoner&Date) -->