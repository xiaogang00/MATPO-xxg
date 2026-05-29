<div align="center">

# MATPO-PR: Multi-Agent Tool-Integrated Policy Optimization with Process Reward

Train Multiple Agent Roles Within a Single LLM via Reinforcement Learning with Process Reward.
MATPO-PR is an upgraded implementation of MATPO.

<!-- [![arXiv](https://img.shields.io/badge/arXiv-Coming_Soon.svg)](https://arxiv.org/pdf/2510.04678)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Code](https://img.shields.io/badge/code-GitHub-black.svg)](https://github.com/xiaogang00/MATPO-PR) -->

<!-- <hr> -->
<div align="center">

[![Models](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/veggiebird/MATPO-PR-14B-05)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/datasets/veggiebird/MATPO-data)
[![Paper](https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.04678)
[![Github](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/xiaogang00/MATPO-PR)
</div>


</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/score_lines.png" width="660px" alt="GAIA Results"><br>
        <em>GAIA, FRAMES, WebWalkerQA Results</em>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/reward_heatmap_baseline_val2.png" width="330px" alt="Visualization of process rewards on GAIA for MATPO"><br>
        <em>Visualization of process rewards on GAIA for MATPO</em>
      </td>
      <td align="center">
        <img src="assets/reward_heatmap_ours_val2.png" width="330px" alt="Visualization of process rewards on GAIA for MATPO-PR"><br>
        <em>Visualization of process rewards on GAIA for MATPO-PR</em>
      </td>
    </tr>
  </table>
</div>


<p align="center">
  <em>
MATPO-PR enhances the original [MATPO-PR](https://github.com/mzf666/MATPO) by integrating process rewards (a core formulation that can be adopted by arbitrary state-of-the-art agentic systems), achieving an 7.81% relative improvement over MATPO on GAIA-text, FRAMES, and WebWalker-QA in this paper.
  </em>
</p>

## News & Updates

- **[2026-Apr-08]** Code and training scripts released
- **[2026-Apr-06]** Arxiv Paper released

## Overview

**MATPO** (Multi-Agent Tool-Integrated Policy Optimization) is a novel reinforcement learning framework that enables training multiple specialized agent roles (planner and worker agents) within a single large language model. 


**MATPO-PR** (Multi-Agent Tool-Integrated Policy Optimization with Process Reward) improves upon the original MATPO by introducing process rewards for reinforcement learning. As this formulation is highly compatible with various agentic frameworks, our future roadmap includes extending this capability to platforms like OpenClaw.


## Key Features

- **Multi-Agent-in-One-Model**: Train planner and worker agents within a single LLM using role-specific system prompts
- **Principled Credit Assignment**: Extends GRPO with theoretically grounded reward distribution across planner and worker rollouts
- **Process Reward and Return**: Compute granular process rewards and returns within RL loops to refine intermediate logic and boost end-to-end success rates
- **Easy Integration**: Built on top of [veRL](https://github.com/volcengine/verl), compatible with existing RL training frameworks
- **Robust Training**: More stable learning curves compared to single-agent approaches, especially with noisy tool responses
- **Infrastructure Efficient**: No need for deployment of separate models or additional rollout engines


## MATPO-PR Architecture

MATPO-PR employs a hierarchical multi-agent framework where a single LLM serves multiple roles:

```
User Query → Planner Agent → Subtask 1 → Worker Agent → Result 1 → Reward 1 → Return 1
                           → Subtask 2 → Worker Agent → Result 2 → Reward 2 → Return 2
                           → ...
                           → Final Answer → Final Answer Reward
```


### Multi-Agent Rollout Process

1. **Planner Agent**: 
   - Receives user query with planner-specific system prompt
   - Generates high-level plan and decomposes it into subtasks
   - Delegates subtasks to worker agents
   - Synthesizes worker responses into final answer
   - Process rewards are calculated following each sub-agent execution (by the main agent), determined by the correctness of the answer, which is made based on the accumulated context at each step

2. **Worker Agent**:
   - Receives subtask with worker-specific system prompt
   - Performs multi-turn tool-integrated planning (search, scrape, analyze)
   - Returns summarized result to planner
   - Maintains isolated context to prevent token overflow

3. **Credit Assignment**:
   - Final answer accuracy determines the corresponding "final answer reward"
   - The process rewards are derived from the accuracy of intermediate answers generated at various execution steps
   - Process rewards are accumulated into process returns to conduct the GRPO
   - Advantages are obtained by normalizing across all planner-worker rollout groups, in terms of process returns and "final answer reward"
   - Gradient flows to both planner actions and worker actions proportionally

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/MATPO_rollout_compact.png" width="330px" alt="GAIA Results"><br>
        <em>Rollout and Reward of MATPO</em>
      </td>
      <td align="center">
        <img src="assets/MATPO_PR_rollout.png" width="330px" alt="GAIA Results"><br>
        <em>Rollout and Reward of MATPO-PR</em>
      </td>
    </tr>
  </table>
</p>

## Quick Start

This section provide a quick start guide from PyTorch basic docker image.

Prerequisites:
- You can train a Qwen3-4B model with 1 x (8 x 80G-A800) NVIDIA A800 80GB GPUs. 
- NVIDIA Driver.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Step 1**: Clone the repository.
```bash
cd YOUR_WORKING_DIR
git clone https://github.com/xiaogang00/MATPO-PR.git
cd MATPO-PR
```

**Step 2**: Download the training and testing datasets to the `data` directory. The prerpocessed datasets can be downloaded [here](https://huggingface.co/datasets/veggiebird/MATPO-data).

**Step 3**: Launch the PyTorch docker container. Make sure the mounted directories are writable.
```bash
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
docker run -it \
    --gpus all \
    --shm-size 16gb \
    --name matpo \
    -v YOUR_WORKING_DIR/MATPO-PR:/workspace/MATPO-PR:rw \
    -v YOUR_WORKING_DIR/models:/workspace/models \
    -w /workspace/MATPO-PR \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    /bin/bash
```
Now you are in the PyTorch docker container.

**Step 4**:  Start to install the python dependencies inside the docker container.
```bash
# Execute the following commands inside the docker container

# Create a new conda environment for MATPO-PR
conda create -n matpo python==3.10 -y
conda init bash
source /opt/conda/etc/profile.d/conda.sh
conda activate matpo

# Install the python dependencies
# You are highy recommended to execute the commands in the install.sh script one by one.
source /workspace/MATPO-PR/examples/sglang_multiturn/install.sh
```

**Step 5**: Setup Node.js for Serper API support inside the docker container. 
> MCP (Model Context Protocol) requires Node.js to run MCP servers. Node.js version 18+ is recommended for optimal compatibility with MCP tools. Configure the Node.js paths and HTTP / HTTPS proxies (if necessary) in the `examples/sglang_multiturn/run_in_docker/launch.sh` script properly.
```bash
# Install Node.js
apt-get update && apt-get install -y wget xz-utils git
cd /workspace
wget https://nodejs.org/dist/v24.2.0/node-v24.2.0-linux-x64.tar.xz
cd -

NODEJS_HOME=/workspace/nodejs
mkdir -p $NODEJS_HOME
tar -xf /workspace/node-v24.2.0-linux-x64.tar.xz -C $NODEJS_HOME

# Configure Node.js environment variables
export PATH=$NODEJS_HOME/bin:$PATH
export NODE_SHARED=$NODEJS_HOME/node-shared/node_modules
export PATH=$NODE_SHARED/.bin:$PATH

# Verify Node.js installation
node --version
npm --version

# Install Serper MCP Server
mkdir -p $target_path/node-shared
cd $target_path/node-shared
npm init -y
npm install serper-search-scrape-mcp-server

# Back to MATPO repository
cd /workspace/MATPO-PR 
```

If you counter any issues during the environment setup, you can refer to the `examples/sglang_multiturn/pip_list_reference.txt` for the expected python dependencies list and check the installation process step by step.

**Step 6**: Train a Qwen3-14B model with MATPO on the MuSiQue dataset and evaluate on the GAIA-text datasets using computation platforms with multiple GPU nodes. Remember to adjust the directory paths in `examples/sglang_multiturn/launch.sh` accordingly. 

```bash
# tested on 16 x (8 x 80G-A800) nodes

export SERPER_API_KEY="YOUR_SERPER_API_KEY" && \
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" && \
export WANDB_API_KEY="YOUR_WANDB_API_KEY" && \
export SINGLENODE=true && \
export RAY_DEBUG=legacy && \
export HYDRA_FULL_ERROR=1 && \
source YOUR_CONDA_PATH activate matpo && \
cd YOUR_PROJECT_PATH && \
bash examples/sglang_multiturn/launch.sh \
    examples/sglang_multiturn/qwen3-14b_musique_MATPO.sh
```

Similarly, train the model with MATPO-PR as the following.

```bash
# tested on 16 x (8 x 80G-A800) nodes

export SERPER_API_KEY="YOUR_SERPER_API_KEY" && \
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" && \
export WANDB_API_KEY="YOUR_WANDB_API_KEY" && \
export SINGLENODE=true && \
export RAY_DEBUG=legacy && \
export HYDRA_FULL_ERROR=1 && \
source YOUR_CONDA_PATH activate matpo && \
cd YOUR_PROJECT_PATH && \
bash examples/sglang_multiturn/launch.sh \
    examples/sglang_multiturn/qwen3-14b_musique_MATPO_turn.sh
```


Evaluate a trained MATPO / MATPO-PR checkpoint.
```bash
# test on 2 x (8 x 80G-A800) nodes

export SERPER_API_KEY="YOUR_SERPER_API_KEY" && \
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY" && \
export WANDB_API_KEY="YOUR_WANDB_API_KEY" && \
export SINGLENODE=true && \
export RAY_DEBUG=legacy && \
export HYDRA_FULL_ERROR=1 && \
source YOUR_CONDA_PATH activate matpo && \
cd YOUR_PROJECT_PATH && \
bash examples/sglang_multiturn/launch.sh \
    examples/sglang_multiturn/eval_MATPO.sh

# # To evaluate a trained single-agent GRPO model checkpoint:
# bash examples/sglang_multiturn/launch.sh \
#     examples/sglang_multiturn/eval_single_agent.sh
```

## Experiments and Results

### Main Results

MATPO-PR consistently outperforms MATPO and single-agent GRPO baselines across all benchmarks:

| Method | GAIA-text | WebWalkerQA | FRAMES |
|--------|-----------|-------------|---------|
| Single-Agent GRPO | 33.89% | 28.97% | 57.34% |
| **MATPO** | **36.89%** | **30.29%** | **62.14%** |
| **MATPO-PR** | **40.90%** | **33.09%** | **64.20%** |

### Note on Result Variability Due to External Tools

During our experiments, we observed that agent performance (e.g., MATPO) may vary over time due to changes in the external tools and servers they access.
Upon investigation, we identified the primary cause as changes in the Serper API, a critical tool for Google search and web scraping. Specifically, we found that this API has recently undergone parameter changes, particularly in its Google search component.

This is why the performance metrics for MATPO and Single-Agent GRPO deviate slightly from the original [MATPO-PR](https://github.com/mzf666/MATPO) implementation.

### Training Configuration

- **Base Model**: Qwen3-14B-base
- **Training Dataset**: Filtered MuSiQue dataset
- **Training Steps**: 120 steps
- **Rollouts per Query**: 8 (for group normalization)
- **Reward Function**: 0.9 × accuracy + 0.1 × tool_format_reward

### Model Checkpoints and Rollouts

We release the trained Qwen3-14B-base model checkpoints at the 120th training step of both [single-agent GRPO](https://huggingface.co/veggiebird/MATPO-single-agent-14b) and [MATPO](https://huggingface.co/veggiebird/MATPO-14b) and [MATPO-PR](https://huggingface.co/veggiebird/MATPO-14b)

### Key Findings

- **More Stable Training**: MATPO-PR exhibits more stable learning curves and avoids catastrophic performance drops observed in MATPO and single-agent training

- **Better Credit Assignment**: Principled reward distribution across planner and worker rollouts leads to more effective learning than MATPO


### Practical Implementation Tips

Based on our experiments, we recommend:

- **Final Summary**: Final summaries from worker agents are critical for clean planner-worker interfaces
- **Query Recap**: Recapping original user query in worker prompt significantly improves performance
- **URL Blocking**: Remember to blocking HuggingFace search results to avoid data leakage


## Adding New Tools and Worker-Agents

To add new tools or worker-agents, you need to:

1. **Update the agent-tool configuration file**: modify the `examples/sglang_multiturn/config/tool_config/mcp_tool_config_full_agent.yaml` agent-tool configuration file following the instructions in the following subsections. 
2. **Update the system prompt of your training data**: update the system prompt of your training data accordingly, to ensure the planner-agent recieve the instructions to call new tools or new worker-agents correctly. A script to establish training data with updated system prompts based on a new agent-tool configuration file is provided in `examples/data_preprocess/update_system_prompt.py`.

In the following subsections, we present some examples on how to add new tools or worker-agents by modifying the agent-tool configuration file.

### Adding New MCP Tools to the Planner-Agent
You can add a new MCP tool to the planner-agent by adding a new entry to the `tools` list. Before adding the new tool, you need to wrap up your tool server into a standard MCP server. A tutorial on how to building new MCP tools is provided [here](https://miromindai.github.io/MiroFlow/contribute_tools/). 

In this example, we add a new MCP tool for calling a Python interpreter.

```yaml
tools:
  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: "serper-mcp"
      args: []
      env: ["SERPER_API_KEY", "HTTPS_PROXY"]
      server_name: "search_and_scrape_webpage"
    tool_schema:
      type: "mcp"
      function: {}

  # A newly added Python interpreter MCP tool.
  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: "python3"
      args: ["verl/tools/python_server.py"]
      env: ["E2B_API_KEY", "HTTPS_PROXY"]
      server_name: "python_interpreter"
    tool_schema:
      type: "mcp"
      function: {}

  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: ""
      server_name: "browsing_agent"
    tool_schema:
      type: "agent"
      function: {}
      schema:
        name: "search_and_browse"
        description: "This tool is an agent that performs the subtask of searching and browsing the web for specific missing information and generating the desired answer. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks. \nArgs: \n\tsubtask: the subtask to be performed. \nReturns: \n\tthe result of the subtask."
        parameters:
          properties:
            subtask:
              title: "Subtask"
              type: "string"
          required:
            - "subtask"
          title: "search_and_browseArguments"
          type: "object"
        server_name: "browsing_agent"

agents:
  - agent_type: "main_agent"
    tools:
      - browsing_agent
      - python_interpreter # Now, the planner-agent can directly call a Python interpreter tool, or it can delegate a searching-and-browsing subtask to the browsing-agent.
  - agent_type: "browsing_agent"
    tools:
      - search_and_scrape_webpage
```

### Adding New Worker-Agents

You can define and add a new worker-agent by adding a new entry to the `tools` and `agents` list. In this example, we add a new worker-agent for tackling programming subtasks using a Python interpreter. 
```yaml
tools:
  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: "serper-mcp"
      args: []
      env: ["SERPER_API_KEY", "HTTPS_PROXY"]
      server_name: "search_and_scrape_webpage"
    tool_schema:
      type: "mcp"
      function: {}

  # A newly added Python interpreter MCP tool.
  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: "python3"
      args: ["verl/tools/python_server.py"]
      env: ["E2B_API_KEY", "HTTPS_PROXY"]
      server_name: "python_interpreter"
    tool_schema:
      type: "mcp"
      function: {}

  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: ""
      server_name: "browsing_agent"
    tool_schema:
      type: "agent"
      function: {}
      schema:
        name: "search_and_browse"
        description: "This tool is an agent that performs the subtask of searching and browsing the web for specific missing information and generating the desired answer. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks. \nArgs: \n\tsubtask: the subtask to be performed. \nReturns: \n\tthe result of the subtask."
        parameters:
          properties:
            subtask:
              title: "Subtask"
              type: "string"
          required:
            - "subtask"
          title: "search_and_browseArguments"
          type: "object"
        server_name: "browsing_agent"

  # A newly added Python coding agent.
  - class_name: "verl.tools.mcp_tool.MCPTool"
    config:
      command: ""
      server_name: "python_agent"
    tool_schema:
      type: "agent"
      function: {}
      schema:
        name: "coding_in_python"
        description: "This tool is an agent that address subtasks that can be solved via Python programes. The subtask should be clearly defined, include relevant background, and focus on factual gaps. It does not perform vague or speculative subtasks. \nArgs: \n\tsubtask: the subtask to be performed. \nReturns: \n\tthe result of the subtask."
        parameters:
          properties:
            subtask:
              title: "Subtask"
              type: "string"
          required:
            - "subtask"
          title: "coding_in_pythonArguments"
          type: "object"
        server_name: "python_agent"

agents:
  - agent_type: "main_agent"
    tools:
      - browsing_agent
      - python_agent # The main-agent can delegate a coding subtask to the python-agent.

  - agent_type: "browsing_agent"
    tools:
      - search_and_scrape_webpage

  # A newly added Python coding agent.
  - agent_type: "python_agent"
    tools:
      - python_interpreter # The python-agent can directly call a Python interpreter tool to solve the coding subtask from the main-agent.
```

### Customizing Your Rollout Orchestration

In this repository, the multi-agent rollout orchestration is implemented the `_async_rollout_a_request()` function in `verl/workers/rollout/sglang_rollout/sglang_rollout.py`. The workflow is sketched as follows:

1. At each intermediate step in planner-agent rollout:
  - Once a MCP tool-call is parsed from the LLM response, the `_async_rollout_a_request()` function will be called to determine which agent to call, and return the MCP tool-call response. 
  - Once a worker-agent-call is parsed from the LLM response, the `_async_rollout_a_request()` function will create a new `AsyncRolloutRequest` object for this worker-agent-call, and pass it to `_async_rollout_a_request()` again to trigger a new agentic rollout. 
2. After the planner-agent rollout terminates (either no further tool-calls are parsed), all the worker-agent rollouts will be collected, serialized, and packed with the planner-agent rollout. 
3. The packed rollout will be passed to `verl/trainer/ppo/ray_trainer.py` to proceed MATPO training.

You can customize your own rollout orchestration by modifying the logic and scheduling in the `_async_rollout_a_request()` function.

## Citation

If you find MATPO-PR helpful in your research, please consider citing our paper:

```bibtex
@misc{xiaogang2026matpo,
  author       = {Zhanfeng Mo and Xiaogang Xu and Xingxuan Li and Lidong Bing},
  title        = {Multi-Agent Tool-Integrated Policy Optimization with Process Reward},
  year         = {2026},
  howpublished = {GitHub Repository},
  url          = {https://github.com/xiaogang00/MATPO-PR/blob/main/MATPO-PR-Report.pdf},
  note         = {Accessed: \today. Available at \url{https://github.com/xiaogang00/MATPO-PR}}
}

@misc{mo2025multi-agent-tool-integrated-policy-optimization,
      title={Multi-Agent Tool-Integrated Policy Optimization}, 
      author={Zhanfeng Mo and Xingxuan Li and Yuntao Chen and Lidong Bing},
      year={2025},
      eprint={2510.04678},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.04678}, 
}
```


## Acknowledgments

We would like to thank:

- [MATPO](https://github.com/mzf666/MATPO)
- **VolcEngine** for developing and open-sourcing [veRL](https://github.com/volcengine/verl), the RL training framework that powers MATPO
- **Alibaba Cloud** for the Qwen3 model series
- **Google** for the Serper API that enables web search capabilities
- The authors of **GAIA**, **WebWalkerQA**, **FRAMES**, and **MuSiQue** datasets
- The open-source community for valuable feedback and contributions


## FAQ

<details>
<summary><b>Q: What's the difference between MATPO and MATPO-PR</b></summary>

MATPO-PR enhances the MATPO by introducing step-wise process rewards. At each step, the main agent attempts to generate the target answer using the current available context to determine the reward. These process rewards are aggregated into a process return, which is then normalized across all rollouts to compute the GRPO advantage using global mean and variance.
</details>

<details>
<summary><b>Q: Can I use MATPO/MATPO-PR with models other than Qwen3?</b></summary>

Yes! MATPO-PR and MATPO is model-agnostic. You can use any decoder-only LLM that supports tool calling and multi-turn conversations. We've tested with Qwen3-14B-base, but models like Llama 3, Mistral, or other reasoning-capable LLMs should work.
</details>

<details>
<summary><b>Q: Can I apply MATPO-PR into other agentic frameworks?</b></summary>

Yes! Our process reward mechanism is designed for any multi-turn agentic system. It enables the computation of step-wise rewards at critical rounds of task execution. We are currently working on integrating this approach into existing ecosystems such as OpenClaw. We invite the community to follow our progress as we expand these capabilities.
</details>

<details>
<summary><b>Q: How many GPUs do I need for training?</b></summary>

For Qwen3-14B-base, we recommend:
- **Training**: 16 x (8 x A100/A800 GPUs (80GB))
- **Inference**: 2 x (8 x A100/A800 GPUs (80GB))

</details>

<details>
<summary><b>Q: How does MATPO and MATPO handle credit assignment?</b></summary>

MATPO extends GRPO with principled credit assignment:
1. The planner's final answer determines the accuracy reward
2. This reward is normalized across all rollouts in a group
3. Gradients flow proportionally to both planner and worker actions
4. Worker agents receive the same advantage value as their parent planner rollout

MATPO-PR enhances MATPO by incorporating a process-based reward mechanism:
1. Step-wise Evaluation: After each worker agent invocation, the main agent computes a process reward based on the current intermediate state
2. Process Isolation: The rollout for process rewards is decoupled from the primary MATPO process after acquisition to prevent interference with the core rollout
3. Reward Aggregation: Once the rollout is complete, all individual process rewards are aggregated to derive the final process return
4. Advantage Calculation: The advantage is computed by calculating the mean and variance across the final rewards and process returns of all rollouts


See our paper for more details.
</details>

<details>
<summary><b>Q: Can I use MATPO/MATPO-PR for tasks other than web search?</b></summary>

Absolutely! While our paper focuses on web search, the framework of MATPO/MATPO-PR is general. You can extend it to:
- Code generation with execution feedback
- Scientific reasoning with calculator tools
- Data analysis with pandas/SQL tools
- Any multi-turn task with verifiable rewards
</details>

<details>
<summary><b>Q: Do I need to block HuggingFace URLs during training?</b></summary>

For research integrity, yes - especially if your evaluation benchmarks are hosted on HuggingFace. This prevents models from "cheating" by finding ground-truth answers online. 

For production systems with no data leakage concerns, this is optional.
</details>
