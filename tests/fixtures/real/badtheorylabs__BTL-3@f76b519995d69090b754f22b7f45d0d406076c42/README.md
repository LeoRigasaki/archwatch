---
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: text-generation
tags:
  - agent
  - coding
  - reasoning
  - tool-use
  - function-calling
  - qwen
---

<div align="center">

# BTL-3

### A 27B open-weight agent model for agentic coding and structural tool use

**95.1% HumanEval · 88.5% BFCL v4 AST · 88.1% LiveCodeBench v6 (193-case run)**

[Compact edition](https://huggingface.co/badtheorylabs/BTL-3-Compact) ·
[Runtime source](https://github.com/Badtheorylabs/BTL-3) ·
[Bad Theory Labs](https://www.badtheorylabs.com/) ·
[Discord](https://discord.gg/QJBCcB7bF)

</div>

## Introducing BTL-3

BTL-3 is a 27B open-weight agent model built for agentic coding, structural
tool use, repository work, failure recovery, and long multi-turn execution.

The release includes the full BTL-3 model and
[BTL-3 Compact](https://huggingface.co/badtheorylabs/BTL-3-Compact), which
packages the complete text model into one **8.39 GB** native file. That is
smaller than an 8B model stored in FP16 and corresponds to an effective
artifact footprint of **under 2.5 bits per parameter**. On a fresh private
100-turn tool-contract gate, BTL-3 Compact retained **83 of the 90 behaviors
the full model completed correctly: 92.2% measured conditional tool-behavior
retention**.

## Overview

BTL-3 is a post-trained Qwen3.6-27B model for coding agents, repository work,
structured tool use, and long multi-turn execution. It is tuned to reason,
act, inspect tool results, recover from failures, and stop when no action is
required.

This repository contains the frozen **RL-0013 rank-32 PEFT adapter** and its
tokenizer configuration. The base checkpoint is pinned to an exact revision
for reproducible loading.

## Highlights

- Strong structured tool use across single, multiple, and parallel calls.
- **91.2% BFCL irrelevance**, measuring when the model correctly avoids an
  unnecessary tool call.
- Thinking-mode coding with **95.12% HumanEval pass@1**.
- Qwen3.6 hybrid-attention architecture with a declared **262,144-token**
  context window.
- Open weights under Apache-2.0, deployable with Transformers or vLLM.
- An independent [8.39 GB Compact edition](https://huggingface.co/badtheorylabs/BTL-3-Compact)
  is available for native local inference.

## Results

All values below belong to the frozen BTL-3 RL-0013 release.

| Evaluation | Score | Protocol |
|---|---:|---|
| BFCL v4 AST | **88.5% (1097/1240)** | Complete official full set |
| HumanEval | **95.12% (156/164)** | pass@1, thinking mode |
| LiveCodeBench v6 | **88.1% (170/193)** | Completed 193-case run, thinking mode |
| BigCodeBench-Hard Instruct | **26.35% (39/148)** | Official strict pass@1 |
| BigCodeBench functional tests | **59.25% (506/854)** | Supplementary test-level score |

### BFCL v4 category breakdown

| Category | Score |
|---|---:|
| Simple | **93.2%** |
| Multiple | **95.5%** |
| Parallel | **87.0%** |
| Parallel-multiple | **70.0%** |
| Irrelevance | **91.2%** |

## Model specification

| Item | Specification |
|---|---|
| Base model | `Qwen/Qwen3.6-27B` |
| Base revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Release checkpoint | BTL-3 RL-0013 |
| Adapter | PEFT LoRA, rank 32, alpha 64 |
| Adapter size | 933,974,032 bytes |
| Architectural context | 262,144 tokens |
| Maximum RL sequence length | 65,536 tokens |
| Launch benchmark context | 32,768 tokens |
| Recommended mode | Thinking enabled for coding and reasoning |
| License | Apache-2.0 |

## Quickstart

### Transformers

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3.6-27B"
base_revision = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
adapter_id = "badtheorylabs/BTL-3"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision=base_revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)

messages = [
    {
        "role": "user",
        "content": "Inspect this repository, fix the failing tests, and explain the patch.",
    }
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=4096)
completion = output[0, inputs.input_ids.shape[1]:]
print(tokenizer.decode(completion, skip_special_tokens=False))
```

The supported release default is non-thinking mode. Thinking remains available
for controlled evaluation, but it is currently discouraged because the
RL-0013 reasoning policy can become repetitive or fail to terminate on some
prompts.

### vLLM

BTL-3 was evaluated with vLLM 0.23.0:

```bash
vllm serve Qwen/Qwen3.6-27B \
  --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --served-model-name BTL-3 \
  --enable-lora \
  --max-lora-rank 32 \
  --lora-modules BTL-3=/path/to/BTL-3 \
  --lora-target-modules \
    q_proj k_proj v_proj o_proj \
    in_proj_qkv in_proj_z in_proj_b in_proj_a out_proj \
    gate_proj up_proj down_proj \
  --reasoning-parser qwen3 \
  --language-model-only \
  --max-model-len 32768
```

For structured tools, enable the Qwen XML tool parser supported by your
installed vLLM version.

## Intended use

- coding, debugging, and test-driven repair;
- repository and terminal agents;
- structured function calling and multi-tool workflows;
- private or self-hosted agent deployments;
- long multi-turn tasks that require verification and recovery.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| `adapter_model.safetensors` | `37a8f519039707eba5906591cdb14268768db43f80489a9c2f83b3e51e5e89db` |

## Operational guidance

Run generated code and tool calls in a sandbox. Require explicit confirmation
before destructive, privileged, financial, or otherwise high-impact actions.

## License and citation

The adapter is released under Apache-2.0 and requires the separately
distributed Qwen3.6-27B base model.

```bibtex
@software{btl3_2026,
  title  = {BTL-3: A 27B Agentic Coding and Tool-Use Model},
  author = {Bad Theory Labs},
  year   = {2026},
  url    = {https://huggingface.co/badtheorylabs/BTL-3}
}
```

For questions and release updates, visit
[Bad Theory Labs](https://www.badtheorylabs.com/) or join the
[community Discord](https://discord.gg/QJBCcB7bF).
