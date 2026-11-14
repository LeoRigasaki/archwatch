# Primary-source gold exact match

Ten-model development gold set, manually inspected primary sources; not human-reviewed or a held-out accuracy estimate.

Published fields: 117/144 exact matches across 10 labels.

| Field | Exact matches | Labeled | Not published |
|---|---:|---:|---:|
| activation | 8 | 10 | 2 |
| block_layout | 0 | 2 | 2 |
| context_length | 9 | 10 | 1 |
| dense_ffn_intermediate | 2 | 2 | 0 |
| dense_layers | 2 | 2 | 0 |
| ffn_gated | 0 | 10 | 10 |
| ffn_intermediate | 7 | 8 | 1 |
| head_dim | 9 | 10 | 1 |
| hidden_size | 9 | 10 | 1 |
| layer_type_counts | 8 | 10 | 1 |
| mla_kv_lora_rank | 2 | 2 | 0 |
| mla_qk_rope_head_dim | 2 | 2 | 0 |
| moe_active_experts | 2 | 2 | 0 |
| moe_expert_intermediate | 2 | 2 | 0 |
| moe_latent_dim | 1 | 1 | 0 |
| moe_num_experts | 2 | 2 | 0 |
| moe_shared_experts | 2 | 2 | 0 |
| num_attention_heads | 9 | 10 | 1 |
| num_kv_heads | 9 | 10 | 1 |
| num_layers | 9 | 10 | 1 |
| params_total | 9 | 10 | 1 |
| sandwich_norm | 1 | 3 | 2 |
| short_conv_kernel | 1 | 1 | 0 |
| sliding_window | 3 | 3 | 0 |
| vocab_size | 9 | 10 | 1 |

| Model | Exact matches | Labeled | Revision matches |
|---|---:|---:|---|
| LiquidAI/LFM3-1.7B-Base | 13 | 15 | True |
| Qwen/Qwen3-0.6B-Base | 11 | 12 | True |
| deepseek-ai/DeepSeek-V2-Lite | 18 | 19 | True |
| google/gemma-3-1b-pt | 12 | 14 | True |
| meta-llama/Llama-3.1-8B | 11 | 12 | True |
| microsoft/phi-2 | 11 | 13 | True |
| mistralai/Mistral-7B-v0.1 | 11 | 13 | True |
| moonshotai/Kimi-K3.1-Pro | 19 | 20 | True |
| nvidia/Nemotron-H-8B-Base-8K | 0 | 13 | False |
| zai-org/GLM-4-9B-0414 | 11 | 13 | True |

## LiquidAI/LFM3-1.7B-Base

- `ffn_gated`: expected `True`, got `None` (not published).
- `activation`: expected `silu`, got `None` (not published).

## Qwen/Qwen3-0.6B-Base

- `ffn_gated`: expected `True`, got `None` (not published).

## deepseek-ai/DeepSeek-V2-Lite

- `ffn_gated`: expected `True`, got `None` (not published).

## google/gemma-3-1b-pt

- `ffn_gated`: expected `True`, got `None` (not published).
- `sandwich_norm`: expected `True`, got `None` (not published).

## meta-llama/Llama-3.1-8B

- `ffn_gated`: expected `True`, got `None` (not published).

## microsoft/phi-2

- `ffn_gated`: expected `False`, got `None` (not published).
- `block_layout`: expected `parallel`, got `None` (not published).

## mistralai/Mistral-7B-v0.1

- `layer_type_counts`: expected `{'sliding_attention': 32}`, got `{'full_attention': 32}` (different value).
- `ffn_gated`: expected `True`, got `None` (not published).

## moonshotai/Kimi-K3.1-Pro

- `ffn_gated`: expected `True`, got `None` (not published).

## nvidia/Nemotron-H-8B-Base-8K

- `num_layers`: expected `52`, got `None` (source revision mismatch/missing record).
- `hidden_size`: expected `4096`, got `None` (source revision mismatch/missing record).
- `vocab_size`: expected `131072`, got `None` (source revision mismatch/missing record).
- `num_attention_heads`: expected `32`, got `None` (source revision mismatch/missing record).
- `num_kv_heads`: expected `8`, got `None` (source revision mismatch/missing record).
- `head_dim`: expected `128`, got `None` (source revision mismatch/missing record).
- `ffn_intermediate`: expected `21504`, got `None` (source revision mismatch/missing record).
- `context_length`: expected `8192`, got `None` (source revision mismatch/missing record).
- `params_total`: expected `8100852736`, got `None` (source revision mismatch/missing record).
- `layer_type_counts`: expected `{'mamba2': 24, 'mlp': 24, 'full_attention': 4}`, got `None` (source revision mismatch/missing record).
- `ffn_gated`: expected `False`, got `None` (source revision mismatch/missing record).
- `activation`: expected `relu2`, got `None` (source revision mismatch/missing record).
- `block_layout`: expected `standalone`, got `None` (source revision mismatch/missing record).

## zai-org/GLM-4-9B-0414

- `ffn_gated`: expected `True`, got `None` (not published).
- `sandwich_norm`: expected `True`, got `None` (not published).
