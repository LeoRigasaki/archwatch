# Primary-source gold exact match

Frozen newest held-out labels (counting convention 1) against current records as published (convention 2: MTP modules reported separately). An adaptation/regression score, not held-out accuracy; labels unchanged.

Published fields: 134/139 exact matches across 7 labels.

| Field | Exact matches | Labeled | Not published |
|---|---:|---:|---:|
| activation | 7 | 7 | 0 |
| canvas_length | 1 | 1 | 0 |
| context_length | 7 | 7 | 0 |
| dense_ffn_intermediate | 2 | 2 | 0 |
| dense_layers | 5 | 5 | 0 |
| ffn_gated | 7 | 7 | 0 |
| ffn_intermediate | 2 | 2 | 0 |
| generation_type | 2 | 2 | 0 |
| global_head_dim | 1 | 1 | 0 |
| head_dim | 7 | 7 | 0 |
| hidden_size | 7 | 7 | 0 |
| hyper_connection_multiplier | 3 | 3 | 0 |
| input_layers | 1 | 1 | 0 |
| layer_type_counts | 7 | 7 | 0 |
| linear_conv_kernel | 1 | 1 | 0 |
| linear_key_head_dim | 1 | 1 | 0 |
| linear_num_key_heads | 1 | 1 | 0 |
| linear_num_value_heads | 1 | 1 | 0 |
| linear_value_head_dim | 1 | 1 | 0 |
| mla_kv_lora_rank | 2 | 2 | 0 |
| mla_qk_rope_head_dim | 2 | 2 | 0 |
| moe_active_experts | 5 | 5 | 0 |
| moe_expert_intermediate | 5 | 5 | 0 |
| moe_num_experts | 5 | 5 | 0 |
| moe_shared_expert_intermediate | 1 | 1 | 0 |
| moe_shared_experts | 5 | 5 | 0 |
| ngram_layer_ids | 1 | 1 | 0 |
| num_attention_heads | 7 | 7 | 0 |
| num_global_kv_heads | 1 | 1 | 0 |
| num_kv_heads | 7 | 7 | 0 |
| num_layers | 7 | 7 | 0 |
| num_mtp_layers | 3 | 3 | 0 |
| output_layers | 1 | 1 | 0 |
| params_total | 2 | 7 | 1 |
| qk_norm | 1 | 1 | 0 |
| recurrent_kv_cache_mode | 1 | 1 | 0 |
| recurrent_layers | 1 | 1 | 0 |
| recurrent_loops | 1 | 1 | 0 |
| shared_kv_layers | 1 | 1 | 0 |
| sliding_window | 2 | 2 | 0 |
| vision_encoder_layers | 1 | 1 | 0 |
| vision_hidden_size | 1 | 1 | 0 |
| vocab_size | 7 | 7 | 0 |

| Model | Exact matches | Labeled | Revision matches |
|---|---:|---:|---|
| Qwen/Qwen3.8-Flash-Next | 22 | 23 | True |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 18 | 19 | True |
| google/diffusiongemma-26B-A4B-it | 21 | 22 | True |
| microsoft/Mage-VL | 15 | 15 | True |
| mistralai/Mistral-Large-4-1T-A52B | 19 | 20 | True |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | 19 | 19 | True |
| zai-org/GLM-5.3-Flash-BF16 | 20 | 21 | True |

## Qwen/Qwen3.8-Flash-Next

- `params_total`: expected `179999981424`, got `177392830576` (different value).

## deepseek-ai/DeepSeek-V4.1-Flash-1017

- `params_total`: expected `763205315794`, got `748979953264` (different value).

## google/diffusiongemma-26B-A4B-it

- `params_total`: expected `25823781228`, got `None` (not published).

## mistralai/Mistral-Large-4-1T-A52B

- `params_total`: expected `1053161553184`, got `1051061584160` (different value).

## zai-org/GLM-5.3-Flash-BF16

- `params_total`: expected `321323031390`, got `313890438974` (different value).
