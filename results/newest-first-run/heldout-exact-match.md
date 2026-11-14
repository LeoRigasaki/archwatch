# Primary-source gold exact match

Frozen FIRST RUN on seven previously unseen readable newest releases, before any family/tensor/extraction fixes. Two known development overlaps and one access-blocked publisher are excluded from accuracy and explicitly reported in the cohort manifest. Assistant-manual labels; human review pending.

Published fields: 97/139 exact matches across 7 labels.

| Field | Exact matches | Labeled | Not published |
|---|---:|---:|---:|
| activation | 7 | 7 | 0 |
| canvas_length | 0 | 1 | 1 |
| context_length | 7 | 7 | 0 |
| dense_ffn_intermediate | 2 | 2 | 0 |
| dense_layers | 4 | 5 | 1 |
| ffn_gated | 4 | 7 | 2 |
| ffn_intermediate | 1 | 2 | 1 |
| generation_type | 0 | 2 | 2 |
| global_head_dim | 0 | 1 | 1 |
| head_dim | 7 | 7 | 0 |
| hidden_size | 7 | 7 | 0 |
| hyper_connection_multiplier | 0 | 3 | 3 |
| input_layers | 0 | 1 | 1 |
| layer_type_counts | 3 | 7 | 0 |
| linear_conv_kernel | 0 | 1 | 1 |
| linear_key_head_dim | 0 | 1 | 1 |
| linear_num_key_heads | 0 | 1 | 1 |
| linear_num_value_heads | 0 | 1 | 1 |
| linear_value_head_dim | 0 | 1 | 1 |
| mla_kv_lora_rank | 2 | 2 | 0 |
| mla_qk_rope_head_dim | 2 | 2 | 0 |
| moe_active_experts | 4 | 5 | 1 |
| moe_expert_intermediate | 5 | 5 | 0 |
| moe_num_experts | 5 | 5 | 0 |
| moe_shared_expert_intermediate | 0 | 1 | 1 |
| moe_shared_experts | 3 | 5 | 2 |
| ngram_layer_ids | 0 | 1 | 1 |
| num_attention_heads | 7 | 7 | 0 |
| num_global_kv_heads | 0 | 1 | 1 |
| num_kv_heads | 7 | 7 | 0 |
| num_layers | 6 | 7 | 1 |
| num_mtp_layers | 0 | 3 | 3 |
| output_layers | 0 | 1 | 1 |
| params_total | 3 | 7 | 1 |
| qk_norm | 1 | 1 | 0 |
| recurrent_kv_cache_mode | 0 | 1 | 1 |
| recurrent_layers | 0 | 1 | 1 |
| recurrent_loops | 0 | 1 | 1 |
| shared_kv_layers | 0 | 1 | 1 |
| sliding_window | 2 | 2 | 0 |
| vision_encoder_layers | 1 | 1 | 0 |
| vision_hidden_size | 1 | 1 | 0 |
| vocab_size | 6 | 7 | 1 |

| Model | Exact matches | Labeled | Revision matches |
|---|---:|---:|---|
| Qwen/Qwen3.8-Flash-Next | 12 | 23 | True |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 13 | 19 | True |
| google/diffusiongemma-26B-A4B-it | 14 | 22 | True |
| microsoft/Mage-VL | 13 | 15 | True |
| mistralai/Mistral-Large-4-1T-A52B | 17 | 20 | True |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | 12 | 19 | True |
| zai-org/GLM-5.3-Flash-BF16 | 16 | 21 | True |

## Qwen/Qwen3.8-Flash-Next

- `params_total`: expected `179999981424`, got `179999981459` (different value).
- `moe_shared_experts`: expected `1`, got `None` (not published).
- `ffn_gated`: expected `True`, got `None` (not published).
- `layer_type_counts`: expected `{'gated_delta_net': 36, 'qwen_sparse_attention': 12}`, got `{'linear_attention': 36, 'full_attention': 12}` (different value).
- `linear_num_key_heads`: expected `16`, got `None` (not published).
- `linear_num_value_heads`: expected `48`, got `None` (not published).
- `linear_key_head_dim`: expected `128`, got `None` (not published).
- `linear_value_head_dim`: expected `128`, got `None` (not published).
- `linear_conv_kernel`: expected `4`, got `None` (not published).
- `hyper_connection_multiplier`: expected `4`, got `None` (not published).
- `num_mtp_layers`: expected `1`, got `None` (not published).

## deepseek-ai/DeepSeek-V4.1-Flash-1017

- `params_total`: expected `763205315794`, got `508182659298` (different value).
- `ffn_gated`: expected `True`, got `None` (not published).
- `layer_type_counts`: expected `{'compressed_attention': 40}`, got `{'full_attention': 40}` (different value).
- `hyper_connection_multiplier`: expected `4`, got `None` (not published).
- `ngram_layer_ids`: expected `[1, 14]`, got `None` (not published).
- `num_mtp_layers`: expected `3`, got `None` (not published).

## google/diffusiongemma-26B-A4B-it

- `global_head_dim`: expected `512`, got `None` (not published).
- `num_global_kv_heads`: expected `2`, got `None` (not published).
- `params_total`: expected `25823781228`, got `None` (not published).
- `moe_active_experts`: expected `8`, got `None` (not published).
- `moe_shared_experts`: expected `1`, got `None` (not published).
- `moe_shared_expert_intermediate`: expected `2112`, got `None` (not published).
- `generation_type`: expected `block_diffusion`, got `None` (not published).
- `canvas_length`: expected `256`, got `None` (not published).

## microsoft/Mage-VL

- `vocab_size`: expected `151936`, got `None` (not published).
- `ffn_intermediate`: expected `9728`, got `None` (not published).

## mistralai/Mistral-Large-4-1T-A52B

- `params_total`: expected `1053161553184`, got `1053225397792` (different value).
- `layer_type_counts`: expected `{'indexed_mla': 72}`, got `{'indexed_attention': 72}` (different value).
- `shared_kv_layers`: expected `9`, got `None` (not published).

## nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16

- `ffn_gated`: expected `True`, got `False` (different value).
- `generation_type`: expected `latent_recurrent`, got `None` (not published).
- `recurrent_loops`: expected `2`, got `None` (not published).
- `recurrent_layers`: expected `40`, got `None` (not published).
- `input_layers`: expected `16`, got `None` (not published).
- `output_layers`: expected `16`, got `None` (not published).
- `recurrent_kv_cache_mode`: expected `per_loop`, got `None` (not published).

## zai-org/GLM-5.3-Flash-BF16

- `num_layers`: expected `45`, got `None` (not published).
- `dense_layers`: expected `3`, got `None` (not published).
- `layer_type_counts`: expected `{'kda_linear': 34, 'indexed_mla': 11}`, got `{'linear_attention': 34, 'deepseek_sparse_attention': 11}` (different value).
- `hyper_connection_multiplier`: expected `4`, got `None` (not published).
- `num_mtp_layers`: expected `1`, got `None` (not published).
