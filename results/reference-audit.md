# Reference audit (internal; not part of the site)

Answer sheet: `rasbt/llm-architecture-gallery` `models.yml`, last changed at commit `c5624cfa9285550769e30cac118ebfc12ecfe5f0` (repository head `660cc1886a9d2a009c46ecc8e17e98b1075f58b2`). Its values are never a source for ArchWatch records.

Entries: 120; mapped to a Hub repo: 118; analyzed by our deterministic pipeline: 106; compared values: 531; agreeing: 419; differences: 112.

| Group | Differences |
|---|---:|
| ours_wrong | 33 |
| his_wrong | 6 |
| name_rounding | 11 |
| definition | 62 |
| unresolved | 0 |

Groups: **ours_wrong** = our number under the fixed rule is outside his accepted range and nothing else explains it (open); **his_wrong** = the model's own files contradict his value; **name_rounding** = his value is the size in the model name or a figure the card gives as approximate ('~'), ours is the exact count; **definition** = our number under the fixed rule (or a rule the card states) lands in his accepted range; **unresolved** = review queue.

Fixed rules (the same for every model without a card statement): `params_total` = all stored tensors minus vision tower and projector (text-only figure); `params_active` = per-token count (unselected experts, MTP heads and lookup tables not active); `kv_cache_per_token_bf16` = full retention, every attention layer appends its KV per token (sliding-window layers not capped). An exclusion (embedding, output head, MTP, lookup tables) or a vision inclusion is applied only when the card sentence that states his figure says so; that sentence is quoted below. Accepted range: 2% or the card's own rounding.

## ArchWatch errors found and fixed during this audit

| Error | Regression test |
|---|---|
| MXFP4 expert blocks counted at one value per byte (gpt-oss) | `test_mxfp4_blocks_hold_two_values_per_byte_and_scales_are_metadata` |
| FP4 expert flag on config.json ignored (DeepSeek-V4) | `test_native_fp4_experts_flagged_on_the_model_config` |
| Mistral consolidated.safetensors counted on top of the HF shards (Mistral Small 3.1/4) | `test_consolidated_export_is_not_counted_twice_but_extra_components_are_kept` |
| Tensor-parallel shards merged by name, keeping one rank (grok-2) | `test_tensor_parallel_slices_count_every_rank` |
| MTP layers stored after the decoder stack counted as main/active weights (DeepSeek-V3/R1, GLM, Qwen3.8-2.4T) | `test_mtp_layers_after_the_decoder_stack_are_auxiliary_not_active` |
| Multimodal projectors counted as text weights (Mistral Small 4, Llama 4, Mistral Large 4) | `test_multimodal_projectors_are_not_text_weights` |
| Native output heads (head.weight, output.weight) not recognized, so tied embeddings were misread | `test_native_output_heads_are_recognized` |
| moe_num_experts / moe_top_k not read, so MoE models looked dense (Step-3.5, Step-5) | `test_moe_keys_with_moe_prefix_are_read` |
| Inkling's native embedding names (model.llm.embed / unembed) not recognized as embedding and output head | `test_inkling_embed_and_unembed_are_embedding_and_head` |
| Comparison parser: 'Mamba-2' read as 2 layers, MTP counted as a layer, 128,000 vs 128K not normalised | `test_layer_mix_counts_leading_terms_and_skips_mtp` |

## Differences

| Model | Field | Ours (published) | His | Gap | Ours recomputed | Rule | Group | Evidence |
|---|---|---:|---|---:|---:|---|---|---|
| Aleph-Alpha/Kolibri-1 | params_active | 3,785,438,560 | 78B total, 3.46B active (4.4% active) | 9.4% | 3,785,438,560 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 3,785,438,560; his accepted range 3,390,800,000-3,529,200,000. |
| MiniMaxAI/MiniMax-M2 | params_active | 11,031,148,288 | 230B total, 10B active (4.3% active) | 10.3% | 11,031,148,288 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 11,031,148,288; his accepted range 9,500,000,000-10,500,000,000. |
| MiniMaxAI/MiniMax-M2.5 | params_active | 11,031,148,288 | 230B total, 10B active (4.3% active) | 10.3% | 11,031,148,288 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 11,031,148,288; his accepted range 9,500,000,000-10,500,000,000. |
| MiniMaxAI/MiniMax-M2.7 | params_active | 11,031,148,288 | 230B total, 10B active (4.3% active) | 10.3% | 11,031,148,288 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 11,031,148,288; his accepted range 9,500,000,000-10,500,000,000. |
| PrimeIntellect/INTELLECT-3 | params_active | 13,424,129,664 | 106B total, 12B active (11.3% active) | 11.9% | 13,424,129,664 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 13,424,129,664; his accepted range 11,500,000,000-12,500,000,000. |
| Qwen/Qwen3.8-Flash-Next | params_active | 7,306,994,560 | 125B core, 6B active (4.8% active); plus 51B n-gram embeddings and 4B MTP | 21.8% | 7,306,994,560 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 7,306,994,560; his accepted range 5,500,000,000-6,500,000,000. |
| XiaomiMiMo/MiMo-V2.5 | params_active | 15,707,829,184 | 310B total, 15B active (4.8% active) | 4.7% | 15,707,829,184 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 15,707,829,184; his accepted range 14,500,000,000-15,500,000,000. |
| XiaomiMiMo/MiMo-V2.6-Flash-RL | params_active | 15,707,251,648 | 309B total, 15B active (4.9% active) | 4.7% | 15,707,251,648 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 15,707,251,648; his accepted range 14,500,000,000-15,500,000,000. |
| Zyphra/ZAYA1-8B | params_active | 1,290,742,264 | 8.4B total, 760M active/token (9% active) | 69.8% | 1,290,742,264 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 1,290,742,264; his accepted range 744,800,000-775,200,000. |
| deepseek-ai/DeepSeek-V3.2 | params_active | 38,406,046,640 | 671B total, 37B active (5.5% active) | 3.8% | 38,406,046,640 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 38,406,046,640; his accepted range 36,260,000,000-37,740,000,000. |
| deepseek-ai/DeepSeek-V4-Flash | params_active | 13,799,632,471 | 284B total, 13B active (4.6% active) | 6.2% | 13,799,632,471 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 13,799,632,471; his accepted range 12,500,000,000-13,500,000,000. |
| deepseek-ai/DeepSeek-V4.1-Flash | params_active | 16,865,928,304 | 552B backbone + 196B Engram; 8B active prefill, 16B active decode (2.9% of backbone) | 110.8% | 16,865,928,304 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 16,865,928,304; his accepted range 7,840,000,000-16,320,000,000. |
| inclusionAI/Ling-3.0-flash | params_active | 5,505,693,152 | 124B total, 5.1B active (4.1% active) | 8.0% | 5,505,693,152 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 5,505,693,152; his accepted range 4,998,000,000-5,202,000,000. |
| openai/gpt-oss-120b | params_active | 5,711,982,912 | 117B total, 5.1B active (4.4% active) | 12.0% | 5,711,982,912 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 5,711,982,912; his accepted range 4,998,000,000-5,202,000,000. |
| openai/gpt-oss-20b | params_active | 4,187,440,704 | 21B total, 3.6B active (17.1% active) | 16.3% | 4,187,440,704 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 4,187,440,704; his accepted range 3,528,000,000-3,672,000,000. |
| sarvamai/sarvam-105b | params_active | 12,414,902,144 | 105B total, 10.3B active (9.8% active) | 20.5% | 12,414,902,144 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 12,414,902,144; his accepted range 10,094,000,000-10,506,000,000. |
| zai-org/GLM-4.5 | params_active | 33,632,266,144 | 355B total, 32B active (9% active) | 5.1% | 33,632,266,144 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 33,632,266,144; his accepted range 31,360,000,000-32,640,000,000. |
| zai-org/GLM-4.5-Air | params_active | 13,424,129,664 | 106B total, 12B active (11.3% active) | 11.9% | 13,424,129,664 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 13,424,129,664; his accepted range 11,500,000,000-12,500,000,000. |
| zai-org/GLM-4.7 | params_active | 33,632,266,144 | 355B total, 32B active (9% active) | 5.1% | 33,632,266,144 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 33,632,266,144; his accepted range 31,360,000,000-32,640,000,000. |
| zai-org/GLM-5 | params_active | 41,784,728,832 | 744B total, 40B active (5.4% active) | 4.5% | 41,784,728,832 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 41,784,728,832; his accepted range 39,200,000,000-40,800,000,000. |
| zai-org/GLM-5.1 | params_active | 41,784,728,832 | 744B total, 40B active (5.4% active) | 4.5% | 41,784,728,832 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 41,784,728,832; his accepted range 39,200,000,000-40,800,000,000. |
| zai-org/GLM-5.2 | params_active | 41,250,530,304 | 744B total, 40B active (5.4% active) | 3.1% | 41,250,530,304 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 41,250,530,304; his accepted range 39,200,000,000-40,800,000,000. |
| zai-org/GLM-5.3-Flash | params_active | 17,377,397,566 | 320B total, 18B active (5.6% active) | 3.5% | 17,377,397,566 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | ours_wrong | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 17,377,397,566; his accepted range 17,500,000,000-18,500,000,000. |
| Zyphra/ZAYA1-8B | params_total | 8,840,489,464 | 8.4B total, 760M active/token (9% active) | 5.2% | 8,840,489,464 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 8,840,489,464; his accepted range 8,232,000,000-8,568,000,000. |
| baidu/ERNIE-5.5-Flash-Base | params_total | 286,262,969,530 | 280B total, 14B active (5.0% active) | 2.2% | 286,262,969,530 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 286,262,969,530; his accepted range 274,400,000,000-285,600,000,000. |
| deepseek-ai/DeepSeek-R1 | params_total | 684,531,386,000 | 671B total, 37B active (5.5% active) | 2.0% | 684,531,386,000 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 684,531,386,000; his accepted range 657,580,000,000-684,420,000,000. |
| deepseek-ai/DeepSeek-V3.2 | params_total | 685,396,921,376 | 671B total, 37B active (5.5% active) | 2.1% | 685,396,921,376 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 685,396,921,376; his accepted range 657,580,000,000-684,420,000,000. |
| deepseek-ai/DeepSeek-V4-Flash | params_total | 290,942,289,362 | 284B total, 13B active (4.6% active) | 2.4% | 290,942,289,362 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 290,942,289,362; his accepted range 278,320,000,000-289,680,000,000. |
| inclusionAI/Ling-3.0-flash | params_total | 127,486,405,600 | 124B total, 5.1B active (4.1% active) | 2.8% | 127,486,405,600 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 127,486,405,600; his accepted range 121,520,000,000-126,480,000,000. |
| inclusionAI/Ling-3.1-flash | params_total | 572,155,509,344 | 560B total, 25B active (4.5% active) | 2.2% | 572,155,509,344 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 572,155,509,344; his accepted range 548,800,000,000-571,200,000,000. |
| microsoft/phi-4 | params_total | 14,659,507,200 | 14B parameters | 4.7% | 14,659,507,200 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 14,659,507,200; his accepted range 13,500,000,000-14,500,000,000. |
| thinkingmachines/Inkling | params_total | 952,377,623,626 | 975B total, 41B active (4.2% active) | 2.3% | 952,293,880,394 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 952,293,880,394; his accepted range 955,500,000,000-994,500,000,000. |
| zai-org/GLM-4.5-Air | params_total | 110,468,824,832 | 106B total, 12B active (11.3% active) | 4.2% | 110,468,824,832 | default: all stored tensors minus vision tower and projector (text-only figure) | ours_wrong | Closest recomputation: default: all stored tensors minus vision tower and projector (text-only figure) = 110,468,824,832; his accepted range 103,880,000,000-108,120,000,000. |
| MiniMaxAI/MiniMax-M2.7 | context_length | 204,800 | 196,608 | 4.2% | - | primary source contradicts | his_wrong | MiniMaxAI/MiniMax-M2.7 config.json at 0a6536713b: max_position_embeddings = 204800; the README states no context length. 196,608 appears in neither file. |
| deepseek-ai/DeepSeek-V3.2 | context_length | 163,840 | 128,000 | 28.0% | - | primary source contradicts | his_wrong | deepseek-ai/DeepSeek-V3.2 config.json at a7e62ac04e: max_position_embeddings = 163840 (rope_scaling yarn, factor 40 over original_max_position_embeddings 4096); the README states no context length. 128,000 / 128K appears in neither file. |
| google/gemma-3-270m | context_length | 32,768 | 128,000 | 74.4% | - | primary source contradicts | his_wrong | google/gemma-3-270m README.md at 9b0cfec892: "Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B and 270M sizes"; config.json max_position_embeddings = 32768. 128K is the family figure for the larger sizes. |
| google/gemma-4-12B-it | context_length | 262,144 | 128,000 | 104.8% | - | primary source contradicts | his_wrong | google/gemma-4-12B-it README.md at 43c9d014b2: "Context windows of up to 128K tokens (E2B/E4B) and 256K tokens (12B, 26B A4B/31B)" and its table lists Context Length 256K tokens for 12B; config.json text_config.max_position_embeddings = 262144. |
| poolside/Laguna-XS.2 | context_length | 262,144 | 131,072 | 100.0% | - | primary source contradicts | his_wrong | poolside/Laguna-XS.2 README.md at a397bde045: "Context window: 262,144 tokens"; config.json max_position_embeddings = 262144. 131,072 appears in neither. |
| Nanbeige/Nanbeige4.2-3B | kv_cache_per_token_bf16 | 90,112 | 176 KiB | 50.0% | - | primary source contradicts | his_wrong | Nanbeige/Nanbeige4.2-3B config.json at b82e54bd60: num_hidden_layers = 22, num_key_value_heads = 8, head_dim = 128, no sliding_window. His own formula (layers x KV heads x head_dim x 4) gives 22 x 8 x 128 x 4 = 90,112 B = 88 KiB; his 176 KiB is twice that. |
| JetBrains/Mellum2-12B-A2.5B-Thinking | params_active | 2,439,060,736 | 12B total, 2.5B active (20.8% active) | 2.4% | 2,439,060,736 | nominal size: model name | name_rounding | Nominal figure: his 2.5B is the size in the model name (Mellum2-12B-A2.5B-Thinking). Ours is the exact count from the weights (default: per-token count (unselected experts, MTP heads and lookup tables not active) = 2,439,060,736). |
| MiniMaxAI/MiniMax-M3 | params_active | 25,962,473,856 | 428B total, 23B active (5.4% active) | 12.9% | 25,962,473,856 | nominal size: card '~' figure | name_rounding | Nominal figure: the card gives it as approximate: "It has ~428B parameters and ~23B activated parameters.". Ours is the exact count from the weights (default: per-token count (unselected experts, MTP heads and lookup tables not active) = 25,962,473,856). |
| MiniMaxAI/MiniMax-M3.1-Flash | params_active | 26,304,278,316 | 428B total, 23B active (5.4% active) | 14.4% | 26,304,278,316 | nominal size: card '~' figure | name_rounding | Nominal figure: the card gives it as approximate: "- **Parameters:** Approximately 428B total; 23B activated per token". Ours is the exact count from the weights (default: per-token count (unselected experts, MTP heads and lookup tables not active) = 26,304,278,316). |
| Qwen/Qwen3-Next-80B-A3B-Instruct | params_active | 3,874,929,408 | 80B total, 3B active (3.8% active) | 29.2% | 3,874,929,408 | nominal size: model name | name_rounding | Nominal figure: his 3B is the size in the model name (Qwen3-Next-80B-A3B-Instruct). Ours is the exact count from the weights (default: per-token count (unselected experts, MTP heads and lookup tables not active) = 3,874,929,408). |
| meta-models/Muse-Spark-Open-816B-A42B | params_active | 46,208,726,784 | 816B total, 42B active (5.1% active) | 10.0% | 46,208,726,784 | nominal size: model name | name_rounding | Nominal figure: his 42B is the size in the model name (Muse-Spark-Open-816B-A42B). Ours is the exact count from the weights (default: per-token count (unselected experts, MTP heads and lookup tables not active) = 46,208,726,784). |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | params_active | 3,580,076,364 | 30B total, 3B active (10% active) | 19.3% | 3,580,076,364 | nominal size: model name | name_rounding | Nominal figure: his 3B is the size in the model name (NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4). Ours is the exact count from the weights (default: per-token count (unselected experts, MTP heads and lookup tables not active) = 3,580,076,364). |
| Nanbeige/Nanbeige4.1-3B | params_total | 3,933,637,120 | 3B parameters | 31.1% | 3,933,637,120 | nominal size: model name | name_rounding | Nominal figure: his 3B is the size in the model name (Nanbeige4.1-3B). Ours is the exact count from the weights (default: all stored tensors minus vision tower and projector (text-only figure) = 3,933,637,120). |
| Qwen/Qwen3-32B | params_total | 32,762,123,264 | 32B parameters | 2.4% | 32,762,123,264 | nominal size: model name | name_rounding | Nominal figure: his 32B is the size in the model name (Qwen3-32B). Ours is the exact count from the weights (default: all stored tensors minus vision tower and projector (text-only figure) = 32,762,123,264). |
| ibm-granite/granite-4.1-30b | params_total | 28,865,728,512 | 30B parameters | 3.8% | 28,865,728,512 | nominal size: model name | name_rounding | Nominal figure: his 30B is the size in the model name (granite-4.1-30b). Ours is the exact count from the weights (default: all stored tensors minus vision tower and projector (text-only figure) = 28,865,728,512). |
| moonshotai/Kimi-Linear-48B-A3B-Base | params_total | 49,122,681,728 | 48B total, 3B active (6.3% active) | 2.3% | 49,122,681,728 | nominal size: model name | name_rounding | Nominal figure: his 48B is the size in the model name (Kimi-Linear-48B-A3B-Base). Ours is the exact count from the weights (default: all stored tensors minus vision tower and projector (text-only figure) = 49,122,681,728). |
| sarvamai/sarvam-30b | params_total | 32,152,650,368 | 30B total, 2.4B active (8% active) | 7.2% | 32,152,650,368 | nominal size: model name | name_rounding | Nominal figure: his 30B is the size in the model name (sarvam-30b). Ours is the exact count from the weights (default: all stored tensors minus vision tower and projector (text-only figure) = 32,152,650,368). |
| HuggingFaceTB/SmolLM3-3B-Base | context_length | 65,536 | 131,072 | 50.0% | 131,072 | advertised context the model card states ("128k") | definition | Rule: advertised context the model card states ("128k"): "xture and training configs - **Long context:** Trained on 64k context and suppots up to **128k tokens** using YARN extrapolation - **Multilingual**: 6 natively supported (English, Fren". Ours recomputed: 131,072. His: 131,072 (accepted 128,451-133,693); gap after the rule 0.00%. |
| arcee-ai/Trinity-Large-Base | context_length | 262,144 | 512,000 | 48.8% | 512,000 | advertised context the model card states ("512k") | definition | Rule: advertised context the model card states ("512k"): "se layers / 6 / / Pretraining context length / 8,192 / / Context length after extention / 512k / / Architecture / Sparse MoE / ## Benchmark Results / Benchmark / N-sh". Ours recomputed: 512,000. His: 512,000 (accepted 501,760-522,240); gap after the rule 0.00%. |
| deepseek-ai/DeepSeek-R1 | context_length | 163,840 | 128,000 | 28.0% | 128,000 | advertised context the model card states ("128K") | definition | Rule: advertised context the model card states ("128K"): "-: / :------------: / :------------: / :------------: / / DeepSeek-R1-Zero / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-R1 / 671B / 37B / 128K / [🤗 HuggingFace] / <". Ours recomputed: 128,000. His: 128,000 (accepted 125,440-130,560); gap after the rule 0.00%. |
| deepseek-ai/DeepSeek-V3 | context_length | 163,840 | 128,000 | 28.0% | 128,000 | advertised context the model card states ("128K") | definition | Rule: advertised context the model card states ("128K"): "-: / :------------: / :------------: / :------------: / / DeepSeek-V3-Base / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-V3 / 671B / 37B / 128K / [🤗 HuggingFace] / <". Ours recomputed: 128,000. His: 128,000 (accepted 125,440-130,560); gap after the rule 0.00%. |
| inclusionAI/Ling-2.5-1T | context_length | 131,072 | 256,000 | 48.8% | 256,000 | advertised context the model card states ("256K") | definition | Rule: advertised context the model card states ("256K"): "----------------------------------------------------------------------: / / Ling-2.5-1T / 256K -> 1M / [🤗 HuggingFace] &nbsp;&nbsp; [🤖 ModelScope] / </center> Note: If you are int". Ours recomputed: 256,000. His: 256,000 (accepted 250,880-261,120); gap after the rule 0.00%. |
| inclusionAI/Ling-3.1-flash | context_length | 262,144 | 1M reported; 262,144 in config | 73.8% | 1,000,000 | advertised context the model card states ("1M") | definition | Rule: advertised context the model card states ("1M"): "world tasks. With **560B** total parameters, **25B** active parameters per token, and a **1M-token context window**, Ling-3.1-flash builds upon Ling-3.0-flash with improvements in ar". Ours recomputed: 1,000,000. His: 1M reported; 262,144 in config (accepted 500,000-1,500,000); gap after the rule 0.00%. |
| mistralai/Mistral-Small-4-119B-2603 | context_length | 1,048,576 | 256,000 | 309.6% | 256,000 | advertised context the model card states ("256k") | definition | Rule: advertised context the model card states ("256k"): "E**: 128 experts, 4 active. - **119B parameters**, with **6.5B activated per token**. - **256k context length**. - **Multimodal input**: Accepts both text and image input, with text ou". Ours recomputed: 256,000. His: 256,000 (accepted 250,880-261,120); gap after the rule 0.00%. |
| CohereLabs/tiny-aya-base | kv_cache_per_token_bf16 | 18,432 | 72 KiB | 75.0% | 73,728 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 73,728. His: 72 KiB (accepted 72,253-75,203); gap after the rule 0.00%. |
| JetBrains/Mellum2-12B-A2.5B-Thinking | kv_cache_per_token_bf16 | 14,336 | 56 KiB | 75.0% | 57,344 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 57,344. His: 56 KiB (accepted 56,197-58,491); gap after the rule 0.00%. |
| LiquidAI/LFM3-1.7B | kv_cache_per_token_bf16 | 12,288 | 22 KiB | 45.5% | 22,528 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 22,528. His: 22 KiB (accepted 22,016-23,040); gap after the rule 0.00%. |
| RecursiveSI/flint | kv_cache_per_token_bf16 | 28,672 | 98 KiB | 71.4% | 100,352 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 100,352. His: 98 KiB (accepted 98,345-102,359); gap after the rule 0.00%. |
| allenai/Olmo-3-1025-7B | kv_cache_per_token_bf16 | 131,072 | 512 KiB | 75.0% | 524,288 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 524,288. His: 512 KiB (accepted 513,802-534,774); gap after the rule 0.00%. |
| allenai/Olmo-3-32B-Think | kv_cache_per_token_bf16 | 65,536 | 256 KiB | 75.0% | 262,144 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 262,144. His: 256 KiB (accepted 256,901-267,387); gap after the rule 0.00%. |
| google/gemma-3-270m | kv_cache_per_token_bf16 | 3,072 | 18 KiB | 83.3% | 18,432 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 18,432. His: 18 KiB (accepted 17,920-18,944); gap after the rule 0.00%. |
| meta-models/Muse-Glimmer-30B | kv_cache_per_token_bf16 | 13,312 | 52 KiB | 75.0% | 53,248 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 53,248. His: 52 KiB (accepted 52,183-54,313); gap after the rule 0.00%. |
| poolside/Laguna-S-2.1 | kv_cache_per_token_bf16 | 49,152 | 192 KiB | 75.0% | 196,608 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 196,608. His: 192 KiB (accepted 192,676-200,540); gap after the rule 0.00%. |
| poolside/Laguna-XS-2.1 | kv_cache_per_token_bf16 | 40,960 | 160 KiB | 75.0% | 163,840 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 163,840. His: 160 KiB (accepted 160,563-167,117); gap after the rule 0.00%. |
| poolside/Laguna-XS.2 | kv_cache_per_token_bf16 | 40,960 | 160 KiB | 75.0% | 163,840 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 163,840. His: 160 KiB (accepted 160,563-167,117); gap after the rule 0.00%. |
| reflection/Beam-1.0-501B-A23B | kv_cache_per_token_bf16 | 36,864 | 208 KiB | 82.7% | 212,992 | default: full retention, every attention layer appends its KV per token (sliding-window layers not capped) | definition | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 212,992. His: 208 KiB (accepted 208,732-217,252); gap after the rule 0.00%. |
| Zyphra/ZAYA1-8B | num_layers | 40 | 40 CCA/GQA attention + 40 top-1 MoE FFN entries | 50.0% | 80 | attention and FFN entries counted separately: 2 x config num_hidden_layers (40) | definition | Rule: attention and FFN entries counted separately: 2 x config num_hidden_layers (40). Ours recomputed: 80. His: 40 CCA/GQA attention + 40 top-1 MoE FFN entries (accepted 78-82). Ours: config.json num_hidden_layers = 40, layer_types = 40 x 'hybrid' (attention and MoE in each layer) at 67d34da515. His: attention entries and FFN entries counted separately (40 + 40). |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | num_layers | 72 | 8 input + 20 shared × 2 passes + 8 output (56 applications) | 100.0% | 36 | attention+MLP blocks: config num_hidden_layers (72 operator entries) / 2 | definition | Rule: attention+MLP blocks: config num_hidden_layers (72 operator entries) / 2. Ours recomputed: 36. His: 8 input + 20 shared × 2 passes + 8 output (56 applications) (accepted 35-37). Ours: config.json num_hidden_layers = 72, layer_types alternate attention and MLP (n_input_layers 16 + n_recurrent_layers 40 + n_output_layers 16), weights hold layers 0-71 (at ee517dd388); one entry per operator. His: attention+MLP blocks, 8 + 20 + 8 = 36, the same 72 operators. |
| stepfun-ai/Step-3.5-Flash | num_layers | 45 | 36 sliding-window + 12 global | 6.2% | 48 | decoder layers plus MTP layers: num_hidden_layers (45) + num_nextn_predict_layers (3) | definition | Rule: decoder layers plus MTP layers: num_hidden_layers (45) + num_nextn_predict_layers (3). Ours recomputed: 48. His: 36 sliding-window + 12 global (accepted 47-49). Ours: config.json num_hidden_layers = 45 decoder layers (at ab446a3de5). His: 36 + 12 = 48 entries of layer_types, which also covers the num_nextn_predict_layers = 3 MTP layers. |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | params_active | 3,353,032,704 | 30B total, 3.3B active (11% active) | 1.6% | 3,353,032,704 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active). Ours recomputed: 3,353,032,704. His: 3.3B (accepted 3,234,000,000-3,366,000,000); gap after the rule 1.61%. |
| RecursiveSI/flint | params_active | 2,822,983,456 | 39.8B total, 2.5B active (6.3% active) | 12.9% | 2,515,488,544 | default: per-token count (unselected experts, MTP heads and lookup tables not active); card states: excluding input embedding | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active); card states: excluding input embedding. Ours recomputed: 2,515,488,544. His: 2.5B (accepted 2,450,000,000-2,550,000,000); gap after the rule 0.62%. Card: "/ Active parameters per token / Approximately 2.5B, including the output head¹ / [footnote] ¹ The active-parameter convention includes the output head and excludes the full input embedding matrix. The active body alone is approximately 2.2B parameters. “40B-A2.5B” is the rounded model-size designation.". |
| deepseek-ai/DeepSeek-R1 | params_active | 37,554,471,512 | 671B total, 37B active (5.5% active) | 1.5% | 37,554,471,512 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active). Ours recomputed: 37,554,471,512. His: 37B (accepted 36,260,000,000-37,740,000,000); gap after the rule 1.50%. |
| deepseek-ai/DeepSeek-V3 | params_active | 37,554,471,512 | 671B total, 37B active (5.5% active) | 1.5% | 37,554,471,512 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active). Ours recomputed: 37,554,471,512. His: 37B (accepted 36,260,000,000-37,740,000,000); gap after the rule 1.50%. |
| deepseek-ai/DeepSeek-V4-Pro | params_active | 49,779,081,059 | 1.6T total, 49B active (3.1% active) | 1.6% | 49,779,081,059 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active). Ours recomputed: 49,779,081,059. His: 49B (accepted 48,020,000,000-49,980,000,000); gap after the rule 1.59%. |
| inclusionAI/Ling-2.5-1T | params_active | 63,580,258,816 | 1T total, 63B active (6.3% active) | 0.9% | 63,580,258,816 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active). Ours recomputed: 63,580,258,816. His: 63B (accepted 61,740,000,000-64,260,000,000); gap after the rule 0.92%. |
| inclusionAI/Ling-2.6-1T | params_active | 63,580,258,816 | 1T total, 63B active (6.3% active) | 0.9% | 63,580,258,816 | default: per-token count (unselected experts, MTP heads and lookup tables not active) | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active). Ours recomputed: 63,580,258,816. His: 63B (accepted 61,740,000,000-64,260,000,000); gap after the rule 0.92%. |
| sarvamai/sarvam-30b | params_active | 4,520,575,616 | 30B total, 2.4B active (8% active) | 88.4% | 2,373,091,968 | default: per-token count (unselected experts, MTP heads and lookup tables not active); card states: excluding input embedding, output head | definition | Rule: default: per-token count (unselected experts, MTP heads and lookup tables not active); card states: excluding input embedding, output head. Ours recomputed: 2,373,091,968. His: 2.4B (accepted 2,350,000,000-2,450,000,000); gap after the rule 1.12%. Card: "**Sarvam-30B** is an advanced Mixture-of-Experts (MoE) model with 2.4B non-embedding active parameters, designed primarily for practical deployment.". |
| CohereLabs/command-a-plus-05-2026-w4a4 | params_total | 218,750,558,544 | 218B total, 25B active/token (11.5% active) | 0.3% | 218,255,218,784 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 218,255,218,784. His: 218B total (accepted 213,640,000,000-222,360,000,000); gap after the rule 0.12%. |
| LiquidAI/LFM2.5-350M | params_total | 354,483,968 | 350M parameters | 1.3% | 354,483,968 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 354,483,968. His: 350M parameters (accepted 343,000,000-357,000,000); gap after the rule 1.28%. |
| MiniMaxAI/MiniMax-M2 | params_total | 228,703,644,928 | 230B total, 10B active (4.3% active) | 0.6% | 228,703,644,928 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 228,703,644,928. His: 230B total (accepted 225,400,000,000-234,600,000,000); gap after the rule 0.56%. |
| MiniMaxAI/MiniMax-M2.5 | params_total | 228,703,644,928 | 230B total, 10B active (4.3% active) | 0.6% | 228,703,644,928 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 228,703,644,928. His: 230B total (accepted 225,400,000,000-234,600,000,000); gap after the rule 0.56%. |
| MiniMaxAI/MiniMax-M2.7 | params_total | 228,703,644,928 | 230B total, 10B active (4.3% active) | 0.6% | 228,703,644,928 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 228,703,644,928. His: 230B total (accepted 225,400,000,000-234,600,000,000); gap after the rule 0.56%. |
| MiniMaxAI/MiniMax-M3 | params_total | 427,040,140,160 | 428B total, 23B active (5.4% active) | 0.2% | 426,174,572,928 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 426,174,572,928. His: 428B total (accepted 419,440,000,000-436,560,000,000); gap after the rule 0.43%. |
| MiniMaxAI/MiniMax-M3.1-Flash | params_total | 427,381,965,824 | 428B total, 23B active (5.4% active) | 0.1% | 426,516,398,592 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 426,516,398,592. His: 428B total (accepted 419,440,000,000-436,560,000,000); gap after the rule 0.35%. |
| PrimeIntellect/INTELLECT-3 | params_total | 106,852,251,264 | 106B total, 12B active (11.3% active) | 0.8% | 106,852,251,264 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 106,852,251,264. His: 106B total (accepted 103,880,000,000-108,120,000,000); gap after the rule 0.80%. |
| Qwen/Qwen3-30B-A3B | params_total | 30,532,122,624 | 30B total, 3B active (10% active) | 1.8% | 30,532,122,624 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 30,532,122,624. His: 30B total (accepted 29,400,000,000-30,600,000,000); gap after the rule 1.77%. |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | params_total | 30,532,122,624 | 30B total, 3.3B active (11% active) | 1.8% | 30,532,122,624 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 30,532,122,624. His: 30B total (accepted 29,400,000,000-30,600,000,000); gap after the rule 1.77%. |
| Qwen/Qwen3-Next-80B-A3B-Instruct | params_total | 81,324,862,720 | 80B total, 3B active (3.8% active) | 1.7% | 81,324,862,720 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 81,324,862,720. His: 80B total (accepted 78,400,000,000-81,600,000,000); gap after the rule 1.66%. |
| Qwen/Qwen3.5-397B-A17B | params_total | 403,397,928,944 | 397B total, 17B active (4.3% active) | 1.6% | 402,941,918,464 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 402,941,918,464. His: 397B total (accepted 389,060,000,000-404,940,000,000); gap after the rule 1.50%. |
| Qwen/Qwen3.6-27B | params_total | 27,781,427,952 | 27B parameters | 2.9% | 27,320,697,856 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 27,320,697,856. His: 27B parameters (accepted 26,460,000,000-27,540,000,000); gap after the rule 1.19%. |
| Qwen/Qwen3.6-35B-A3B | params_total | 35,951,822,704 | 35B total, 3B active (8.6% active) | 2.7% | 35,505,251,456 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 35,505,251,456. His: 35B total (accepted 34,300,000,000-35,700,000,000); gap after the rule 1.44%. |
| Qwen/Qwen3.8-27B | params_total | 27,781,427,952 | 27B parameters | 2.9% | 27,320,697,856 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 27,320,697,856. His: 27B parameters (accepted 26,460,000,000-27,540,000,000); gap after the rule 1.19%. |
| XiaomiMiMo/MiMo-V2-Flash | params_total | 309,785,318,400 | 309B total, 15B active (4.9% active) | 0.3% | 309,785,318,400 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 309,785,318,400. His: 309B total (accepted 302,820,000,000-315,180,000,000); gap after the rule 0.25%. |
| XiaomiMiMo/MiMo-V2.5 | params_total | 310,775,040,000 | 310B total, 15B active (4.8% active) | 0.3% | 310,046,445,568 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 310,046,445,568. His: 310B total (accepted 303,800,000,000-316,200,000,000); gap after the rule 0.01%. |
| deepseek-ai/DeepSeek-V3 | params_total | 684,531,386,000 | 671B total, 37B active (5.5% active) | 2.0% | 671,067,257,432 | default: all stored tensors minus vision tower and projector (text-only figure); card states: excluding MTP / auxiliary heads | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure); card states: excluding MTP / auxiliary heads. Ours recomputed: 671,067,257,432. His: 671B total (accepted 657,580,000,000-684,420,000,000); gap after the rule 0.01%. Card: "**NOTE: The total size of DeepSeek-V3 models on HuggingFace is 685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights.**". |
| deepseek-ai/DeepSeek-V4.1-Flash | params_total | 763,205,315,794 | 552B backbone + 196B Engram; 8B active prefill, 16B active decode (2.9% of backbone) | 2.0% | 762,793,473,234 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 762,793,473,234. His: 552B backbone + 196B Engram (accepted 733,040,000,000-762,960,000,000); gap after the rule 1.98%. |
| google/gemma-3-270m | params_total | 268,098,176 | 270M parameters | 0.7% | 268,098,176 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 268,098,176. His: 270M parameters (accepted 264,600,000-275,400,000); gap after the rule 0.70%. |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | params_total | 401,583,781,376 | 400B total, 17B active (4.3% active) | 0.4% | 400,711,848,960 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 400,711,848,960. His: 400B total (accepted 392,000,000,000-408,000,000,000); gap after the rule 0.18%. |
| meta-models/Muse-Glimmer-30B | params_total | 29,776,626,688 | 29.6B parameters (including perception encoder) | 0.6% | 29,776,626,688 | card: all stored tensors including the vision encoder | definition | Rule: card: all stored tensors including the vision encoder. Ours recomputed: 29,776,626,688. His: 29.6B parameters (including perception encoder) (accepted 29,008,000,000-30,192,000,000); gap after the rule 0.60%. Card: "/ **Number of Parameters** / 29.6B (including vision encoder) /". |
| ornith-ai/Ornith-1.5-35B-A3B | params_total | 35,951,822,704 | 35B total, 3B active (8.6% active) | 2.7% | 35,505,251,456 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 35,505,251,456. His: 35B total (accepted 34,300,000,000-35,700,000,000); gap after the rule 1.44%. |
| sarvamai/sarvam-105b | params_total | 106,031,767,424 | 105B total, 10.3B active (9.8% active) | 1.0% | 106,031,767,424 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 106,031,767,424. His: 105B total (accepted 102,900,000,000-107,100,000,000); gap after the rule 0.98%. |
| zai-org/GLM-4.5 | params_total | 358,337,791,296 | 355B total, 32B active (9% active) | 0.9% | 358,337,791,296 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 358,337,791,296. His: 355B total (accepted 347,900,000,000-362,100,000,000); gap after the rule 0.94%. |
| zai-org/GLM-4.7 | params_total | 358,337,791,296 | 355B total, 32B active (9% active) | 0.9% | 358,337,791,296 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 358,337,791,296. His: 355B total (accepted 347,900,000,000-362,100,000,000); gap after the rule 0.94%. |
| zai-org/GLM-5 | params_total | 753,864,139,008 | 744B total, 40B active (5.4% active) | 1.3% | 753,864,139,008 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 753,864,139,008. His: 744B total (accepted 729,120,000,000-758,880,000,000); gap after the rule 1.33%. |
| zai-org/GLM-5.1 | params_total | 753,864,139,008 | 744B total, 40B active (5.4% active) | 1.3% | 753,864,139,008 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 753,864,139,008. His: 744B total (accepted 729,120,000,000-758,880,000,000); gap after the rule 1.33%. |
| zai-org/GLM-5.2 | params_total | 753,329,940,480 | 744B total, 40B active (5.4% active) | 1.3% | 753,329,940,480 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 753,329,940,480. His: 744B total (accepted 729,120,000,000-758,880,000,000); gap after the rule 1.25%. |
| zai-org/GLM-5.3-Flash | params_total | 321,342,220,638 | 320B total, 18B active (5.6% active) | 0.4% | 320,778,593,630 | default: all stored tensors minus vision tower and projector (text-only figure) | definition | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 320,778,593,630. His: 320B total (accepted 313,600,000,000-326,400,000,000); gap after the rule 0.24%. |
| google/gemma-3-27b-it | vocab_size | 262,208 | 262,144 (~262k) | 0.0% | 262,144 | tokenizer.json base vocabulary instead of embedding rows | definition | Rule: tokenizer.json base vocabulary instead of embedding rows. Ours recomputed: 262,144. His: 262,144 (~262k) (accepted 256,901-267,387). Ours: 262,208 embedding rows (config/weights at 005ad3404e). His: 262,144, the tokenizer.json base vocabulary at the same commit (6,415 added tokens reuse ids inside it). |
| thinkingmachines/Inkling | vocab_size | 201,024 | 200,058 (~200k) | 0.5% | 200,058 | tokenizer.json size (highest token id + 1) instead of padded embedding rows | definition | Rule: tokenizer.json size (highest token id + 1) instead of padded embedding rows. Ours recomputed: 200,058. His: 200,058 (~200k) (accepted 196,057-204,059). Ours: 201,024 embedding rows, padded (at 828496eeae). His: 200,058, the tokenizer.json size at the same commit (highest token id + 1). |

## Rows not on the fixed default rule

| Model | Field | Group | Rule | Source quote |
|---|---|---|---|---|
| HuggingFaceTB/SmolLM3-3B-Base | context_length | definition | advertised context the model card states ("128k") | xture and training configs - **Long context:** Trained on 64k context and suppots up to **128k tokens** using YARN extrapolation - **Multilingual**: 6 natively supported (English, Fren |
| arcee-ai/Trinity-Large-Base | context_length | definition | advertised context the model card states ("512k") | se layers / 6 / / Pretraining context length / 8,192 / / Context length after extention / 512k / / Architecture / Sparse MoE / ## Benchmark Results / Benchmark / N-sh |
| deepseek-ai/DeepSeek-R1 | context_length | definition | advertised context the model card states ("128K") | -: / :------------: / :------------: / :------------: / / DeepSeek-R1-Zero / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-R1 / 671B / 37B / 128K / [🤗 HuggingFace] / < |
| deepseek-ai/DeepSeek-V3 | context_length | definition | advertised context the model card states ("128K") | -: / :------------: / :------------: / :------------: / / DeepSeek-V3-Base / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-V3 / 671B / 37B / 128K / [🤗 HuggingFace] / < |
| inclusionAI/Ling-2.5-1T | context_length | definition | advertised context the model card states ("256K") | ----------------------------------------------------------------------: / / Ling-2.5-1T / 256K -> 1M / [🤗 HuggingFace] &nbsp;&nbsp; [🤖 ModelScope] / </center> Note: If you are int |
| inclusionAI/Ling-3.1-flash | context_length | definition | advertised context the model card states ("1M") | world tasks. With **560B** total parameters, **25B** active parameters per token, and a **1M-token context window**, Ling-3.1-flash builds upon Ling-3.0-flash with improvements in ar |
| mistralai/Mistral-Small-4-119B-2603 | context_length | definition | advertised context the model card states ("256k") | E**: 128 experts, 4 active. - **119B parameters**, with **6.5B activated per token**. - **256k context length**. - **Multimodal input**: Accepts both text and image input, with text ou |
| Zyphra/ZAYA1-8B | num_layers | definition | attention and FFN entries counted separately: 2 x config num_hidden_layers (40) | (file evidence) config.json num_hidden_layers = 40, layer_types = 40 x 'hybrid' (attention and MoE in each layer) at 67d34da515. His: attention entries and FFN entries counted separately (40 + 40). |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | num_layers | definition | attention+MLP blocks: config num_hidden_layers (72 operator entries) / 2 | (file evidence) config.json num_hidden_layers = 72, layer_types alternate attention and MLP (n_input_layers 16 + n_recurrent_layers 40 + n_output_layers 16), weights hold layers 0-71 (at ee517dd388); one entry per operator. His: attention+MLP blocks, 8 + 20 + 8 = 36, the same 72 operators. |
| stepfun-ai/Step-3.5-Flash | num_layers | definition | decoder layers plus MTP layers: num_hidden_layers (45) + num_nextn_predict_layers (3) | (file evidence) config.json num_hidden_layers = 45 decoder layers (at ab446a3de5). His: 36 + 12 = 48 entries of layer_types, which also covers the num_nextn_predict_layers = 3 MTP layers. |
| RecursiveSI/flint | params_active | definition | default: per-token count (unselected experts, MTP heads and lookup tables not active); card states: excluding input embedding | / Active parameters per token / Approximately 2.5B, including the output head¹ / [footnote] ¹ The active-parameter convention includes the output head and excludes the full input embedding matrix. The active body alone is approximately 2.2B parameters. “40B-A2.5B” is the rounded model-size designation. |
| sarvamai/sarvam-30b | params_active | definition | default: per-token count (unselected experts, MTP heads and lookup tables not active); card states: excluding input embedding, output head | **Sarvam-30B** is an advanced Mixture-of-Experts (MoE) model with 2.4B non-embedding active parameters, designed primarily for practical deployment. |
| deepseek-ai/DeepSeek-V3 | params_total | definition | default: all stored tensors minus vision tower and projector (text-only figure); card states: excluding MTP / auxiliary heads | **NOTE: The total size of DeepSeek-V3 models on HuggingFace is 685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights.** |
| meta-models/Muse-Glimmer-30B | params_total | definition | card: all stored tensors including the vision encoder | / **Number of Parameters** / 29.6B (including vision encoder) / |
| google/gemma-3-27b-it | vocab_size | definition | tokenizer.json base vocabulary instead of embedding rows | (file evidence) 262,208 embedding rows (config/weights at 005ad3404e). His: 262,144, the tokenizer.json base vocabulary at the same commit (6,415 added tokens reuse ids inside it). |
| thinkingmachines/Inkling | vocab_size | definition | tokenizer.json size (highest token id + 1) instead of padded embedding rows | (file evidence) 201,024 embedding rows, padded (at 828496eeae). His: 200,058, the tokenizer.json size at the same commit (highest token id + 1). |
| JetBrains/Mellum2-12B-A2.5B-Thinking | params_active | name_rounding | nominal size: model name | model name: Mellum2-12B-A2.5B-Thinking |
| MiniMaxAI/MiniMax-M3 | params_active | name_rounding | nominal size: card '~' figure | It has ~428B parameters and ~23B activated parameters. |
| MiniMaxAI/MiniMax-M3.1-Flash | params_active | name_rounding | nominal size: card '~' figure | - **Parameters:** Approximately 428B total; 23B activated per token |
| Qwen/Qwen3-Next-80B-A3B-Instruct | params_active | name_rounding | nominal size: model name | model name: Qwen3-Next-80B-A3B-Instruct |
| meta-models/Muse-Spark-Open-816B-A42B | params_active | name_rounding | nominal size: model name | model name: Muse-Spark-Open-816B-A42B |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | params_active | name_rounding | nominal size: model name | model name: NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 |
| Nanbeige/Nanbeige4.1-3B | params_total | name_rounding | nominal size: model name | model name: Nanbeige4.1-3B |
| Qwen/Qwen3-32B | params_total | name_rounding | nominal size: model name | model name: Qwen3-32B |
| ibm-granite/granite-4.1-30b | params_total | name_rounding | nominal size: model name | model name: granite-4.1-30b |
| moonshotai/Kimi-Linear-48B-A3B-Base | params_total | name_rounding | nominal size: model name | model name: Kimi-Linear-48B-A3B-Base |
| sarvamai/sarvam-30b | params_total | name_rounding | nominal size: model name | model name: sarvam-30b |

## Rows that changed group since the previous run

| Model | Field | Before | Now |
|---|---|---|---|
| Aleph-Alpha/Kolibri-1 | params_active | definition | ours_wrong |
| JetBrains/Mellum2-12B-A2.5B-Thinking | params_active | ours_wrong | name_rounding |
| MiniMaxAI/MiniMax-M2 | params_active | definition | ours_wrong |
| MiniMaxAI/MiniMax-M2.5 | params_active | definition | ours_wrong |
| MiniMaxAI/MiniMax-M2.7 | params_active | definition | ours_wrong |
| MiniMaxAI/MiniMax-M3 | params_active | ours_wrong | name_rounding |
| MiniMaxAI/MiniMax-M3.1-Flash | params_active | ours_wrong | name_rounding |
| Nanbeige/Nanbeige4.1-3B | params_total | definition | name_rounding |
| PrimeIntellect/INTELLECT-3 | params_active | definition | ours_wrong |
| Qwen/Qwen3-32B | params_total | definition | name_rounding |
| Qwen/Qwen3-Next-80B-A3B-Instruct | params_active | definition | name_rounding |
| Qwen/Qwen3.8-Flash-Next | params_active | definition | ours_wrong |
| XiaomiMiMo/MiMo-V2.5 | params_active | definition | ours_wrong |
| XiaomiMiMo/MiMo-V2.6-Flash-RL | params_active | definition | ours_wrong |
| Zyphra/ZAYA1-8B | params_total | definition | ours_wrong |
| Zyphra/ZAYA1-8B | params_active | definition | ours_wrong |
| baidu/ERNIE-5.5-Flash-Base | params_total | definition | ours_wrong |
| deepseek-ai/DeepSeek-R1 | params_total | definition | ours_wrong |
| deepseek-ai/DeepSeek-V3.2 | params_total | definition | ours_wrong |
| deepseek-ai/DeepSeek-V3.2 | params_active | definition | ours_wrong |
| deepseek-ai/DeepSeek-V3.2 | context_length | ours_wrong | his_wrong |
| deepseek-ai/DeepSeek-V4-Flash | params_total | definition | ours_wrong |
| deepseek-ai/DeepSeek-V4-Flash | params_active | definition | ours_wrong |
| deepseek-ai/DeepSeek-V4.1-Flash | params_active | definition | ours_wrong |
| ibm-granite/granite-4.1-30b | params_total | ours_wrong | name_rounding |
| inclusionAI/Ling-3.0-flash | params_total | definition | ours_wrong |
| inclusionAI/Ling-3.0-flash | params_active | definition | ours_wrong |
| inclusionAI/Ling-3.1-flash | params_total | definition | ours_wrong |
| meta-models/Muse-Spark-Open-816B-A42B | params_active | not compared | name_rounding |
| meta-models/Muse-Spark-Open-816B-A42B | context_length | not compared | agree |
| meta-models/Muse-Spark-Open-816B-A42B | vocab_size | not compared | agree |
| meta-models/Muse-Spark-Open-816B-A42B | num_layers | not compared | agree |
| meta-models/Muse-Spark-Open-816B-A42B | decoder_type | not compared | agree |
| microsoft/phi-4 | params_total | definition | ours_wrong |
| moonshotai/Kimi-Linear-48B-A3B-Base | params_total | definition | name_rounding |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | params_active | definition | name_rounding |
| openai/gpt-oss-120b | params_active | definition | ours_wrong |
| openai/gpt-oss-20b | params_active | definition | ours_wrong |
| sarvamai/sarvam-105b | params_active | definition | ours_wrong |
| sarvamai/sarvam-30b | params_total | definition | name_rounding |
| zai-org/GLM-4.5 | params_active | definition | ours_wrong |
| zai-org/GLM-4.5-Air | params_total | definition | ours_wrong |
| zai-org/GLM-4.5-Air | params_active | definition | ours_wrong |
| zai-org/GLM-4.7 | params_active | definition | ours_wrong |
| zai-org/GLM-5 | params_active | definition | ours_wrong |
| zai-org/GLM-5.1 | params_active | definition | ours_wrong |
| zai-org/GLM-5.2 | params_active | definition | ours_wrong |
| zai-org/GLM-5.3-Flash | params_active | definition | ours_wrong |

## Not analyzed

Not counted in the groups above: there is no ArchWatch value to compare. `access_needed` repos need the owner to request access; `pipeline_error` repos use formats the extractor does not support yet (open ArchWatch work).

| Entry | Repo | Status | Reason |
|---|---|---|---|
| Llama 3.2 1B | - | unmapped | no config.repo; other links disagree on the variant (https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt; https://artificialanalysis.ai/models/llama-3-2-instruct-1b) |
| Llama 3.2 3B | - | unmapped | no config.repo; other links disagree on the variant (https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/LICENSE.txt; https://artificialanalysis.ai/models/llama-3-2-instruct-3b) |
| - | fdtn-ai/antares-1b | access_needed | HTTPError: HTTP Error 403: Forbidden |
| - | badtheorylabs/BTL-3 | pipeline_error | ValueError: No readable root config.json; this source format requires a separate adapter |
| - | openai-community/gpt2-xl | pipeline_error | KeyError: 'num_hidden_layers' |
| - | meituan-longcat/LongCat-Flash-Lite | pipeline_error | KeyError: 'num_hidden_layers' |
| - | mistralai/Mistral-Large-3-675B-Instruct-2512 | pipeline_error | ValueError: No readable root config.json; this source format requires a separate adapter |
| - | Motif-Technologies/Motif-3-Beta | pipeline_error | TypeError: unsupported operand type(s) for %: 'int' and 'str' |
| - | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | pipeline_error | KeyError: 'num_hidden_layers' |
| - | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | pipeline_error | ValueError: Invalid Nemotron-H block pattern |
| - | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | pipeline_error | ValueError: Invalid Nemotron-H block pattern |
| - | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | pipeline_error | KeyError: 'num_hidden_layers' |
| - | Soofi-Project/Soofi-S-Base | access_needed | HTTPError: HTTP Error 403: Forbidden |
| - | NX-AI/xLSTM-7b | pipeline_error | KeyError: 'num_hidden_layers' |

## Mapping

| Entry | Hub repo | How |
|---|---|---|
| Arcee AI Trinity Large 400B | arcee-ai/Trinity-Large-Base | config.repo link |
| Antares 1B | fdtn-ai/antares-1b | config.repo link |
| BTL-3 27B | badtheorylabs/BTL-3 | config.repo link |
| DeepSeek R1 | deepseek-ai/DeepSeek-R1 | config.repo link |
| DeepSeek V3 | deepseek-ai/DeepSeek-V3 | config.repo link |
| DeepSeek V3.2 | deepseek-ai/DeepSeek-V3.2 | config.repo link |
| DeepSeek V4-Pro | deepseek-ai/DeepSeek-V4-Pro | config.repo link |
| DeepSeek V4-Flash | deepseek-ai/DeepSeek-V4-Flash | config.repo link |
| DeepSeek V4.1-Flash | deepseek-ai/DeepSeek-V4.1-Flash | config.repo link |
| Dory1.0 4B | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | config.repo link |
| ERNIE 5.5 Flash Base | baidu/ERNIE-5.5-Flash-Base | config.repo link |
| Ember-1 | moonshotai/Kimi-K3 | config.repo link |
| GLM-4.5 355B | zai-org/GLM-4.5 | config.repo link |
| GLM-4.5-Air | zai-org/GLM-4.5-Air | config.repo link |
| GLM-4.7 355B | zai-org/GLM-4.7 | config.repo link |
| GLM-5 744B | zai-org/GLM-5 | config.repo link |
| GLM-5.1 | zai-org/GLM-5.1 | config.repo link |
| GLM-5.2 | zai-org/GLM-5.2 | config.repo link |
| GLM-5.3-Flash | zai-org/GLM-5.3-Flash | config.repo link |
| GPT-2 XL 1.5B | openai-community/gpt2-xl | config.repo link |
| GPT-OSS 120B | openai/gpt-oss-120b | config.repo link |
| GPT-OSS 20B | openai/gpt-oss-20b | config.repo link |
| Gemma 3 270M | google/gemma-3-270m | config.repo link |
| Gemma 3 27B | google/gemma-3-27b-it | config.repo link |
| Gemma 4 26B-A4B | google/gemma-4-26B-A4B-it | config.repo link |
| Gemma 4 31B | google/gemma-4-31B-it | config.repo link |
| Gemma 4 12B | google/gemma-4-12B-it | config.repo link |
| Gemma 4 E2B | google/gemma-4-E2B-it | config.repo link |
| Gemma 4 E4B | google/gemma-4-E4B-it | config.repo link |
| Granite 4.1 30B | ibm-granite/granite-4.1-30b | config.repo link |
| Grok 2.5 270B | xai-org/grok-2 | config.repo link |
| Tencent Hy3-preview 295B-A21B | tencent/Hy3-preview | config.repo link |
| Tencent Hy4-preview 770B-A49B | tencent/Hy4-preview | config.repo link |
| INTELLECT-3 | PrimeIntellect/INTELLECT-3 | config.repo link |
| Kimi K2 | moonshotai/Kimi-K2-Base | config.repo link |
| Kimi K2.5 | moonshotai/Kimi-K2.5 | config.repo link |
| Kimi K2.6 | moonshotai/Kimi-K2.6 | config.repo link |
| Kimi K2.7 Code | moonshotai/Kimi-K2.7-Code | config.repo link |
| Kimi K3 | moonshotai/Kimi-K3 | config.repo link |
| Kimi K3.1 Pro | moonshotai/Kimi-K3.1-Pro | config.repo link |
| Kimi Linear 48B-A3B | moonshotai/Kimi-Linear-48B-A3B-Base | config.repo link |
| Kolibri-1 78B | Aleph-Alpha/Kolibri-1 | config.repo link |
| Laguna XS.2 | poolside/Laguna-XS.2 | config.repo link |
| Laguna XS 2.1 | poolside/Laguna-XS-2.1 | config.repo link |
| Laguna S 2.1 | poolside/Laguna-S-2.1 | config.repo link |
| LFM2.5 350M | LiquidAI/LFM2.5-350M | config.repo link |
| LFM2.5 1.2B | LiquidAI/LFM2.5-1.2B-Thinking | config.repo link |
| LFM2.5 8B-A1B | LiquidAI/LFM2.5-8B-A1B | config.repo link |
| LFM3 1.7B | LiquidAI/LFM3-1.7B | config.repo link |
| Ling 2.5 1T | inclusionAI/Ling-2.5-1T | config.repo link |
| Ling 2.6 1T | inclusionAI/Ling-2.6-1T | config.repo link |
| Ling 3.0 Flash | inclusionAI/Ling-3.0-flash | config.repo link |
| Ling 3.1 Flash | inclusionAI/Ling-3.1-flash | config.repo link |
| LongCat-Flash-Lite 68.5B-A3B | meituan-longcat/LongCat-Flash-Lite | config.repo link |
| Llama 3.2 1B | - | no config.repo; other links disagree on the variant (https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt; https://artificialanalysis.ai/models/llama-3-2-instruct-1b) |
| Llama 3 8B | meta-llama/Meta-Llama-3-8B | config.repo link |
| Llama 3.2 3B | - | no config.repo; other links disagree on the variant (https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/LICENSE.txt; https://artificialanalysis.ai/models/llama-3-2-instruct-3b) |
| Llama 4 Maverick | meta-llama/Llama-4-Maverick-17B-128E-Instruct | config.repo link |
| JetBrains Mellum2 Thinking 12B-A2.5B | JetBrains/Mellum2-12B-A2.5B-Thinking | config.repo link |
| MiniMax M2 230B | MiniMaxAI/MiniMax-M2 | config.repo link |
| MiniMax M2.5 230B | MiniMaxAI/MiniMax-M2.5 | config.repo link |
| MiniMax M2.7 230B | MiniMaxAI/MiniMax-M2.7 | config.repo link |
| MiniMax M3 428B | MiniMaxAI/MiniMax-M3 | config.repo link |
| MiniMax M3.1-Flash 428B | MiniMaxAI/MiniMax-M3.1-Flash | config.repo link |
| Mistral Large 3 | mistralai/Mistral-Large-3-675B-Instruct-2512 | config.repo link |
| Mistral Large 4 | mistralai/Mistral-Large-4-1T-A52B | config.repo link |
| Mistral Small 3.1 24B | mistralai/Mistral-Small-3.1-24B-Base-2503 | config.repo link |
| Mistral Small 4 | mistralai/Mistral-Small-4-119B-2603 | config.repo link |
| Naive-N0.5-Flash 309B | NaiveAI/Naive-N0.5-Flash | config.repo link |
| Nanbeige 4.1 3B | Nanbeige/Nanbeige4.1-3B | config.repo link |
| Nanbeige 4.2 3B | Nanbeige/Nanbeige4.2-3B | config.repo link |
| Motif 3 Beta | Motif-Technologies/Motif-3-Beta | config.repo link |
| Muse Glimmer 30B | meta-models/Muse-Glimmer-30B | config.repo link |
| Muse Spark Open 816B-A42B | meta-models/Muse-Spark-Open-816B-A42B | config.repo link |
| Nemotron 3.5 Lightning 30B-A3B | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | config.repo link |
| Nemotron 3.5 Super VL 120B-A12B | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | config.repo link |
| Nemotron 3 Nano 30B-A3B | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | config.repo link |
| Nemotron 3 Nano 4B | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | config.repo link |
| Nemotron 3 Super 120B-A12B | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | config.repo link |
| Nemotron 3 Ultra 550B-A55B | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | config.repo link |
| OLMo 2 7B | allenai/OLMo-2-1124-7B-Instruct | config.repo link |
| OLMo 3 32B | allenai/Olmo-3-32B-Think | config.repo link |
| OLMo 3 7B | allenai/Olmo-3-1025-7B | config.repo link |
| Ouro-Thinking 2.6B | ByteDance/Ouro-2.6B-Thinking | config.repo link |
| Ornith 1.5 35B-A3B | ornith-ai/Ornith-1.5-35B-A3B | config.repo link |
| Phi-4 | microsoft/phi-4 | config.repo link |
| Qwen3 0.6B | Qwen/Qwen3-0.6B-Base | config.repo link |
| Qwen3 235B-A22B | Qwen/Qwen3-235B-A22B | config.repo link |
| Qwen3 30B-A3B | Qwen/Qwen3-30B-A3B | config.repo link |
| Qwen3 32B | Qwen/Qwen3-32B | config.repo link |
| Qwen3 4B | Qwen/Qwen3-4B | config.repo link |
| Qwen3 8B | Qwen/Qwen3-8B | config.repo link |
| Qwen3 Next 80B-A3B | Qwen/Qwen3-Next-80B-A3B-Instruct | config.repo link |
| Qwen3 Coder Flash 30B-A3B | Qwen/Qwen3-Coder-30B-A3B-Instruct | config.repo link |
| Qwen3.5 397B | Qwen/Qwen3.5-397B-A17B | config.repo link |
| Qwen3.6 35B-A3B | Qwen/Qwen3.6-35B-A3B | config.repo link |
| Qwen3.6 27B | Qwen/Qwen3.6-27B | config.repo link |
| Qwen3.8 27B | Qwen/Qwen3.8-27B | config.repo link |
| Qwen3.8-Flash-Next 125B-A6B | Qwen/Qwen3.8-Flash-Next | config.repo link |
| Reflection Beam 501B | reflection/Beam-1.0-501B-A23B | config.repo link |
| Recursive Flint 40B-A2.5B | RecursiveSI/flint | config.repo link |
| Sarvam 105B | sarvamai/sarvam-105b | config.repo link |
| Sarvam 30B | sarvamai/sarvam-30b | config.repo link |
| SmolLM3 3B | HuggingFaceTB/SmolLM3-3B-Base | config.repo link |
| Solar Open 2 | upstage/Solar-Open2-250B | config.repo link |
| Soofi S 30B-A3B | Soofi-Project/Soofi-S-Base | config.repo link |
| Step 3.5 Flash 196B | stepfun-ai/Step-3.5-Flash | config.repo link |
| Step-5 Preview 600B | SHSLab/Step-5-Preview-BF16 | config.repo link |
| Tiny Aya 3.35B | CohereLabs/tiny-aya-base | config.repo link |
| VibeThinker-3B | WeiboAI/VibeThinker-3B | config.repo link |
| Xiaomi MiMo-V2-Flash 309B | XiaomiMiMo/MiMo-V2-Flash | config.repo link |
| Xiaomi MiMo-V2.5 310B | XiaomiMiMo/MiMo-V2.5 | config.repo link |
| Xiaomi MiMo-V2.5-Pro 1.02T | XiaomiMiMo/MiMo-V2.5-Pro | config.repo link |
| Xiaomi MiMo-V2.6-Flash-RL 309B | XiaomiMiMo/MiMo-V2.6-Flash-RL | config.repo link |
| Xiaomi MiMo-V2.6-Pro-RL 1.02T | XiaomiMiMo/MiMo-V2.6-Pro-RL | config.repo link |
| ZAYA1-8B | Zyphra/ZAYA1-8B | config.repo link |
| North Mini Code 30B-A3B | CohereLabs/North-Mini-Code-1.0 | config.repo link |
| Command A+ 218B-A25B | CohereLabs/command-a-plus-05-2026-w4a4 | config.repo link |
| xLSTM 7B | NX-AI/xLSTM-7b | config.repo link |
| Inkling | thinkingmachines/Inkling | config.repo link |
