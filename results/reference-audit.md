# Reference audit (internal; not part of the site)

Answer sheet: `rasbt/llm-architecture-gallery` `models.yml`, last changed at commit `c5624cfa9285550769e30cac118ebfc12ecfe5f0` (repository head `660cc1886a9d2a009c46ecc8e17e98b1075f58b2`). Its values are never a source for ArchWatch records.

Entries: 120; mapped to a Hub repo: 118; analyzed by our deterministic pipeline: 104; compared values: 521; agreeing: 410; differences: 111.

| Group | Differences |
|---|---:|
| ours_wrong | 0 |
| his_wrong | 5 |
| definition | 106 |
| unresolved | 0 |

Groups: **ours_wrong** = ArchWatch error still open; **his_wrong** = the model's own files contradict his value; **definition** = both are right under different definitions (both written down); **unresolved** = review queue.

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
| Comparison parser: 'Mamba-2' read as 2 layers, MTP counted as a layer, 128,000 vs 128K not normalised | `test_layer_mix_counts_leading_terms_and_skips_mtp` |

## Differences

| Model | Field | Ours | His | Group | Evidence |
|---|---|---|---|---|---|
| MiniMaxAI/MiniMax-M2.7 | context_length | 204,800 | 196,608 | his_wrong | MiniMaxAI/MiniMax-M2.7 config.json at 0a6536713b: max_position_embeddings = 204800; the README states no context length. 196,608 appears in neither file. |
| Nanbeige/Nanbeige4.2-3B | kv_cache_per_token_bf16 | 90,112 | 176 KiB | his_wrong | Nanbeige/Nanbeige4.2-3B config.json at b82e54bd60: num_hidden_layers = 22, num_key_value_heads = 8, head_dim = 128, no sliding_window. His own formula (layers x KV heads x head_dim x 4) gives 22 x 8 x 128 x 4 = 90,112 B = 88 KiB; his 176 KiB is twice that. |
| google/gemma-3-270m | context_length | 32,768 | 128,000 | his_wrong | google/gemma-3-270m README.md at 9b0cfec892: "Total input context of 128K tokens for the 4B, 12B, and 27B sizes, and 32K tokens for the 1B and 270M sizes"; config.json max_position_embeddings = 32768. 128K is the family figure for the larger sizes. |
| google/gemma-4-12B-it | context_length | 262,144 | 128,000 | his_wrong | google/gemma-4-12B-it README.md at 43c9d014b2: "Context windows of up to 128K tokens (E2B/E4B) and 256K tokens (12B, 26B A4B/31B)" and its table lists Context Length 256K tokens for 12B; config.json text_config.max_position_embeddings = 262144. |
| poolside/Laguna-XS.2 | context_length | 262,144 | 131,072 | his_wrong | poolside/Laguna-XS.2 README.md at a397bde045: "Context window: 262,144 tokens"; config.json max_position_embeddings = 262144. 131,072 appears in neither. |
| Aleph-Alpha/Kolibri-1 | params_active | 3,785,438,560 | 78B total, 3.46B active (4.4% active) | definition | His figure is the one the model card states (README.md at abfe85f673: "tecture / Mixture-of-Experts / / Total parameters / 78B / / Active parameters / token / 3.46B / / Languages / German, English / / Context length / `1,048,576 tokens`; we recommend ≤"). Ours (3,785,438,560) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| CohereLabs/command-a-plus-05-2026-w4a4 | params_total | 218,750,558,544 | 218B total, 25B active/token (11.5% active) | definition | His figure is the one the model card states (README.md at ebd2d72c0f: "Model Summary** Command A+ is an open source model with 25 billion active parameters and 218B total parameters model optimized for agentic, multilingual, and reasoning-heavy tasks wit"). Ours (218,750,558,544) is the exact count of every stored tensor in the safetensors headers. |
| HuggingFaceTB/SmolLM3-3B-Base | context_length | 65,536 | 131,072 | definition | His figure is the one the model card states (README.md at d78a42f791: "xture and training configs - **Long context:** Trained on 64k context and suppots up to **128k tokens** using YARN extrapolation - **Multilingual**: 6 natively supported (English, Fren"). Ours (65,536) is the config.json max_position_embeddings (positions the released config supports). |
| JetBrains/Mellum2-12B-A2.5B-Thinking | kv_cache_per_token_bf16 | 14,336 | 56 KiB | definition | Ours: growing cache per token = 14,336 B (sliding-window layers excluded: their cache is capped at the 1024-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 57,344 B, which matches his value. |
| JetBrains/Mellum2-12B-A2.5B-Thinking | params_active | 2,439,060,736 | 12B total, 2.5B active (20.8% active) | definition | His figure is the size in the model's name ('Mellum2-12B-A2.5B-Thinking'), a rounded label. Ours (2,439,060,736) is the per-token count from tensor headers. |
| LiquidAI/LFM2.5-350M | params_total | 354,483,968 | 350M parameters | definition | Rounding: his '350M' has 2 significant figure(s); ours 354,483,968 rounds to it. Same quantity, different precision. |
| LiquidAI/LFM3-1.7B | kv_cache_per_token_bf16 | 12,288 | 22 KiB | definition | Ours: growing cache per token = 12,288 B (sliding-window layers excluded: their cache is capped at the 256-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 22,528 B, which matches his value. |
| MiniMaxAI/MiniMax-M2 | params_active | 11,031,148,288 | 230B total, 10B active (4.3% active) | definition | His figure is the one the model card states (README.md at 04312f0cfb: "reported from --- ## Why activation size matters By maintaining activations around **10B** , the plan → act → verify loop in the agentic workflow is streamlined, improving respon"). Ours (11,031,148,288) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| MiniMaxAI/MiniMax-M2 | params_total | 228,703,644,928 | 230B total, 10B active (4.3% active) | definition | Rounding: his '230B' has 2 significant figure(s); ours 228,703,644,928 rounds to it. Same quantity, different precision. |
| MiniMaxAI/MiniMax-M2.5 | params_active | 11,031,148,288 | 230B total, 10B active (4.3% active) | definition | Ours: 11,031,148,288 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 9,801,955,072, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| MiniMaxAI/MiniMax-M2.5 | params_total | 228,703,644,928 | 230B total, 10B active (4.3% active) | definition | Rounding: his '230B' has 2 significant figure(s); ours 228,703,644,928 rounds to it. Same quantity, different precision. |
| MiniMaxAI/MiniMax-M2.7 | params_active | 11,031,148,288 | 230B total, 10B active (4.3% active) | definition | Ours: 11,031,148,288 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 9,801,955,072, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| MiniMaxAI/MiniMax-M2.7 | params_total | 228,703,644,928 | 230B total, 10B active (4.3% active) | definition | Rounding: his '230B' has 2 significant figure(s); ours 228,703,644,928 rounds to it. Same quantity, different precision. |
| MiniMaxAI/MiniMax-M3 | params_active | 26,151,229,824 | 428B total, 23B active (5.4% active) | definition | His figure is the one the model card states (README.md at d95e30cb91: "p> MiniMax-M3 is a native multimodal model with 1M context. It has ~428B parameters and ~23B activated parameters. **Highlights:** - **Native Multimodality:** M3 undergoes mixed-mod"). Ours (26,151,229,824) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| MiniMaxAI/MiniMax-M3 | params_total | 427,040,140,160 | 428B total, 23B active (5.4% active) | definition | His figure is the one the model card states (README.md at d95e30cb91: "alt="LICENSE"></a> </p> MiniMax-M3 is a native multimodal model with 1M context. It has ~428B parameters and ~23B activated parameters. **Highlights:** - **Native Multimodality:** M3"). Ours (427,040,140,160) is the exact count of every stored tensor in the safetensors headers. |
| MiniMaxAI/MiniMax-M3.1-Flash | params_active | 26,493,034,284 | 428B total, 23B active (5.4% active) | definition | His figure is the one the model card states (README.md at ce7f4df87d: "ure:** Native multimodal mixture-of-experts - **Parameters:** Approximately 428B total; 23B activated per token - **Inputs / output:** Text, image, and video inputs; text output - *"). Ours (26,493,034,284) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| MiniMaxAI/MiniMax-M3.1-Flash | params_total | 427,381,965,824 | 428B total, 23B active (5.4% active) | definition | His figure is the one the model card states (README.md at ce7f4df87d: "**Architecture:** Native multimodal mixture-of-experts - **Parameters:** Approximately 428B total; 23B activated per token - **Inputs / output:** Text, image, and video inputs; text"). Ours (427,381,965,824) is the exact count of every stored tensor in the safetensors headers. |
| Nanbeige/Nanbeige4.1-3B | params_total | 3,933,637,120 | 3B parameters | definition | His figure is the size in the model's name ('Nanbeige4.1-3B'), a rounded label. Ours (3,933,637,120) is the exact count of every stored tensor. |
| PrimeIntellect/INTELLECT-3 | params_active | 13,424,129,664 | 106B total, 12B active (11.3% active) | definition | Ours: 13,424,129,664 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 12,182,615,680, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| PrimeIntellect/INTELLECT-3 | params_total | 106,852,251,264 | 106B total, 12B active (11.3% active) | definition | His figure is the one the model card states (README.md at ff39d4a468: "a href=" / <a href=" Intellect Platform</a> </p> ## Introduction **INTELLECT-3** is a 106B parameter Mixture-of-Experts reasoning model post-trained from [GLM-4.5-Air-Base] usin"). Ours (106,852,251,264) is the exact count of every stored tensor in the safetensors headers. |
| Qwen/Qwen3-30B-A3B | params_total | 30,532,122,624 | 30B total, 3B active (10% active) | definition | Rounding: his '30B' has 1 significant figure(s); ours 30,532,122,624 rounds to it. Same quantity, different precision. |
| Qwen/Qwen3-32B | params_total | 32,762,123,264 | 32B parameters | definition | His figure is the size in the model's name ('Qwen3-32B'), a rounded label. Ours (32,762,123,264) is the exact count of every stored tensor. |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | params_active | 3,353,032,704 | 30B total, 3.3B active (11% active) | definition | His figure is the one the model card states (README.md at b2cff646eb: "- Training Stage: Pretraining & Post-training - Number of Parameters: 30.5B in total and 3.3B activated - Number of Layers: 48 - Number of Attention Heads : 32 for Q and 4 for KV - N"). Ours (3,353,032,704) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | params_total | 30,532,122,624 | 30B total, 3.3B active (11% active) | definition | Rounding: his '30B' has 1 significant figure(s); ours 30,532,122,624 rounds to it. Same quantity, different precision. |
| Qwen/Qwen3-Next-80B-A3B-Instruct | params_active | 3,874,929,408 | 80B total, 3B active (3.8% active) | definition | His figure is the one the model card states (README.md at 9c7f2fbe84: "Stage: Pretraining (15T tokens) & Post-training - Number of Parameters: 80B in total and 3B activated - Number of Paramaters : 79B - Hidden Dimension: 2048 - Number of Layers: 48"). Ours (3,874,929,408) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| Qwen/Qwen3-Next-80B-A3B-Instruct | params_total | 81,324,862,720 | 80B total, 3B active (3.8% active) | definition | Rounding: his '80B' has 1 significant figure(s); ours 81,324,862,720 rounds to it. Same quantity, different precision. |
| Qwen/Qwen3.5-397B-A17B | params_total | 403,397,928,944 | 397B total, 17B active (4.3% active) | definition | His figure is the one the model card states (README.md at 8472618112: "Training Stage: Pre-training & Post-training - Language Model - Number of Parameters: 397B in total and 17B activated - Hidden Dimension: 4096 - Token Embedding: 248320"). Ours (403,397,928,944) is the exact count of every stored tensor in the safetensors headers. |
| Qwen/Qwen3.6-27B | params_total | 27,781,427,952 | 27B parameters | definition | His figure is the one the model card states (README.md at 8f33b98937: "Training Stage: Pre-training & Post-training - Language Model - Number of Parameters: 27B - Hidden Dimension: 5120 - Token Embedding: 248320 - Number of Layers: 64"). Ours (27,781,427,952) is the exact count of every stored tensor in the safetensors headers. |
| Qwen/Qwen3.6-35B-A3B | params_total | 35,951,822,704 | 35B total, 3B active (8.6% active) | definition | His figure is the one the model card states (README.md at 345d467b11: "Training Stage: Pre-training & Post-training - Language Model - Number of Parameters: 35B in total and 3B activated - Hidden Dimension: 2048 - Token Embedding: 248320"). Ours (35,951,822,704) is the exact count of every stored tensor in the safetensors headers. |
| Qwen/Qwen3.8-27B | params_total | 27,781,427,952 | 27B parameters | definition | His figure is the one the model card states (README.md at 37561816e6: "Training Stage: Pre-training & Post-training - Language Model - Number of Parameters: 27B - Hidden Dimension: 5120 - Token Embedding: 248,320 - Number of Layers: 64"). Ours (27,781,427,952) is the exact count of every stored tensor in the safetensors headers. |
| Qwen/Qwen3.8-Flash-Next | params_active | 58,507,240,320 | 125B core, 6B active (4.8% active); plus 51B n-gram embeddings and 4B MTP | definition | His figure is the one the model card states (README.md at 812d96e10a: "tage: Pre-training & Post-training - Language Model - Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP - Hidden Dimension: 2560 - Token"). Ours (58,507,240,320) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| RecursiveSI/flint | kv_cache_per_token_bf16 | 28,672 | 98 KiB | definition | Ours: growing cache per token = 28,672 B (sliding-window layers excluded: their cache is capped at the 513-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 100,352 B, which matches his value. |
| RecursiveSI/flint | params_active | 2,822,983,456 | 39.8B total, 2.5B active (6.3% active) | definition | His figure is the one the model card states (README.md at 2a247be0c0: "Transformer / / Total parameters / 39.8B / / Active parameters per token / Approximately 2.5B, including the output head¹ / / Layers / 49: 2 dense layers followed by 47 MoE layers / /"). Ours (2,822,983,456) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| XiaomiMiMo/MiMo-V2-Flash | params_total | 309,785,318,400 | 309B total, 15B active (4.9% active) | definition | His figure is the one the model card states (README.md at 8b60b288e8: "<br/> # MiMo-V2-Flash **MiMo-V2-Flash** is a Mixture-of-Experts language model with **309B total parameters** and **15B active parameters**. Designed for high-speed reasoning and a"). Ours (309,785,318,400) is the exact count of every stored tensor in the safetensors headers. |
| XiaomiMiMo/MiMo-V2.5 | params_active | 15,707,829,184 | 310B total, 15B active (4.8% active) | definition | His figure is the one the model card states (README.md at d6bf9db686: "div> ## Model Summary - **Architecture**: Sparse MoE (Mixture of Experts), 310B total / 15B activated parameters - **Context Length**: Up to 1M tokens - **Modalities**: Text, Image,"). Ours (15,707,829,184) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| XiaomiMiMo/MiMo-V2.5 | params_total | 310,775,040,000 | 310B total, 15B active (4.8% active) | definition | Rounding: his '310B' has 2 significant figure(s); ours 310,775,040,000 rounds to it. Same quantity, different precision. |
| XiaomiMiMo/MiMo-V2.6-Flash-RL | params_active | 10,976,076,736 | 309B total, 15B active (4.9% active) | definition | His figure is the one the model card states (README.md at f657ba2f7d: "sks. ## Model Summary - **Architecture**: Sparse MoE (Mixture of Experts), 309B total / 15B activated parameters - **Context Length**: 1M tokens - **Modalities**: Text, Image, Video"). Ours (10,976,076,736) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| XiaomiMiMo/MiMo-V2.6-Pro-RL | params_active | 31,771,694,208 | 1.02T total, 42B active (4.1% active) | definition | His figure is the one the model card states (README.md at a066ba0057: "ks. ## Model Summary - **Architecture**: Sparse MoE (Mixture of Experts), 1.02T total / 42B activated parameters - **Context Length**: 1M tokens - **Modalities**: Text, Image, Video"). Ours (31,771,694,208) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| Zyphra/ZAYA1-8B | num_layers | 40 | 40 CCA/GQA attention + 40 top-1 MoE FFN entries | definition | Ours: config.json num_hidden_layers = 40, layer_types = 40 x 'hybrid' (attention and MoE in each layer) at 67d34da515. His: attention entries and FFN entries counted separately (40 + 40). |
| Zyphra/ZAYA1-8B | params_active | 1,290,742,264 | 8.4B total, 760M active/token (9% active) | definition | His figure is the one the model card states (README.md at 67d34da515: "at [Zyphra/ZAYA1-8B-legacy] * ZAYA1-8B is a small mixture of experts language model with 760M active parameters and 8.4B total parameters trained end-to-end by Zyphra. ZAYA1-8B sets a"). Ours (1,290,742,264) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| Zyphra/ZAYA1-8B | params_total | 8,840,489,464 | 8.4B total, 760M active/token (9% active) | definition | His figure is the one the model card states (README.md at 67d34da515: "* ZAYA1-8B is a small mixture of experts language model with 760M active parameters and 8.4B total parameters trained end-to-end by Zyphra. ZAYA1-8B sets a new standard of intelligen"). Ours (8,840,489,464) is the exact count of every stored tensor in the safetensors headers. |
| allenai/Olmo-3-1025-7B | kv_cache_per_token_bf16 | 131,072 | 512 KiB | definition | Ours: growing cache per token = 131,072 B (sliding-window layers excluded: their cache is capped at the 4096-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 524,288 B, which matches his value. |
| allenai/Olmo-3-32B-Think | kv_cache_per_token_bf16 | 65,536 | 256 KiB | definition | Ours: growing cache per token = 65,536 B (sliding-window layers excluded: their cache is capped at the 4096-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 262,144 B, which matches his value. |
| arcee-ai/Trinity-Large-Base | context_length | 262,144 | 512,000 | definition | His figure is the one the model card states (README.md at b2a665b40e: "se layers / 6 / / Pretraining context length / 8,192 / / Context length after extention / 512k / / Architecture / Sparse MoE / ## Benchmark Results / Benchmark / N-sh"). Ours (262,144) is the config.json max_position_embeddings (positions the released config supports). |
| baidu/ERNIE-5.5-Flash-Base | params_total | 286,262,969,530 | 280B total, 14B active (5.0% active) | definition | His figure is the one the model card states (README.md at 028544b2ff: "on **ERNIE 5.5 Flash** is an efficient sparse Mixture-of-Experts language model with **280B total parameters and 14B activated per token**, pre-trained from scratch on **33.5T token"). Ours (286,262,969,530) is the exact count of every stored tensor in the safetensors headers. |
| deepseek-ai/DeepSeek-R1 | context_length | 163,840 | 128,000 | definition | His figure is the one the model card states (README.md at 56d4cbbb4d: "-: / :------------: / :------------: / :------------: / / DeepSeek-R1-Zero / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-R1 / 671B / 37B / 128K / [🤗 HuggingFace] / <"). Ours (163,840) is the config.json max_position_embeddings (positions the released config supports). |
| deepseek-ai/DeepSeek-R1 | params_active | 37,554,471,512 | 671B total, 37B active (5.5% active) | definition | His figure is the one the model card states (README.md at 56d4cbbb4d: "-------: / :------------: / :------------: / :------------: / / DeepSeek-R1-Zero / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-R1 / 671B / 37B / 128K / [🤗 HuggingFace]"). Ours (37,554,471,512) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| deepseek-ai/DeepSeek-R1 | params_total | 684,531,386,000 | 671B total, 37B active (5.5% active) | definition | His figure is the one the model card states (README.md at 56d4cbbb4d: ":------------: / :------------: / :------------: / :------------: / / DeepSeek-R1-Zero / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-R1 / 671B / 37B / 128K / [🤗 Hugging"). Ours (684,531,386,000) is the exact count of every stored tensor in the safetensors headers. |
| deepseek-ai/DeepSeek-V3 | context_length | 163,840 | 128,000 | definition | His figure is the one the model card states (README.md at e815299b0b: "-: / :------------: / :------------: / :------------: / / DeepSeek-V3-Base / 671B / 37B / 128K / [🤗 HuggingFace] / / DeepSeek-V3 / 671B / 37B / 128K / [🤗 HuggingFace] / <"). Ours (163,840) is the config.json max_position_embeddings (positions the released config supports). |
| deepseek-ai/DeepSeek-V3 | params_active | 37,554,471,512 | 671B total, 37B active (5.5% active) | definition | His figure is the one the model card states (README.md at e815299b0b: "DeepSeek-V3, a strong Mixture-of-Experts language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, De"). Ours (37,554,471,512) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| deepseek-ai/DeepSeek-V3 | params_total | 684,531,386,000 | 671B total, 37B active (5.5% active) | definition | His figure is the one the model card states (README.md at e815299b0b: ". Introduction We present DeepSeek-V3, a strong Mixture-of-Experts language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and c"). Ours (684,531,386,000) is the exact count of every stored tensor in the safetensors headers. |
| deepseek-ai/DeepSeek-V3.2 | context_length | 163,840 | 128,000 | definition | Ours: config.json max_position_embeddings = 163840 (deepseek-ai/DeepSeek-V3.2 at a7e62ac04e). His: 128K, the advertised context of the family; DeepSeek-V3.2's own README states none, and DeepSeek-V3's README release table states 128K (at e815299b0b). |
| deepseek-ai/DeepSeek-V3.2 | params_active | 38,406,046,640 | 671B total, 37B active (5.5% active) | definition | Ours: 38,406,046,640 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 36,552,688,560, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| deepseek-ai/DeepSeek-V3.2 | params_total | 685,396,921,376 | 671B total, 37B active (5.5% active) | definition | Ours: 685,396,921,376 (all stored tensors). His figure counts a narrower scope: ours without MTP/auxiliary heads, without the input embedding is 670,992,153,520, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| deepseek-ai/DeepSeek-V4-Flash | params_active | 13,799,632,471 | 284B total, 13B active (4.6% active) | definition | His figure is the one the model card states (README.md at 60d8d70770: "ro** with 1.6T parameters (49B activated) and **DeepSeek-V4-Flash** with 284B parameters (13B activated) — both supporting a context length of **one million tokens**. DeepSeek-V4 ser"). Ours (13,799,632,471) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| deepseek-ai/DeepSeek-V4-Flash | params_total | 290,942,289,362 | 284B total, 13B active (4.6% active) | definition | His figure is the one the model card states (README.md at 60d8d70770: "— **DeepSeek-V4-Pro** with 1.6T parameters (49B activated) and **DeepSeek-V4-Flash** with 284B parameters (13B activated) — both supporting a context length of **one million tokens**."). Ours (290,942,289,362) is the exact count of every stored tensor in the safetensors headers. |
| deepseek-ai/DeepSeek-V4-Pro | params_active | 49,779,081,059 | 1.6T total, 49B active (3.1% active) | definition | His figure is the one the model card states (README.md at b5968e9190: "o strong Mixture-of-Experts language models — **DeepSeek-V4-Pro** with 1.6T parameters (49B activated) and **DeepSeek-V4-Flash** with 284B parameters (13B activated) — both supporti"). Ours (49,779,081,059) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| deepseek-ai/DeepSeek-V4.1-Flash | params_active | 213,479,777,904 | 552B backbone + 196B Engram; 8B active prefill, 16B active decode (2.9% of backbone) | definition | His figure is the one the model card states (README.md at 314113f8a7: "ved from each decoder layer's own hidden states. This allows the model to activate only **8B parameters per token during prefill** and **16B during decode**, substantially improving"). Ours (213,479,777,904) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| deepseek-ai/DeepSeek-V4.1-Flash | params_total | 763,205,315,794 | 552B backbone + 196B Engram; 8B active prefill, 16B active decode (2.9% of backbone) | definition | His figure is the one the model card states (README.md at 314113f8a7: "ction We introduce **DeepSeek-V4.1-Flash**, a multimodal Mixture-of-Experts model with 552B backbone parameters and support for contexts of up to one million tokens. The model nativ"). Ours (763,205,315,794) is the exact count of every stored tensor in the safetensors headers. |
| google/gemma-3-270m | kv_cache_per_token_bf16 | 3,072 | 18 KiB | definition | Ours: growing cache per token = 3,072 B (sliding-window layers excluded: their cache is capped at the 512-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 18,432 B, which matches his value. |
| google/gemma-3-270m | params_total | 268,098,176 | 270M parameters | definition | Rounding: his '270M' has 2 significant figure(s); ours 268,098,176 rounds to it. Same quantity, different precision. |
| google/gemma-3-27b-it | vocab_size | 262,208 | 262,144 (~262k) | definition | Ours: 262,208 embedding rows (config/weights at 005ad3404e). His: 262,144, the tokenizer.json base vocabulary at the same commit (6,415 added tokens reuse ids inside it). |
| ibm-granite/granite-4.1-30b | params_total | 28,865,728,512 | 30B parameters | definition | His figure is the one the model card states (README.md at 4fae6278f7: "[![mof-class3-qualified] # Granite-4.1-30B **Model Summary:** Granite-4.1-30B is a 30B parameter long-context instruct model finetuned from *Granite-4.1-30B-Base* using a combi"). Ours (28,865,728,512) is the exact count of every stored tensor in the safetensors headers. |
| inclusionAI/Ling-2.5-1T | context_length | 131,072 | 256,000 | definition | His figure is the one the model card states (README.md at 9c9d569b91: "----------------------------------------------------------------------: / / Ling-2.5-1T / 256K -> 1M / [🤗 HuggingFace] &nbsp;&nbsp; [🤖 ModelScope] / </center> Note: If you are int"). Ours (131,072) is the config.json max_position_embeddings (positions the released config supports). |
| inclusionAI/Ling-2.5-1T | params_active | 63,580,258,816 | 1T total, 63B active (6.3% active) | definition | His figure is the one the model card states (README.md at 9c9d569b91: "ccessible AI to a new level of quality. - Ling-2.5-1T features 1T total parameters (with 63B active parameters). Its pre-training corpus has expanded from 20T to 29T tokens compared"). Ours (63,580,258,816) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| inclusionAI/Ling-2.6-1T | params_active | 63,580,258,816 | 1T total, 63B active (6.3% active) | definition | Ours: 63,580,258,816 per token from tensor headers (at 653026b0ba), identical to Ling-2.5-1T. His 63B is Ling-2.5-1T's card figure ("1T total parameters (with 63B active parameters)", README at 9c9d569b91); Ling-2.6's README states no active count. |
| inclusionAI/Ling-3.0-flash | params_active | 5,505,693,152 | 124B total, 5.1B active (4.1% active) | definition | His figure is the one the model card states (README.md at fcc60a52b3: "h, our next-generation native hybrid reasoning model. Operating with **124B** total and **5.1B** active parameters (~12.4% and ~8.1% of our previous 1T-class flagship Ring-2.6-1T), Lin"). Ours (5,505,693,152) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| inclusionAI/Ling-3.0-flash | params_total | 127,486,405,600 | 124B total, 5.1B active (4.1% active) | definition | His figure is the one the model card states (README.md at fcc60a52b3: "ucing Ling-3.0-flash, our next-generation native hybrid reasoning model. Operating with **124B** total and **5.1B** active parameters (~12.4% and ~8.1% of our previous 1T-class flagshi"). Ours (127,486,405,600) is the exact count of every stored tensor in the safetensors headers. |
| inclusionAI/Ling-3.1-flash | context_length | 262,144 | 1M reported; 262,144 in config | definition | Ours: config.json max_position_embeddings = 262144 (at dde57b1c6a). His: the README's "1M-token context window" (he also notes 262,144 in config). Positions the released config supports vs advertised extended context. |
| inclusionAI/Ling-3.1-flash | params_total | 572,155,509,344 | 560B total, 25B active (4.5% active) | definition | His figure is the one the model card states (README.md at dde57b1c6a: "an upgraded native hybrid reasoning model designed for complex, real-world tasks. With **560B** total parameters, **25B** active parameters per token, and a **1M-token context window*"). Ours (572,155,509,344) is the exact count of every stored tensor in the safetensors headers. |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | params_total | 401,583,781,376 | 400B total, 17B active (4.3% active) | definition | Rounding: his '400B' has 1 significant figure(s); ours 401,583,781,376 rounds to it. Same quantity, different precision. |
| meta-models/Muse-Glimmer-30B | kv_cache_per_token_bf16 | 13,312 | 52 KiB | definition | Ours: growing cache per token = 13,312 B (sliding-window layers excluded: their cache is capped at the 2048-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 53,248 B, which matches his value. |
| meta-models/Muse-Glimmer-30B | params_total | 29,776,626,688 | 29.6B parameters (including perception encoder) | definition | His figure is the one the model card states (README.md at cec94fcfd6: "Causal Transformer with Perception Encoder / / :---- / :---- / / **Total Parameters** / \~29.6B / / **Language Model** / / / **Architecture** / Dense Causal Transformer / / **Num"). Ours (29,776,626,688) is the exact count of every stored tensor in the safetensors headers. |
| microsoft/phi-4 | params_total | 14,659,507,200 | 14B parameters | definition | His figure is the one the model card states (README.md at 2db69c1c3e: "ruction adherence and robust safety measures / / **Architecture** / 14B parameters, dense decoder-only Transformer model / / **Inputs**"). Ours (14,659,507,200) is the exact count of every stored tensor in the safetensors headers. |
| mistralai/Mistral-Small-4-119B-2603 | context_length | 1,048,576 | 256,000 | definition | His figure is the one the model card states (README.md at a11f36bebf: "E**: 128 experts, 4 active. - **119B parameters**, with **6.5B activated per token**. - **256k context length**. - **Multimodal input**: Accepts both text and image input, with text ou"). Ours (1,048,576) is the config.json max_position_embeddings (positions the released config supports). |
| moonshotai/Kimi-Linear-48B-A3B-Base | params_total | 49,122,681,728 | 48B total, 3B active (6.3% active) | definition | His figure is the one the model card states (README.md at 3b171c17bf: "------------------------------------------------------: / / Kimi-Linear-Base / 48B / 3B / 1M / [🤗 Hugging Face] / / Kimi-Line"). Ours (49,122,681,728) is the exact count of every stored tensor in the safetensors headers. |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | params_active | 2,485,952,844 | 30B total, 3B active (10% active) | definition | His figure is the one the model card states (README.md at bee7596271: "/> </a> </div> ![] ## Model Summary / / / /:---/:---/ / **Total Parameters** / 30B (3B active) / / **Architecture** / MoE - Mamba-2 + MoE + Attention hybrid / / **Context Lengt"). Ours (2,485,952,844) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | num_layers | 72 | 8 input + 20 shared × 2 passes + 8 output (56 applications) | definition | Ours: config.json num_hidden_layers = 72, layer_types alternate attention and MLP (n_input_layers 16 + n_recurrent_layers 40 + n_output_layers 16), weights hold layers 0-71 (at ee517dd388); one entry per operator. His: attention+MLP blocks, 8 + 20 + 8 = 36, the same 72 operators. |
| openai/gpt-oss-120b | params_active | 5,711,982,912 | 117B total, 5.1B active (4.4% active) | definition | His figure is the one the model card states (README.md at b5c939de8f: "es that fit into a single 80GB GPU (like NVIDIA H100 or AMD MI300X) (117B parameters with 5.1B active parameters) - `gpt-oss-20b` — for lower latency, and local or specialized use case"). Ours (5,711,982,912) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| openai/gpt-oss-20b | params_active | 4,187,440,704 | 21B total, 3.6B active (17.1% active) | definition | His figure is the one the model card states (README.md at 6cee5e81ee: "gpt-oss-20b` — for lower latency, and local or specialized use cases (21B parameters with 3.6B active parameters) Both models were trained on our [harmony response format] and should"). Ours (4,187,440,704) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| ornith-ai/Ornith-1.5-35B-A3B | params_total | 35,951,822,704 | 35B total, 3B active (8.6% active) | definition | His figure is the one the model card states (README.md at 10fbf86fed: "g style="width: 100%; max-width: 900px;" src="assets/ornith_35b_eval.png" alt="Ornith 1.5 35B Benchmark Results" title="Ornith 1.5 35B Benchmark Results"> ## Ornith 1.5 35B-A3B This"). Ours (35,951,822,704) is the exact count of every stored tensor in the safetensors headers. |
| poolside/Laguna-S-2.1 | kv_cache_per_token_bf16 | 49,152 | 192 KiB | definition | Ours: growing cache per token = 49,152 B (sliding-window layers excluded: their cache is capped at the 512-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 196,608 B, which matches his value. |
| poolside/Laguna-XS-2.1 | kv_cache_per_token_bf16 | 40,960 | 160 KiB | definition | Ours: growing cache per token = 40,960 B (sliding-window layers excluded: their cache is capped at the 512-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 163,840 B, which matches his value. |
| poolside/Laguna-XS.2 | kv_cache_per_token_bf16 | 40,960 | 160 KiB | definition | Ours: growing cache per token = 40,960 B (sliding-window layers excluded: their cache is capped at the 512-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 163,840 B, which matches his value. |
| reflection/Beam-1.0-501B-A23B | kv_cache_per_token_bf16 | 36,864 | 208 KiB | definition | Ours: growing cache per token = 36,864 B (sliding-window layers excluded: their cache is capped at the 512-token window). His: full-retention estimate counting every attention layer per token. Recomputed from our own published layer dims with sliding layers included: 212,992 B, which matches his value. |
| sarvamai/sarvam-105b | params_active | 12,414,902,144 | 105B total, 10.3B active (9.8% active) | definition | His figure is the one the model card states (README.md at abe264bdb8: "tion] ## Introduction **Sarvam-105B** is an advanced Mixture-of-Experts model with 10.3B active parameters, designed for superior performance across a wide range of complex tasks"). Ours (12,414,902,144) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| sarvamai/sarvam-105b | params_total | 106,031,767,424 | 105B total, 10.3B active (9.8% active) | definition | His figure is the one the model card states (README.md at abe264bdb8: "ed under the **Apache License**. For more details, see our [blog] . ## Architecture The 105B model adopts an MLA-style attention stack with decoupled QK head dimensions (`q_head_dim="). Ours (106,031,767,424) is the exact count of every stored tensor in the safetensors headers. |
| sarvamai/sarvam-30b | params_active | 4,520,575,616 | 30B total, 2.4B active (8% active) | definition | His figure is the one the model card states (README.md at 2784c524a2: "ation] ## Introduction **Sarvam-30B** is an advanced Mixture-of-Experts model with 2.4B non-embedding active parameters, designed primarily for practical deployment. It combines"). Ours (4,520,575,616) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| sarvamai/sarvam-30b | params_total | 32,152,650,368 | 30B total, 2.4B active (8% active) | definition | His figure is the one the model card states (README.md at 2784c524a2: "ed under the **Apache License**. For more details, see our [blog] . ## Architecture The 30B MoE model is designed for throughput and memory efficiency. It uses 19 layers, a dense FF"). Ours (32,152,650,368) is the exact count of every stored tensor in the safetensors headers. |
| stepfun-ai/Step-3.5-Flash | num_layers | 45 | 36 sliding-window + 12 global | definition | Ours: config.json num_hidden_layers = 45 decoder layers (at ab446a3de5). His: 36 + 12 = 48 entries of layer_types, which also covers the num_nextn_predict_layers = 3 MTP layers. |
| thinkingmachines/Inkling | params_total | 952,377,623,626 | 975B total, 41B active (4.2% active) | definition | His figure is the one the model card states (README.md at 828496eeae: "ojected into a shared hidden space and processed jointly by the decoder. ### Parameters 975B total, 41B active ### Numerics support BF16 and NVFP4 ### Input modalities Inkling ac"). Ours (952,377,623,626) is the exact count of every stored tensor in the safetensors headers. |
| thinkingmachines/Inkling | vocab_size | 201,024 | 200,058 (~200k) | definition | Ours: 201,024 embedding rows, padded (at 828496eeae). His: 200,058, the tokenizer.json size at the same commit (highest token id + 1). |
| zai-org/GLM-4.5 | params_active | 33,632,266,144 | 355B total, 32B active (9% active) | definition | Ours: 33,632,266,144 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 32,080,373,664, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-4.5 | params_total | 358,337,791,296 | 355B total, 32B active (9% active) | definition | His figure is the one the model card states (README.md at cbb2c7cfb5: "---------/ / GLM-4.5 / [🤗 Hugging Face] <br> [🤖 ModelScope] / 355B-A32B / BF16 / / GLM-4.5-Air / [🤗 Hugging Face] <br> [🤖 ModelScope]"). Ours (358,337,791,296) is the exact count of every stored tensor in the safetensors headers. |
| zai-org/GLM-4.5-Air | params_active | 13,424,129,664 | 106B total, 12B active (11.3% active) | definition | Ours: 13,424,129,664 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 12,182,615,680, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-4.5-Air | params_total | 110,468,824,832 | 106B total, 12B active (11.3% active) | definition | Ours: 110,468,824,832 (all stored tensors). His figure counts a narrower scope: ours without MTP/auxiliary heads, without input embedding and output head is 105,610,737,280, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-4.7 | params_active | 33,632,266,144 | 355B total, 32B active (9% active) | definition | Ours: 33,632,266,144 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 32,080,373,664, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-4.7 | params_total | 358,337,791,296 | 355B total, 32B active (9% active) | definition | Ours: 358,337,791,296, every stored tensor (at 602d01efcd), identical to GLM-4.5. His 355B is GLM-4.5's card figure ("355B-A32B", README at cbb2c7cfb5); GLM-4.7's README states no total. |
| zai-org/GLM-5 | params_active | 41,784,728,832 | 744B total, 40B active (5.4% active) | definition | His figure is the one the model card states (README.md at c183ef8c61: ". Compared to GLM-4.5, GLM-5 scales from 355B parameters (32B active) to 744B parameters (40B active), and increases pre-training data from 23T to 28.5T tokens. GLM-5 also integrates"). Ours (41,784,728,832) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| zai-org/GLM-5 | params_total | 753,864,139,008 | 744B total, 40B active (5.4% active) | definition | His figure is the one the model card states (README.md at c183ef8c61: "al Intelligence . Compared to GLM-4.5, GLM-5 scales from 355B parameters (32B active) to 744B parameters (40B active), and increases pre-training data from 23T to 28.5T tokens. GLM-5"). Ours (753,864,139,008) is the exact count of every stored tensor in the safetensors headers. |
| zai-org/GLM-5.1 | params_active | 41,784,728,832 | 744B total, 40B active (5.4% active) | definition | Ours: 41,784,728,832 (per-token, all non-expert weights). His figure counts a narrower scope: ours without input embedding and output head is 39,881,563,392, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-5.1 | params_total | 753,864,139,008 | 744B total, 40B active (5.4% active) | definition | Ours: 753,864,139,008 (all stored tensors). His figure counts a narrower scope: ours without MTP/auxiliary heads is 743,911,218,432, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-5.2 | params_active | 41,250,530,304 | 744B total, 40B active (5.4% active) | definition | Ours: 41,250,530,304 (per-token, all non-expert weights). His figure counts a narrower scope: ours without the input embedding is 40,298,947,584, which matches his value at his rounding. Component sizes from our tensor-header counts. |
| zai-org/GLM-5.2 | params_total | 753,329,940,480 | 744B total, 40B active (5.4% active) | definition | Ours: 753,329,940,480, every stored tensor (at cf457fa734). His 744B is GLM-5's card figure ("744B parameters (40B active)", README at c183ef8c61); GLM-5.2's README states no total. |
| zai-org/GLM-5.3-Flash | params_active | 17,377,397,566 | 320B total, 18B active (5.6% active) | definition | His figure is the one the model card states (README.md at 46a050bec1: "first natively multimodal model in the GLM-5 series. With 320B total parameters and just 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at o"). Ours (17,377,397,566) is the per-token count from tensor headers (all non-expert weights + selected experts). |
| zai-org/GLM-5.3-Flash | params_total | 321,342,220,638 | 320B total, 18B active (5.6% active) | definition | Rounding: his '320B' has 2 significant figure(s); ours 321,342,220,638 rounds to it. Same quantity, different precision. |

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
| - | meta-models/Muse-Spark-Open-816B-A42B | access_needed | HTTPError: HTTP Error 403: Forbidden |
| - | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | pipeline_error | KeyError: 'num_hidden_layers' |
| - | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 | pipeline_error | ValueError: Invalid Nemotron-H block pattern |
| - | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | pipeline_error | ValueError: Invalid Nemotron-H block pattern |
| - | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 | pipeline_error | KeyError: 'num_hidden_layers' |
| - | Soofi-Project/Soofi-S-Base | access_needed | HTTPError: HTTP Error 403: Forbidden |
| - | CohereLabs/tiny-aya-base | access_needed | HTTPError: HTTP Error 403: Forbidden |
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
