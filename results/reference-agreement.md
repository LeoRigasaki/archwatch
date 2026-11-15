# Agreement with an outside reference

Reference: [rasbt/llm-architecture-gallery](https://github.com/rasbt/llm-architecture-gallery) `models.yml` at commit `660cc1886a` (Apache-2.0), by Sebastian Raschka. **Comparison only**: it is never a source for ArchWatch's published fields, gold labels or diagrams, and its images are not used.

Ours: published, verifier-approved record fields (KV cache recomputed from published values, bf16, growing part per token). Theirs: fact-sheet strings, compared within their displayed rounding.

## Per-field agreement (models in both sets)

| Field | Agree | Compared |
|---|---:|---:|
| params_total | 7 | 9 |
| params_active | 3 | 6 |
| context_length | 9 | 9 |
| vocab_size | 6 | 6 |
| num_layers | 8 | 9 |
| growing cache per token (sliding-window and constant states excluded) | 3 | 4 |
| FFN sparsity (dense / MoE) | 9 | 9 |

Overall: 45 of 52 compared values agree across 9 models.

## Disagreements

Causes: definition difference (recomputed) 4, real error or unreconciled (ArchWatch) 3. A definition difference is accepted only when ArchWatch's own number, under one fixed rule or an exclusion the model card states, lands within 2% or the reference's own rounding (rule in each row).

| Model | Field | Ours | Theirs | Cause | Why | Our evidence |
|---|---|---|---|---|---|---|
| LiquidAI/LFM3-1.7B | growing cache per token (sliding-window and constant states excluded) | 12,288 | 22 KiB | **definition difference (recomputed)** | Rule: default: full retention, every attention layer appends its KV per token (sliding-window layers not capped). Ours recomputed: 22,528. His: 22 KiB (accepted 22,016-23,040); gap after the rule 0.00%. | 2 x 4 KV heads x 128 head_dim x 2 B x 6 layers |
| Qwen/Qwen3.8-Flash-Next | params_active | 7,306,994,560 | 125B core, 6B active (4.8% active); plus 51B n-gram embeddings and 4B MTP | **real error or unreconciled (ArchWatch)** | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 7,306,994,560; his accepted range 5,500,000,000-6,500,000,000. | single source: weights:total - unselected routed experts (text model, all non-expert weights included) |
| deepseek-ai/DeepSeek-V4.1-Flash | params_total | 763,205,315,794 | 552B backbone + 196B Engram; 8B active prefill, 16B active decode (2.9% of backbone) | **definition difference (recomputed)** | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 762,793,473,234. His: 552B backbone + 196B Engram (accepted 733,040,000,000-762,960,000,000); gap after the rule 1.98%. | single source: weights:sum of tensor shapes |
| deepseek-ai/DeepSeek-V4.1-Flash | params_active | 16,865,928,304 | 552B backbone + 196B Engram; 8B active prefill, 16B active decode (2.9% of backbone) | **real error or unreconciled (ArchWatch)** | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 16,865,928,304; his accepted range 7,840,000,000-16,320,000,000. | single source: weights:total - unselected routed experts (text model, all non-expert weights included) |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | num_layers | 72 | 8 input + 20 shared × 2 passes + 8 output (56 applications) | **definition difference (recomputed)** | Rule: attention+MLP blocks: config num_hidden_layers (72 operator entries) / 2. Ours recomputed: 36. His: 8 input + 20 shared × 2 passes + 8 output (56 applications) (accepted 35-37). Ours: config.json num_hidden_layers = 72, layer_types alternate attention and MLP (n_input_layers 16 + n_recurrent_layers 40 + n_output_layers 16), weights hold layers 0-71 (at ee517dd388); one entry per operator. His: attention+MLP blocks, 8 + 20 + 8 = 36, the same 72 operators. | verified: config:num_hidden_layers; weights:highest layer index + 1 |
| zai-org/GLM-5.2 | params_total | 753,329,940,480 | 744B total, 40B active (5.4% active) | **definition difference (recomputed)** | Rule: default: all stored tensors minus vision tower and projector (text-only figure). Ours recomputed: 753,329,940,480. His: 744B total (accepted 729,120,000,000-758,880,000,000); gap after the rule 1.25%. | single source: weights:sum of tensor shapes |
| zai-org/GLM-5.2 | params_active | 41,250,530,304 | 744B total, 40B active (5.4% active) | **real error or unreconciled (ArchWatch)** | Closest recomputation: default: per-token count (unselected experts, MTP heads and lookup tables not active) = 41,250,530,304; his accepted range 39,200,000,000-40,800,000,000. | single source: weights:total - unselected routed experts (text model, all non-expert weights included) |

A disagreement is a prompt to re-check sources, not a correction: ArchWatch publishes what the model's own files state.

## Gallery models ArchWatch has not processed (110)

| Date | Model | Company | Hub repo |
|---|---|---|---|
| 2026-11-05 | ERNIE 5.5 Flash Base | Baidu | baidu/ERNIE-5.5-Flash-Base |
| 2026-10-19 | MiniMax M3.1-Flash 428B | MiniMax | MiniMaxAI/MiniMax-M3.1-Flash |
| 2026-10-18 | Recursive Flint 40B-A2.5B | Recursive | RecursiveSI/flint |
| 2026-10-16 | Reflection Beam 501B | Reflection | reflection/Beam-1.0-501B-A23B |
| 2026-10-15 | Muse Spark Open 816B-A42B | Meta | meta-models/Muse-Spark-Open-816B-A42B |
| 2026-10-15 | Nemotron 3.5 Super VL 120B-A12B | NVIDIA | nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 |
| 2026-10-13 | Ling 3.1 Flash | InclusionAI | inclusionAI/Ling-3.1-flash |
| 2026-10-03 | Kolibri-1 78B | Aleph Alpha | Aleph-Alpha/Kolibri-1 |
| 2026-09-27 | Naive-N0.5-Flash 309B | NaiveAI | NaiveAI/Naive-N0.5-Flash |
| 2026-09-21 | Xiaomi MiMo-V2.6-Flash-RL 309B | Xiaomi | XiaomiMiMo/MiMo-V2.6-Flash-RL |
| 2026-09-21 | Xiaomi MiMo-V2.6-Pro-RL 1.02T | Xiaomi | XiaomiMiMo/MiMo-V2.6-Pro-RL |
| 2026-09-20 | Step-5 Preview 600B | StepFun | SHSLab/Step-5-Preview-BF16 |
| 2026-08-28 | Tencent Hy4-preview 770B-A49B | Tencent | tencent/Hy4-preview |
| 2026-08-26 | GLM-5.3-Flash | Z.ai | zai-org/GLM-5.3-Flash |
| 2026-08-23 | Ornith 1.5 35B-A3B | Ornith | ornith-ai/Ornith-1.5-35B-A3B |
| 2026-08-14 | Qwen3.8 27B | Alibaba Cloud | Qwen/Qwen3.8-27B |
| 2026-08-11 | Nemotron 3.5 Lightning 30B-A3B | NVIDIA | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 |
| 2026-08-10 | Muse Glimmer 30B | Meta | meta-models/Muse-Glimmer-30B |
| 2026-08-04 | Ling 3.0 Flash | InclusionAI | inclusionAI/Ling-3.0-flash |
| 2026-07-22 | Solar Open 2 | Upstage | upstage/Solar-Open2-250B |
| 2026-07-21 | Antares 1B | Cisco | fdtn-ai/antares-1b |
| 2026-07-21 | Laguna S 2.1 | Poolside | poolside/Laguna-S-2.1 |
| 2026-07-21 | Nanbeige 4.2 3B | BOSS Zhipin | Nanbeige/Nanbeige4.2-3B |
| 2026-07-21 | Motif 3 Beta | Motif Technologies | Motif-Technologies/Motif-3-Beta |
| 2026-07-20 | BTL-3 27B | Bad Theory Labs | badtheorylabs/BTL-3 |
| 2026-07-15 | Inkling | Thinking Machines Lab | thinkingmachines/Inkling |
| 2026-07-10 | Soofi S 30B-A3B | SOOFI Consortium | Soofi-Project/Soofi-S-Base |
| 2026-07-02 | Laguna XS 2.1 | Poolside | poolside/Laguna-XS-2.1 |
| 2026-06-15 | VibeThinker-3B | Weibo | WeiboAI/VibeThinker-3B |
| 2026-06-13 | MiniMax M3 428B | MiniMax | MiniMaxAI/MiniMax-M3 |
| 2026-06-12 | Kimi K2.7 Code | Moonshot AI | moonshotai/Kimi-K2.7-Code |
| 2026-06-05 | North Mini Code 30B-A3B | Cohere | CohereLabs/North-Mini-Code-1.0 |
| 2026-06-04 | Nemotron 3 Ultra 550B-A55B | NVIDIA | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 |
| 2026-06-03 | Gemma 4 12B | Google | google/gemma-4-12B-it |
| 2026-06-01 | JetBrains Mellum2 Thinking 12B-A2.5B | JetBrains | JetBrains/Mellum2-12B-A2.5B-Thinking |
| 2026-05-28 | LFM2.5 350M | Liquid AI | LiquidAI/LFM2.5-350M |
| 2026-05-28 | LFM2.5 1.2B | Liquid AI | LiquidAI/LFM2.5-1.2B-Thinking |
| 2026-05-28 | LFM2.5 8B-A1B | Liquid AI | LiquidAI/LFM2.5-8B-A1B |
| 2026-05-20 | Command A+ 218B-A25B | Cohere | CohereLabs/command-a-plus-05-2026-w4a4 |
| 2026-05-06 | ZAYA1-8B | Zyphra | Zyphra/ZAYA1-8B |
| 2026-04-29 | Granite 4.1 30B | IBM | ibm-granite/granite-4.1-30b |
| 2026-04-28 | Laguna XS.2 | Poolside | poolside/Laguna-XS.2 |
| 2026-04-24 | DeepSeek V4-Pro | DeepSeek | deepseek-ai/DeepSeek-V4-Pro |
| 2026-04-24 | DeepSeek V4-Flash | DeepSeek | deepseek-ai/DeepSeek-V4-Flash |
| 2026-04-23 | Tencent Hy3-preview 295B-A21B | Tencent | tencent/Hy3-preview |
| 2026-04-23 | Ling 2.6 1T | InclusionAI | inclusionAI/Ling-2.6-1T |
| 2026-04-22 | Qwen3.6 27B | Alibaba Cloud | Qwen/Qwen3.6-27B |
| 2026-04-22 | Xiaomi MiMo-V2.5 310B | Xiaomi | XiaomiMiMo/MiMo-V2.5 |
| 2026-04-22 | Xiaomi MiMo-V2.5-Pro 1.02T | Xiaomi | XiaomiMiMo/MiMo-V2.5-Pro |
| 2026-04-20 | Kimi K2.6 | Moonshot AI | moonshotai/Kimi-K2.6 |
| 2026-04-15 | Qwen3.6 35B-A3B | Alibaba Cloud | Qwen/Qwen3.6-35B-A3B |
| 2026-04-07 | GLM-5.1 | Z.ai | zai-org/GLM-5.1 |
| 2026-04-02 | Gemma 4 26B-A4B | Google | google/gemma-4-26B-A4B-it |
| 2026-04-02 | Gemma 4 31B | Google | google/gemma-4-31B-it |
| 2026-04-02 | Gemma 4 E2B | Google | google/gemma-4-E2B-it |
| 2026-04-02 | Gemma 4 E4B | Google | google/gemma-4-E4B-it |
| 2026-03-18 | MiniMax M2.7 230B | MiniMax | MiniMaxAI/MiniMax-M2.7 |
| 2026-03-16 | Mistral Small 4 | Mistral AI | mistralai/Mistral-Small-4-119B-2603 |
| 2026-03-16 | Nemotron 3 Nano 4B | NVIDIA | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 |
| 2026-03-11 | Nemotron 3 Super 120B-A12B | NVIDIA | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 |
| 2026-03-03 | Sarvam 105B | Sarvam AI | sarvamai/sarvam-105b |
| 2026-03-03 | Sarvam 30B | Sarvam AI | sarvamai/sarvam-30b |
| 2026-02-16 | Qwen3.5 397B | Alibaba Cloud | Qwen/Qwen3.5-397B-A17B |
| 2026-02-15 | Ling 2.5 1T | InclusionAI | inclusionAI/Ling-2.5-1T |
| 2026-02-13 | Tiny Aya 3.35B | Cohere | CohereLabs/tiny-aya-base |
| 2026-02-12 | MiniMax M2.5 230B | MiniMax | MiniMaxAI/MiniMax-M2.5 |
| 2026-02-11 | GLM-5 744B | Z.ai | zai-org/GLM-5 |
| 2026-02-10 | Nanbeige 4.1 3B | BOSS Zhipin | Nanbeige/Nanbeige4.1-3B |
| 2026-02-01 | Step 3.5 Flash 196B | StepFun | stepfun-ai/Step-3.5-Flash |
| 2026-01-28 | LongCat-Flash-Lite 68.5B-A3B | Meituan | meituan-longcat/LongCat-Flash-Lite |
| 2026-01-27 | Arcee AI Trinity Large 400B | Arcee AI | arcee-ai/Trinity-Large-Base |
| 2026-01-27 | Kimi K2.5 | Moonshot AI | moonshotai/Kimi-K2.5 |
| 2025-12-22 | GLM-4.7 355B | Z.ai | zai-org/GLM-4.7 |
| 2025-12-16 | Xiaomi MiMo-V2-Flash 309B | Xiaomi | XiaomiMiMo/MiMo-V2-Flash |
| 2025-12-04 | Nemotron 3 Nano 30B-A3B | NVIDIA | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 |
| 2025-12-02 | Mistral Large 3 | Mistral AI | mistralai/Mistral-Large-3-675B-Instruct-2512 |
| 2025-12-01 | DeepSeek V3.2 | DeepSeek | deepseek-ai/DeepSeek-V3.2 |
| 2025-11-26 | INTELLECT-3 | Prime Intellect | PrimeIntellect/INTELLECT-3 |
| 2025-11-20 | OLMo 3 32B | Ai2 | allenai/Olmo-3-32B-Think |
| 2025-11-20 | OLMo 3 7B | Ai2 | allenai/Olmo-3-1025-7B |
| 2025-10-30 | Kimi Linear 48B-A3B | Moonshot AI | moonshotai/Kimi-Linear-48B-A3B-Base |
| 2025-10-29 | Ouro-Thinking 2.6B | ByteDance | ByteDance/Ouro-2.6B-Thinking |
| 2025-10-23 | MiniMax M2 230B | MiniMax | MiniMaxAI/MiniMax-M2 |
| 2025-09-09 | Qwen3 Next 80B-A3B | Alibaba Cloud | Qwen/Qwen3-Next-80B-A3B-Instruct |
| 2025-08-22 | Grok 2.5 270B | xAI | xai-org/grok-2 |
| 2025-08-14 | Gemma 3 270M | Google | google/gemma-3-270m |
| 2025-08-04 | GPT-OSS 120B | OpenAI | openai/gpt-oss-120b |
| 2025-08-04 | GPT-OSS 20B | OpenAI | openai/gpt-oss-20b |
| 2025-07-31 | Qwen3 Coder Flash 30B-A3B | Alibaba Cloud | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| 2025-07-28 | GLM-4.5 355B | Z.ai | zai-org/GLM-4.5 |
| 2025-07-28 | GLM-4.5-Air | Z.ai | zai-org/GLM-4.5-Air |
| 2025-07-10 | Kimi K2 | Moonshot AI | moonshotai/Kimi-K2-Base |
| 2025-06-19 | SmolLM3 3B | Hugging Face | HuggingFaceTB/SmolLM3-3B-Base |
| 2025-04-28 | Qwen3 235B-A22B | Alibaba Cloud | Qwen/Qwen3-235B-A22B |
| 2025-04-28 | Qwen3 30B-A3B | Alibaba Cloud | Qwen/Qwen3-30B-A3B |
| 2025-04-28 | Qwen3 32B | Alibaba Cloud | Qwen/Qwen3-32B |
| 2025-04-28 | Qwen3 4B | Alibaba Cloud | Qwen/Qwen3-4B |
| 2025-04-28 | Qwen3 8B | Alibaba Cloud | Qwen/Qwen3-8B |
| 2025-04-05 | Llama 4 Maverick | Meta | meta-llama/Llama-4-Maverick-17B-128E-Instruct |
| 2025-03-18 | Mistral Small 3.1 24B | Mistral AI | mistralai/Mistral-Small-3.1-24B-Base-2503 |
| 2025-03-17 | xLSTM 7B | NXAI | NX-AI/xLSTM-7b |
| 2025-03-11 | Gemma 3 27B | Google | google/gemma-3-27b-it |
| 2025-01-20 | DeepSeek R1 | DeepSeek | deepseek-ai/DeepSeek-R1 |
| 2024-12-26 | DeepSeek V3 | DeepSeek | deepseek-ai/DeepSeek-V3 |
| 2024-12-12 | Phi-4 | Microsoft | microsoft/phi-4 |
| 2024-11-25 | OLMo 2 7B | Ai2 | allenai/OLMo-2-1124-7B-Instruct |
| 2024-09-25 | Llama 3.2 1B | Meta | (none listed) |
| 2024-09-25 | Llama 3.2 3B | Meta | (none listed) |
| 2024-04-18 | Llama 3 8B | Meta | meta-llama/Meta-Llama-3-8B |
| 2019-11-05 | GPT-2 XL 1.5B | OpenAI | openai-community/gpt2-xl |
