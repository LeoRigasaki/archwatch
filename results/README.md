# Saved prototype results

Generated from 26 saved records. This command does not refetch sources. The ten-publisher backfill run is recorded in [backfill-processing.json](backfill-processing.json); original-prototype evidence remains in [live-validation.json](live-validation.json).

| Model | Parameters in saved header counts | Verified fields | Single-source fields | Conflicts |
|---|---:|---:|---:|---:|
| LiquidAI/LFM2.5-1.2B-Instruct | 1,170,340,608 | 6 | 14 | 1 |
| LiquidAI/LFM2.5-24B-A2B-Base | 23,971,587,712 | 10 | 17 | 1 |
| LiquidAI/LFM3-1.7B-Base | 1,728,801,536 | 12 | 16 | 0 |
| LiquidAI/LFM3-1.7B | 1,728,801,536 | 11 | 14 | 0 |
| Qwen/Qwen3-0.6B-Base | 596,049,920 | 6 | 16 | 0 |
| Qwen/Qwen3.8-2.4T-A95B | 2,446,182,725,504 | 8 | 27 | 1 |
| Qwen/Qwen3.8-Flash-Next | 179,999,981,424 | 9 | 29 | 0 |
| deepseek-ai/DeepSeek-V2-Lite | 15,706,484,224 | 8 | 23 | 0 |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 763,205,315,794 | 7 | 26 | 0 |
| deepseek-ai/DeepSeek-V4.1-Flash | 763,205,315,794 | 7 | 26 | 0 |
| google/diffusiongemma-26B-A4B-it | 25,823,781,228 | 8 | 30 | 1 |
| google/gemma-3-1b-pt | 999,885,952 | 5 | 19 | 0 |
| google/gemma-4-26B-A4B | 25,805,936,206 | 9 | 27 | 1 |
| meta-llama/Llama-3.1-8B | 8,030,261,248 | 5 | 16 | 0 |
| meta-llama/Llama-4-Maverick-17B-128E | 401,583,781,376 | 7 | 24 | 0 |
| microsoft/Mage-VL | 5,278,537,216 | 5 | 21 | 0 |
| microsoft/phi-2 | 2,779,683,840 | 6 | 16 | 0 |
| mistralai/Mistral-7B-v0.1 | 7,241,732,096 | 5 | 17 | 0 |
| mistralai/Mistral-Large-4-1T-A52B | 1,053,161,553,184 | 6 | 28 | 1 |
| moonshotai/Kimi-K3.1-Pro | 2,780,110,363,392 | 24 | 24 | 0 |
| moonshotai/Kimi-K3 | 2,779,931,837,184 | 20 | 24 | 0 |
| nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 | 3,975,229,440 | 6 | 24 | 0 |
| nvidia/Nemotron-H-8B-Base-8K | 8,100,852,736 | 6 | 22 | 0 |
| zai-org/GLM-4-9B-0414 | 9,400,279,040 | 5 | 17 | 0 |
| zai-org/GLM-5.2 | 753,329,940,480 | 7 | 26 | 1 |
| zai-org/GLM-5.3-Flash-BF16 | 321,323,031,390 | 8 | 31 | 0 |

Offline tests: 114 passed; 0 failures; 0 errors. Python 3.12.13.

Tensor-group additions: LiquidAI/LFM3-1.7B-Base: 46,598,912; LiquidAI/LFM3-1.7B: 46,598,912; Qwen/Qwen3.8-Flash-Next: 176,791,650,672; deepseek-ai/DeepSeek-V4.1-Flash-1017: 0; google/diffusiongemma-26B-A4B-it: 25,823,781,228; mistralai/Mistral-Large-4-1T-A52B: 6,874,961,696; moonshotai/Kimi-K3.1-Pro: 178,526,208; zai-org/GLM-5.3-Flash-BF16: 320,688,642,910.

Manual primary-source gold labels: 10. [Per-field exact match](gold-exact-match.md) reports the development set; it is not human-reviewed or held-out accuracy. The longer-term 30-model evaluation remains open.
