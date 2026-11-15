# One-time reprocess for tensor shapes (2026-11-15)

`watch.py --repo <id> --process --reprocess` on the eight records analyzed before `block_shapes` existed.
Deterministic: pinned source commits (all unchanged), safetensors headers only, no LLM calls.

| Record | Outcome | Published fields changed (besides the new `block_shapes`) |
|---|---|---|
| Qwen/Qwen3-0.6B-Base | ready_for_diagram | `model_type` newly published; `weight_header_summary` gains `skipped_buffer_tensors: 0` |
| deepseek-ai/DeepSeek-V2-Lite | ready_for_diagram | `model_type`; `moe_shared_expert_intermediate` newly published (2816); header summary counter |
| google/gemma-3-1b-pt | ready_for_diagram | `model_type`; header summary counter |
| microsoft/Mage-VL | ready_for_diagram | none |
| microsoft/phi-2 | ready_for_diagram | `model_type`; header summary counter |
| mistralai/Mistral-7B-v0.1 | ready_for_diagram | `model_type`; header summary counter |
| zai-org/GLM-4-9B-0414 | ready_for_diagram | `model_type`; header summary counter |
| LiquidAI/LFM3-1.7B-Base | **held for review**, record not rewritten | none (record unchanged) |

No previously published architecture value changed. New values come from rules added after these records were
first analyzed.

**LFM3-1.7B-Base hold.** Reprocessing re-checks the selected predecessor, LiquidAI/LFM2.5-24B-A2B-Base, against
the publication gate. That MoE record publishes no shared-expert count, and the gate requires
`moe_shared_experts` for every MoE model, so the successor is held ("Required published inputs are unavailable:
moe_shared_experts"). Open question: should the gate accept an MoE model whose sources state no shared experts?

**Diagram queue after this.** `tools/diagram_queue.py --plan` now queues Qwen3-0.6B-Base, Llama-3.1-8B, Mage-VL,
Mistral-7B-v0.1 and Kimi-K3 (5-model cap), with the remaining ready records waiting for the next slot. Three
specs are not buildable yet and cost no calls: DeepSeek-V2-Lite (no MLA query latent), gemma-3-1b-pt (tied LM
head not published as such) and phi-2 (attention output projection is named `dense`). These are spec-builder
rules to add, not data problems.
