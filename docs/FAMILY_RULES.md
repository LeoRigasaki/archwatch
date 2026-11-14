# Backfill family rules and evidence

The ten-model selection is in `config/backfill.yaml`; exact revisions and primary links are in `evals/gold/`. Downloaded modeling code is inspected with AST/text only, never imported or executed. These rules support the selected releases and do not promise arbitrary members of each family.

| Family | Rule added or confirmed | Primary basis |
|---|---|---|
| DeepSeek V2 | MLA dimensions; expert `gate_proj` naming; one dense prefix layer | Pinned config and expert/header shapes in its gold label |
| Qwen3 / Llama3 | Explicit head dimension; separate gate/up/down FFN tensors; nested RoPE scaling preserved | Pinned configs and tensor headers |
| Mistral | Non-null sliding window without an explicit pattern means sliding attention in every Mistral layer | [Transformers v4.34 implementation](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py), config window 4096 |
| LFM3 | Static SiLU FFN claim; short-conv/sliding/global pattern and sandwich norms | [Pinned Lfm3MLP forward](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/modeling_lfm3.py#L157), config and tensor names |
| Nemotron-H | `M`=Mamba-2, `-`=standalone ungated MLP, `*`=attention; 52 blocks, not 52 attention-plus-FFN pairs | [Pinned block selection](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/94ea861e008c2dfced3e8e1302094024077aa04e/modeling_nemotron_h.py#L748), `backbone` and `mixer` header names |
| Gemma3 | Every sixth layer global; 22 sliding + 4 global; pre/post mixer and FFN norms | [Transformers v4.50 implementation](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/gemma3/modeling_gemma3.py), pinned config and norm tensors |
| Phi2 | Ungated `fc1/fc2` FFN; parallel attention/MLP sharing the pre-norm input | [Transformers v4.37 PhiDecoderLayer](https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/phi/modeling_phi.py), pinned config/headers |
| Kimi | Existing one-based MLA/DSA/KDA and latent MoE rules; packed `w3` provides gate evidence | Pinned custom code, config and packed expert shapes |
| GLM4 | Fused `gate_up_proj` has twice the FFN width; post-MLP norm tensors | Pinned 9B config and header shapes, not larger-family card dimensions |

`tests/test_family_rules.py` exercises these naming/layout rules, invalid patterns, hand-calculated Mamba state sizes, portable RoPE serialization and diagram topology. Standard attention tensors cannot independently distinguish global from sliding attention; those labels rely on config and the referenced implementation rules. The former fabricated RMSNorm default has been removed: generic diagram labels say Norm without asserting a specific norm class.

## Mamba-2 cache boundary

For the selected Nemotron-H mixer, convolution channels are `128 heads × 64 head_dim + 2 × 8 groups × 128 state_size = 10,240`, cross-checked with `conv1d.weight`. Each Mamba block has a mathematical recurrent state of `128 × 64 × 128` elements plus `10,240 × 4` convolution elements. The estimate uses FP32 recurrent state and BF16 convolution buffers; a lower bound uses BF16 recurrent state. Standalone MLP blocks add no persistent decoding cache.

This is a state-size estimate, **not runtime qualification or total VRAM**. The pinned repository includes a cache helper whose allocation differs from the mixer's convolution dimensions; it was not executed. Kernel implementations, cache helpers, dtypes, workspace and batch behavior require separate runtime validation. This observation is not presented as a proven upstream defect.

## Review boundaries

The historical Liquid Instruct records carry explicit Hub fine-tune relations and are held by the watcher/site. The new Liquid gold/backfill target is the primary Base release. The old LFM2.5 context claim disagreement remains visible in its diagnostic record; it is not silently resolved or counted among the ten backfill successes. Predecessors are selected only from the inspected same-org/family catalog; sparse history yields an explicit unresolved selection.
