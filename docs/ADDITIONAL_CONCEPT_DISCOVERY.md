# Additional concept discovery — 17 November 2026

Checked the 24 saved primary records, their publication decisions, and the wider cache (159 repositories with cached current-revision configs). Cache entries include derivatives and conversions; this is a discovery pool, not a count of admitted models. The existing 13 published models and 13 held repositories are unchanged.

Found **30 additional candidate entries** beyond the supplied 28-name inventory: **16 with signals in published records, five represented only in held records, and nine raw primary-source leads**. These include subtypes and structural properties; they are not 30 independent new algorithms. Mamba-2 already has an ArchWatch guide; MQA and KDA are partly covered by broader guides. No website, model record, extractor, diagram or frozen evaluation file was changed.

Every example and source revision is recorded in `results/additional-concept-discovery.json`. A published signal can be single-source; derived names such as SwiGLU/GEGLU still need operation-level qualification.

## Signals already in published records

| Candidate | Example | Evidence and limits |
|---|---|---|
| Multi-query attention (MQA) | [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/tree/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29) (published) | `attention_type`, `num_attention_heads`, `num_kv_heads`. A subtype of grouped-query attention with one KV head. Already explained in the GQA guide, but deserves an explicit subtype in model evidence. |
| Mamba-2 / structured state-space duality | [nvidia/Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/tree/94ea861e008c2dfced3e8e1302094024077aa04e) (published) | `layer_type_counts`, `mamba_state_size`, `mamba_head_dim`. Already has an ArchWatch guide; absent from the supplied 28-name inventory. Keep Mamba state separate from attention KV cache. [Primary reference](https://arxiv.org/abs/2405.21060). |
| Kimi Delta Attention (KDA) | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e) (published) | `layer_type_counts`, `linear_attn_num_heads`, `linear_attn_conv_kernel`. A named linear-attention subtype, not a synonym for all hybrid attention. Already represented in diagrams; currently explained under the broader linear-attention guide. [Primary reference](https://arxiv.org/abs/2510.26692). |
| SwiGLU / SiLU-gated feed-forward networks | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) (published) | `activation`, `ffn_gated`. Derived candidate from the activation and gated tensor layout. Verify multiplication order and active implementation before giving a canonical operation label. [Primary reference](https://arxiv.org/abs/2002.05202). |
| GEGLU-family / GELU-gated feed-forward networks | [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt/tree/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29) (published) | `activation`, `ffn_gated`. GELU-family activation plus gated FFN layout is a useful derived lead; retain the tanh approximation variant and verify implementation. [Primary reference](https://arxiv.org/abs/2002.05202). |
| SiTU-GLU | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e) (published) | `activation`, `activation_card_name`, `ffn_gated`. The card names SiTU-GLU. This must not be relabelled SiLU/SwiGLU. |
| GELU activation variants | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2/tree/810d367871c1d460086d9f82db8696f2e0a0fcd0) (published) | `activation`, `ffn_gated`. Phi uses gelu_new while Gemma uses gelu_pytorch_tanh. Activation family, approximation and FFN gating are separate properties. |
| Squared ReLU (ReLU2) | [nvidia/Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/tree/94ea861e008c2dfced3e8e1302094024077aa04e) (published) | `activation`, `mamba_activation`. The FFN says relu2 while the Mamba component says silu. Scope the tag to the component, not every activation in the model. |
| Sandwich normalization | [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/tree/7bca5038823c0d0f14efe06439109354a10e014c) (published) | `sandwich_norm`. Normalization placement is already represented. It does not establish the normalization equation or imply RMSNorm by itself. |
| Parallel attention and MLP residual block | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2/tree/810d367871c1d460086d9f82db8696f2e0a0fcd0) (published) | `block_layout`. Already represented as parallel. It is different from serial decoder sublayers and from running layers on parallel hardware. |
| Tied input/output embeddings | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd) (published) | `tied_embeddings`. A storage/parameter-sharing property supported by existing checks; distinguish it from sharing embeddings between MTP and decoder modules. |
| Shared MoE experts | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/604d5664dddd88a0433dbae533b7fe9472482de0) (published) | `moe_shared_experts`, `moe_shared_expert_intermediate`. A useful subtype of MoE. Shared experts are separate from top-k routed experts and latent expert projections. |
| Sigmoid expert routing | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/tree/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e) (published) | `moe_router`. The router scoring function is not the attention gate or FFN activation. Normalization/top-k order still needs separate evidence. |
| Partial rotary embeddings | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2/tree/810d367871c1d460086d9f82db8696f2e0a0fcd0) (published) | `rope`. Record the fraction and attention path. Do not assume a local path uses the same fraction as a global path. |
| YaRN context extension | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/tree/604d5664dddd88a0433dbae533b7fe9472482de0) (published) | `rope`. Existing RoPE metadata identifies the extension type and settings; this is not measured long-context quality. [Primary reference](https://arxiv.org/abs/2309.00071). |
| Llama-3 frequency-dependent RoPE scaling | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B/tree/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b) (published) | `rope`. Existing scaling metadata has the llama3 type and frequency factors. Treat it as a RoPE subtype, not another attention operator. |

## Signals in held records only

| Candidate | Example | Evidence and limits |
|---|---|---|
| Gated DeltaNet | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa) (held) | `layer_type_counts`, `linear_num_key_heads`, `linear_num_value_heads`. Explicitly named in the pinned card and represented in a held architecture. Do not infer it from every linear_attention string. [Primary reference](https://arxiv.org/abs/2412.06464). |
| Engram / hashed n-gram conditional memory | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518) (held) | `ngram_layer_ids`. Layer IDs are extracted; memory addressing and module semantics still need qualification. Distinct from PLE and from no_repeat_ngram generation settings. [Primary reference](https://arxiv.org/abs/2601.07372). |
| Block/discrete text diffusion | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it/tree/f7f5b7f5fa82ffc52addd066915886d497f5517b) (held) | `generation_type`, `canvas_length`. A different generation scheme, not a second attention type. Decoder/cache rules remain held. [Primary reference](https://ai.google.dev/gemma/docs/diffusiongemma). |
| Interleaved multimodal RoPE (M-RoPE) | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/812d96e10a0dd82b69d749e68693698cf6c36daa) (held) | `rope`. The metadata contains mrope_interleaved and section widths. Verify spatial/temporal coordinate semantics in code before a multimodal graph. |
| Different local/global attention head geometry | [google/gemma-4-26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B/tree/24548b62aa021d562695c04aaf7758a1ea47990b) (held) | `head_dim`, `global_head_dim`, `num_kv_heads`, `num_global_kv_heads`. A structural property affecting cache formulas, nested under the local/global attention concepts rather than counted as a new attention algorithm. |

## Raw primary-source leads — extraction/semantics still needed

| Candidate | Example | Evidence and limits |
|---|---|---|
| Vision patch merging / visual token compression | [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) (published) | `vision_config.mm_projector_type`, `vision_config.merge_kernel_size`, `vision_config.merge_type`. Source names patchmergerv2 and a merge kernel. Extract and validate projector/token transformations; current decoder diagrams do not qualify this path. |
| Learned attention sinks | [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/8b60b288e8325b7890fcdfe32bbb1d172c591a2d/config.json) (cache_only) | `add_swa_attention_sink_bias`, `add_full_attention_sink_bias`. Pinned modeling_mimo_v2_flash.py lines 113 and 335-337 show sink use/creation. Learned softmax-denominator sink biases are different from retaining initial cached sink tokens. [Primary reference](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf). |
| Chunked attention | [meta-llama/Llama-4-Maverick-17B-128E](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E/blob/10751cb97a4d7c90f7ed89196b98eb8220cfa1c2/config.json) (held) | `text_config.attention_chunk_size`. Verify mask/chunk boundary semantics. A generic chunk_size used by a recurrent kernel is not sufficient. |
| Attention/output logit softcapping | [xai-org/grok-2](https://huggingface.co/xai-org/grok-2/blob/daf4395a80ad177386cfe39641b64fc12b1d70ed/config.json) (cache_only) | `attn_logit_softcapping`, `final_logit_softcapping`. Separate attention logits from vocabulary/output logits, and require a positive enabled cap. Gemma-3-1b has null caps and must not receive this tag. |
| Clamped / bounded SwiGLU variants | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json) (cache_only) | `swiglu_limit`. Inspect the exact activation, clamping and residual operation; a generic silu activation field cannot describe this variant. [Primary reference](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf). |
| Group-limited expert routing | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) (cache_only) | `n_group`, `topk_group`, `topk_method`. The example has eight groups and selects four. A group count of one or a generic use_grouped_topk flag alone does not establish nontrivial group restriction. |
| Bias-corrected expert routing | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) (cache_only) | `topk_method`, `scoring_func`. Follow noaux_tc to e_score_correction_bias and the selection path in modeling_deepseek.py. This is a lead; do not infer a complete training/load-balancing method from the config string. |
| Embedding / residual rescaling | [ibm-granite/granite-4.1-30b](https://huggingface.co/ibm-granite/granite-4.1-30b/blob/4fae6278f7132abf5e971f9de49ebbad09c54cce/config.json) (cache_only) | `embedding_multiplier`, `residual_multiplier`, `attention_multiplier`. Explicit scaling is a structural/numerical property. Do not label it muP or a named training parameterization without the primary definition. |
| Matrix-memory LSTM (mLSTM / xLSTM) | [NX-AI/xLSTM-7b](https://huggingface.co/NX-AI/xLSTM-7b/blob/9dc507bd0939cf372a4a4f667335651d8e49dddb/config.json) (cache_only) | `mlstm_round_up_to_multiple_of`, `chunkwise_kernel`. Inspect the official xLSTM implementation to establish its recurrent cell. Different from a looped Transformer and from an SSM. [Primary reference](https://arxiv.org/abs/2405.04517). |

## What to add first

1. Surface the already approved normalization placement, FFN variants, shared experts, embedding tying and positional subtypes with per-field status and pinned source links.
2. Make MQA, KDA and Mamba-2 explicit in the registry while retaining their parent concepts.
3. Add dedicated tests for sinks, capped activations, group routing, n-gram memory and generation schemes before publishing new tags.
4. Keep the hierarchy explicit: attention family → mechanism → variant; FFN → gate → activation; position encoding → scaling → scope.
5. Keep training settings, storage formats and kernel/runtime implementation details in separate facets. Do not pad the architecture concept count with them.

## Negative cases to preserve

- `num_nextn_predict_layers: 0` is not a positive MTP match.
- `no_repeat_ngram_size` is a generation restriction, not Engram memory.
- `chunk_size` in Mamba/xLSTM can be a compute-kernel setting, not a chunked attention mask.
- `n_group: 1` does not imply nontrivial group-limited routing.
- Null softcaps do not establish softcapping.
- A vision-only normalization flag does not label the entire decoder.
- `silu` in a Mamba path does not override a `relu2` FFN activation.

The audit used cached pinned primary files and browsed original research/model documentation. It did not refresh every Hub repository or verify every candidate operator end-to-end.
