# Additional concepts and primary evidence — 17 November 2026

Reviewed the existing 30-entry discovery catalog against the saved verifier records and pinned source configs, then added two further leads: nGPT and within-layer key/value projection tying. This review covers 32 candidate entries beyond the user-supplied 28 names. Some are subtypes or structural properties; they are not 32 independent algorithms.

**Validation:** 37 baseline examples, 68 baseline field values, two additional config values and one official-card claim checked; zero mismatches. Sources for saved records use their pinned commits. The wider cache is a discovery pool, includes derivatives and is not an admitted-model count. Not every repository was refreshed live.

## How to interpret the evidence

- **Published signals (16):** facts exist in an explorable model. Their per-field verifier status still matters; this does not establish that every canonical mechanism name has been checked in executable logic.
- **Held signals (5):** facts exist in a held model; its diagram, comparison and export remain withheld.
- **Primary leads (11):** source files expose the setting or claim, but concept extraction and/or execution semantics still need qualification.
- A config flag identifies configuration intent. Modeling code shows how it is implemented, and tensor shapes help cross-check the selected path. Missing fields or disabled flags do not establish a positive match.

## Signals in published model records

| Candidate | Example and state | Exact evidence | Source |
|---|---|---|---|
| Multi-query attention (MQA) | google/gemma-3-1b-pt (published) | `attention_type = "MQA"` (single source)<br>`num_attention_heads = 4` (single source)<br>`num_kv_heads = 1` (single source) | [source 1](https://huggingface.co/google/gemma-3-1b-pt/blob/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/config.json) |
| Mamba-2 / structured state-space duality | nvidia/Nemotron-H-8B-Base-8K (published) | `layer_type_counts = {"full_attention": 4, "mamba2": 24, "mlp": 24}` (single source)<br>`mamba_state_size = 128` (single source)<br>`mamba_head_dim = 64` (single source) | [source 1](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/94ea861e008c2dfced3e8e1302094024077aa04e/config.json) |
| Kimi Delta Attention (KDA) | moonshotai/Kimi-K3.1-Pro (published) | `layer_type_counts = {"gated_dsa_mla": 24, "kda_linear": 69}` (verified)<br>`linear_attn_num_heads = 96` (single source)<br>`linear_attn_conv_kernel = 4` (single source) | [source 1](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/README.md) · [source 2](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) |
| SwiGLU / SiLU-gated feed-forward networks | Qwen/Qwen3-0.6B-Base (published) | `activation = "silu"` (single source)<br>`ffn_gated = true` (single source) | [source 1](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json) · [source 2](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/model.safetensors) |
| GEGLU-family / GELU-gated feed-forward networks | google/gemma-3-1b-pt (published) | `activation = "gelu_pytorch_tanh"` (single source)<br>`ffn_gated = true` (single source) | [source 1](https://huggingface.co/google/gemma-3-1b-pt/blob/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/config.json) · [source 2](https://huggingface.co/google/gemma-3-1b-pt/blob/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/model.safetensors) |
| SiTU-GLU | moonshotai/Kimi-K3.1-Pro (published) | `activation = "situ"` (verified)<br>`activation_card_name = "SiTU-GLU"` (single source)<br>`ffn_gated = true` (single source) | [source 1](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/README.md) · [source 2](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) · [source 3](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/model.safetensors.index.json) |
| GELU activation variants | microsoft/phi-2 (published) | `activation = "gelu_new"` (single source)<br>`ffn_gated = false` (single source) | [source 1](https://huggingface.co/microsoft/phi-2/blob/810d367871c1d460086d9f82db8696f2e0a0fcd0/config.json) · [source 2](https://huggingface.co/microsoft/phi-2/blob/810d367871c1d460086d9f82db8696f2e0a0fcd0/model.safetensors.index.json) |
| Squared ReLU (ReLU2) | nvidia/Nemotron-H-8B-Base-8K (published) | `activation = "relu2"` (single source)<br>`mamba_activation = "silu"` (single source) | [source 1](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/94ea861e008c2dfced3e8e1302094024077aa04e/config.json) |
| Sandwich normalization | LiquidAI/LFM3-1.7B-Base (published) | `sandwich_norm = true` (verified) | [source 1](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/config.json) · [source 2](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/model.safetensors) |
| Parallel attention and MLP residual block | microsoft/phi-2 (published) | `block_layout = "parallel"` (single source) | [source 1](https://huggingface.co/microsoft/phi-2/tree/810d367871c1d460086d9f82db8696f2e0a0fcd0) |
| Tied input/output embeddings | Qwen/Qwen3-0.6B-Base (published) | `tied_embeddings = true` (single source) | [source 1](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/da87bfb608c14b7cf20ba1ce41287e8de496c0cd/config.json) |
| Shared MoE experts | deepseek-ai/DeepSeek-V2-Lite (published) | `moe_shared_experts = 2` (single source)<br>`moe_shared_expert_intermediate = 2816` (single source) | [source 1](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) · [source 2](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/model.safetensors.index.json) |
| Sigmoid expert routing | moonshotai/Kimi-K3.1-Pro (published) | `moe_router = "sigmoid"` (single source) | [source 1](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) |
| Partial rotary embeddings | microsoft/phi-2 (published) | `rope = {"partial_rotary_factor": 0.4, "rope_theta": 10000.0}` (single source) | [source 1](https://huggingface.co/microsoft/phi-2/blob/810d367871c1d460086d9f82db8696f2e0a0fcd0/config.json) |
| YaRN context extension | deepseek-ai/DeepSeek-V2-Lite (published) | `rope = {"rope_theta": 10000, "scaling": {"beta_fast": 32, "beta_slow": 1, "factor": 40, "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 4096, "type": "yar…` (single source) | [source 1](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/config.json) |
| Llama-3 frequency-dependent RoPE scaling | meta-llama/Llama-3.1-8B (published) | `rope = {"rope_theta": 500000.0, "scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}}` (single source) | [source 1](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/config.json) |

Qualification notes:

- **Multi-query attention (MQA):** A subtype of grouped-query attention with one KV head. Already explained in the GQA guide, but deserves an explicit subtype in model evidence.
- **Mamba-2 / structured state-space duality:** Already has an ArchWatch guide; absent from the supplied 28-name inventory. Keep Mamba state separate from attention KV cache. [Primary reference](https://arxiv.org/abs/2405.21060)
- **Kimi Delta Attention (KDA):** A named linear-attention subtype, not a synonym for all hybrid attention. Already represented in diagrams; currently explained under the broader linear-attention guide. [Primary reference](https://arxiv.org/abs/2510.26692)
- **SwiGLU / SiLU-gated feed-forward networks:** Derived candidate from the activation and gated tensor layout. Verify multiplication order and active implementation before giving a canonical operation label. [Primary reference](https://arxiv.org/abs/2002.05202)
- **GEGLU-family / GELU-gated feed-forward networks:** GELU-family activation plus gated FFN layout is a useful derived lead; retain the tanh approximation variant and verify implementation. [Primary reference](https://arxiv.org/abs/2002.05202)
- **SiTU-GLU:** The card names SiTU-GLU. This must not be relabelled SiLU/SwiGLU.
- **GELU activation variants:** Phi uses gelu_new while Gemma uses gelu_pytorch_tanh. Activation family, approximation and FFN gating are separate properties.
- **Squared ReLU (ReLU2):** The FFN says relu2 while the Mamba component says silu. Scope the tag to the component, not every activation in the model.
- **Sandwich normalization:** Normalization placement is already represented. It does not establish the normalization equation or imply RMSNorm by itself.
- **Parallel attention and MLP residual block:** Already represented as parallel. It is different from serial decoder sublayers and from running layers on parallel hardware.
- **Tied input/output embeddings:** A storage/parameter-sharing property supported by existing checks; distinguish it from sharing embeddings between MTP and decoder modules.
- **Shared MoE experts:** A useful subtype of MoE. Shared experts are separate from top-k routed experts and latent expert projections.
- **Sigmoid expert routing:** The router scoring function is not the attention gate or FFN activation. Normalization/top-k order still needs separate evidence.
- **Partial rotary embeddings:** Record the fraction and attention path. Do not assume a local path uses the same fraction as a global path.
- **YaRN context extension:** Existing RoPE metadata identifies the extension type and settings; this is not measured long-context quality. [Primary reference](https://arxiv.org/abs/2309.00071)
- **Llama-3 frequency-dependent RoPE scaling:** Existing scaling metadata has the llama3 type and frequency factors. Treat it as a RoPE subtype, not another attention operator.

## Signals in held records

| Candidate | Example and state | Exact evidence | Source |
|---|---|---|---|
| Gated DeltaNet | Qwen/Qwen3.8-Flash-Next (held) | `layer_type_counts = {"gated_delta_net": 36, "qwen_sparse_attention": 12}` (single source)<br>`linear_num_key_heads = 16` (single source)<br>`linear_num_value_heads = 48` (single source) | [source 1](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json) |
| Engram / hashed n-gram conditional memory | deepseek-ai/DeepSeek-V4.1-Flash-1017 (held) | `ngram_layer_ids = [1, 14]` (single source) | [source 1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/config.json) |
| Block/discrete text diffusion | google/diffusiongemma-26B-A4B-it (held) | `generation_type = "block_diffusion"` (single source)<br>`canvas_length = 256` (single source) | [source 1](https://huggingface.co/google/diffusiongemma-26B-A4B-it/blob/f7f5b7f5fa82ffc52addd066915886d497f5517b/config.json) |
| Interleaved multimodal RoPE (M-RoPE) | Qwen/Qwen3.8-Flash-Next (held) | `rope = {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}` (single source) | [source 1](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/812d96e10a0dd82b69d749e68693698cf6c36daa/config.json) |
| Different local/global attention head geometry | google/gemma-4-26B-A4B (held) | `head_dim = 256` (single source)<br>`global_head_dim = 512` (single source)<br>`num_kv_heads = 8` (single source)<br>`num_global_kv_heads = 2` (single source) | [source 1](https://huggingface.co/google/gemma-4-26B-A4B/blob/24548b62aa021d562695c04aaf7758a1ea47990b/config.json) |

Qualification notes:

- **Gated DeltaNet:** Explicitly named in the pinned card and represented in a held architecture. Do not infer it from every linear_attention string. [Primary reference](https://arxiv.org/abs/2412.06464)
- **Engram / hashed n-gram conditional memory:** Layer IDs are extracted; memory addressing and module semantics still need qualification. Distinct from PLE and from no_repeat_ngram generation settings. [Primary reference](https://arxiv.org/abs/2601.07372)
- **Block/discrete text diffusion:** A different generation scheme, not a second attention type. Decoder/cache rules remain held. [Primary reference](https://ai.google.dev/gemma/docs/diffusiongemma)
- **Interleaved multimodal RoPE (M-RoPE):** The metadata contains mrope_interleaved and section widths. Verify spatial/temporal coordinate semantics in code before a multimodal graph.
- **Different local/global attention head geometry:** A structural property affecting cache formulas, nested under the local/global attention concepts rather than counted as a new attention algorithm.

## Primary-source leads not yet extracted or qualified

| Candidate | Example and state | Exact evidence | Source |
|---|---|---|---|
| Vision patch merging / visual token compression | moonshotai/Kimi-K3.1-Pro (published) | `vision_config.mm_projector_type = "patchmergerv2"` (raw primary setting)<br>`vision_config.merge_kernel_size = [2, 2]` (raw primary setting)<br>`vision_config.merge_type = "sd2_tpool"` (raw primary setting) | [source 1](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/d9e99b0792a5e93b78ea9bea62a68167e36ddd9e/config.json) |
| Learned attention sinks | XiaomiMiMo/MiMo-V2-Flash (cache_only) | `add_swa_attention_sink_bias = true` (raw primary setting)<br>`add_full_attention_sink_bias = false` (raw primary setting) | [source 1](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/8b60b288e8325b7890fcdfe32bbb1d172c591a2d/config.json) |
| Chunked attention | meta-llama/Llama-4-Maverick-17B-128E (held) | `text_config.attention_chunk_size = 8192` (raw primary setting) | [source 1](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E/blob/10751cb97a4d7c90f7ed89196b98eb8220cfa1c2/config.json) |
| Attention/output logit softcapping | xai-org/grok-2 (cache_only) | `attn_logit_softcapping = 30.0` (raw primary setting)<br>`final_logit_softcapping = 50` (raw primary setting) | [source 1](https://huggingface.co/xai-org/grok-2/blob/daf4395a80ad177386cfe39641b64fc12b1d70ed/config.json) |
| Clamped / bounded SwiGLU variants | openai/gpt-oss-20b (cache_only) | `swiglu_limit = 7.0` (raw primary setting) | [source 1](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json) |
| Group-limited expert routing | deepseek-ai/DeepSeek-V3 (cache_only) | `n_group = 8` (raw primary setting)<br>`topk_group = 4` (raw primary setting)<br>`topk_method = "noaux_tc"` (raw primary setting) | [source 1](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) |
| Bias-corrected expert routing | deepseek-ai/DeepSeek-V3 (cache_only) | `topk_method = "noaux_tc"` (raw primary setting)<br>`scoring_func = "sigmoid"` (raw primary setting) | [source 1](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) |
| Embedding / residual rescaling | ibm-granite/granite-4.1-30b (cache_only) | `embedding_multiplier = 12.0` (raw primary setting)<br>`residual_multiplier = 0.175` (raw primary setting)<br>`attention_multiplier = 0.0078125` (raw primary setting) | [source 1](https://huggingface.co/ibm-granite/granite-4.1-30b/blob/4fae6278f7132abf5e971f9de49ebbad09c54cce/config.json) |
| Matrix-memory LSTM (mLSTM / xLSTM) | NX-AI/xLSTM-7b (cache_only) | `mlstm_round_up_to_multiple_of = 64` (raw primary setting)<br>`chunkwise_kernel = "chunkwise--triton_xl_chunk"` (raw primary setting) | [source 1](https://huggingface.co/NX-AI/xLSTM-7b/blob/9dc507bd0939cf372a4a4f667335651d8e49dddb/config.json) |
| Normalized Transformer (nGPT) | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (held) | `ngpt_alpha_init = 0.05` (raw primary setting)<br>Official card names Looped (latent-recurrent) normalized-GPT (line 46) | [source 1](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/ee517dd388a9b348890ae1e00dacc5eb4d0c750d/config.json) · [source 2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/blob/ee517dd388a9b348890ae1e00dacc5eb4d0c750d/README.md#L46) |
| Within-layer key/value projection tying (K=V) | google/gemma-4-26B-A4B (held) | `text_config.attention_k_eq_v = true` (raw primary setting) | [source 1](https://huggingface.co/google/gemma-4-26B-A4B/blob/24548b62aa021d562695c04aaf7758a1ea47990b/config.json) |

Qualification notes:

- **Vision patch merging / visual token compression:** Source names patchmergerv2 and a merge kernel. Extract and validate projector/token transformations; current decoder diagrams do not qualify this path.
- **Learned attention sinks:** Pinned modeling_mimo_v2_flash.py lines 113 and 335-337 show sink use/creation. Learned softmax-denominator sink biases are different from retaining initial cached sink tokens. [Primary reference](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
- **Chunked attention:** Verify mask/chunk boundary semantics. A generic chunk_size used by a recurrent kernel is not sufficient.
- **Attention/output logit softcapping:** Separate attention logits from vocabulary/output logits, and require a positive enabled cap. Gemma-3-1b has null caps and must not receive this tag.
- **Clamped / bounded SwiGLU variants:** Inspect the exact activation, clamping and residual operation; a generic silu activation field cannot describe this variant. [Primary reference](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf)
- **Group-limited expert routing:** The example has eight groups and selects four. A group count of one or a generic use_grouped_topk flag alone does not establish nontrivial group restriction.
- **Bias-corrected expert routing:** Follow noaux_tc to e_score_correction_bias and the selection path in modeling_deepseek.py. This is a lead; do not infer a complete training/load-balancing method from the config string.
- **Embedding / residual rescaling:** Explicit scaling is a structural/numerical property. Do not label it muP or a named training parameterization without the primary definition.
- **Matrix-memory LSTM (mLSTM / xLSTM):** Inspect the official xLSTM implementation to establish its recurrent cell. Different from a looped Transformer and from an SSM. [Primary reference](https://arxiv.org/abs/2405.04517)
- **Normalized Transformer (nGPT):** The official card explicitly names normalized-GPT; config supplies an nGPT setting. The extractor does not publish a normalized-Transformer concept. Dory remains held; the card claim does not qualify loop/cache equations. [Primary reference](https://arxiv.org/abs/2410.01131)
- **Within-layer key/value projection tying (K=V):** Pinned config enables K=V projection tying. The official Transformers documentation defines the flag as reusing key projection outputs for values. This differs from reusing KV across decoder layers. Layer scope, tensor checks and diagram/cache qualification remain open; the model stays held. [Primary reference](https://huggingface.co/docs/transformers/main/model_doc/gemma4)

## Operation-level evidence spot checks

- MiMo-V2-Flash: [modeling_mimo_v2_flash.py line 113](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/8b60b288e8325b7890fcdfe32bbb1d172c591a2d/modeling_mimo_v2_flash.py#L113) handles the sink term; lines 335–337 construct the enabled learned bias. This is not the same mechanism as retaining the first cache tokens.
- DeepSeek-V3: [modeling_deepseek.py line 439](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L439) adds correction bias for expert choice; lines 443–457 select groups and mask disallowed experts. This establishes selection-path evidence, not the complete training/load-balancing recipe.
- LFM3: [modeling_lfm3.py line 458](https://huggingface.co/LiquidAI/LFM3-1.7B-Base/blob/7bca5038823c0d0f14efe06439109354a10e014c/modeling_lfm3.py#L458) creates the conditional sandwich norms; lines 489 and 494 use them around the mixer/FFN paths.

## Uninterpreted fields remain leads

These are real settings, not confirmed new mechanism names. They are excluded from the 32 named entries.

| Source snapshot | Exact raw settings | What remains unknown |
|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/config.json) | `text_config.dspark_block_size = 5`<br>`text_config.dspark_markov_rank = 256`<br>`text_config.dspark_target_layer_ids = [37, 38, 39]` | The config exposes an auxiliary module cluster; its intended generation/training mechanism has not been established from primary implementation or report. |
| [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling/blob/828496eeae4c243ff1a22f7f28ff83694f2f7bc9/config.json) | `text_config.use_embed_norm = true`<br>`text_config.shared_expert_sink = true` | These are component flags. Establish the embedding normalization placement and the meaning of the shared-expert sink before attaching a canonical tag. |
| [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/ef5bf64c1b8e7d0e57b284e10c6891f8db6304c4/config.json) | `text_config.use_value_norm_no_gamma = true`<br>`text_config.attn_res_normalize_values = true`<br>`text_config.mlp_gate_softcap = 4.0`<br>`text_config.mlp_up_softcap = 25.0` | These describe normalization/activation variants. Their scope and actual forward operations need inspection; do not classify them from flag names alone. |

No website, model record, extractor, generated diagram, frozen gold label or provider-budget file was changed by this review. The two new review artifacts are uncommitted on local main.

Machine-readable evidence: `results/concept-evidence-review.json`. The original 30-entry report is preserved in `docs/ADDITIONAL_CONCEPT_DISCOVERY.md` and `results/additional-concept-discovery.json`.
