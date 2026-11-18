# Concept coverage audit — 17 November 2026

The 28 distinct labels supplied by the user were checked against the existing saved records, verifier decisions, renderer inputs and pinned primary configs. No gallery model-to-concept assignments were imported. This is a coverage audit, not new extraction or diagram qualification.

Inspected **24 primary records**, excluding fine-tunes, with **13 published models** and **13 held models** (some held repositories have no saved record).

| Concept | Published model matches | Held record matches | Assessment |
|---|---:|---:|---|
| MHA | 1 | 0 | Query, key and value head organization; do not apply a standard head-count inference to MLA. |
| GQA | 8 | 6 | MQA is a separate subtype; the existing GQA guide also explains that endpoint. |
| MLA | 3 | 3 | Require the latent-attention path, not just fewer KV heads. |
| MoE | 4 | 10 | Routed experts must be present; active and stored parameters remain separate. |
| SWA | 3 | 3 | A window value alone does not establish the per-layer schedule. |
| DSA | 1 | 2 | Specific DeepSeek sparse/indexed-MLA path; generic sparse attention is not automatically DSA. |
| QK-Norm | 4 | 4 | Normalization of queries/keys; this does not determine the decoder normalization algorithm. |
| RoPE | 10 | 10 | Retain per-path settings and partial rotary scope. Do not assume that every path uses RoPE. |
| SiLU | 7 | 9 | The published activation must say SiLU/Swish; SiTU-GLU is a different claim. |
| NoPE | 2 | 1 | Scope to the MLA path described by mla_use_nope. Missing RoPE metadata never establishes NoPE. |
| Gated Attention | 3 | 0 | Output gating only; a gated FFN or sigmoid router is not proof of gated attention. |
| Hybrid Attention | 5 | 3 | Use an explicit site definition: attention combined with linear/recurrent/state-space/convolution mixers. Mixing local/global softmax alone is not this category. |
| ShortConv | 4 | 1 | Distinguish standalone decoder mixer layers from the short convolution inside KDA. Never add the nested operation to the decoder layer sum. |
| Attention Residuals | 2 | 0 | Block attention-residual setting is retained. Preserve its source and block scope; do not infer it from a generic residual multiplier. |
| LatentMoE | 2 | 0 | Published expert latent width and routed expert pool support a latent expert path. |
| MTP | 0 | 7 | Only positive module counts/parameters. A zero-layer setting does not mean the model contains MTP; keep module counts separate from the decoder total. |
| KV Sharing | 0 | 1 | Existing structured field occurs only in a held record. Shared index selection is not the same as shared KV cache. |
| Looped Transformer | 0 | 1 | Shared-layer loops and their per-loop cache semantics are present only in a held record. |
| Qwen Sparse Attention | 0 | 1 | Family-specific kind is retained only in a held record; indexer dimensions, micro-block selection and cache semantics still need qualification. |
| RMSNorm | 0 | 0 | No published normalization-type field. rms_norm_eps and norm tensor names are useful leads, not proof of the normalization equation. |
| PLE | 0 | 0 | ple_embed_dim is present in the pinned Qwen config but is not consumed into a checked concept fact. Do not reinterpret n-gram memory as per-layer embedding. |
| CSA/HCA | 0 | 0 | Compression ratios are retained in primary configs, but extraction collapses the mechanism to an unsupported compressed-attention kind. Add distinct compression/selection and per-layer rules. |
| mHC | 0 | 0 | hyper_connection_multiplier is too generic. The pinned DeepSeek config also has Sinkhorn settings; validate the constrained mixing implementation before calling it mHC. |
| IndexShare | 0 | 0 | Pinned configs contain indexer_types or index_source_layer_ids, but the extractor has no checked index-sharing schedule. Keep index reuse separate from KV reuse. |
| Gated Residuals | 0 | 0 | The Qwen card describes this, while extraction retains only a generic hyper-connection count. Read and verify the read/write gate semantics independently of mHC. |
| Attention Budgeting | 0 | 0 | No qualified field or representative recorded family. Define whether the budget is per layer/head, static/dynamic, and training/inference before adding a rule. |
| CCA | 0 | 0 | No qualified field or representative recorded family. Confirm the full term and model-specific implementation from the primary report/code. |
| PolyNorm | 0 | 0 | No qualified field or representative recorded family. Confirm the normalization definition and scope from its primary implementation. |

## Interpretation

- `published_fields` means the required concept signals exist in approved fields of at least one explorable model. It does not mean the website already has this guide/tag, nor that every supporting field is cross-verified. Single-source decisions must stay labelled.
- `held_fields` means structured signals exist only in held records. A concept guide may explain the mechanism, but these models must remain excluded from published model pages, diagrams, comparisons and data exports.
- `needs_extraction_or_semantics` means more extraction and primary-source interpretation is needed. Raw config leads in the JSON are diagnostic, not publishable tags.
- This checks the models we actually have, not all 124 models in the supplied reference list. Lack of a match is unknown or uncovered, not evidence that a concept is absent.

## Recommended implementation

1. Define a concept registry with canonical names, aliases, explicit criteria, scope and primary educational references. Keep it separate from model facts.
2. Derive concept matches from approved fields and retain the field, exact value, verifier status, pinned source link and scope for every match.
3. Add the eight currently missing guides/tags supported by published fields: QK-Norm, RoPE, SiLU, NoPE, Gated Attention, Hybrid Attention, Attention Residuals and LatentMoE. Improve ShortConv membership to include KDA sublayers without changing the decoder layer count.
4. Add dedicated extractor/AST/weight-header tests for normalization type, per-layer embeddings, residual variants, index sharing and compressed-attention schedules. An epsilon value, tensor name or generic multiplier is insufficient by itself.
5. Surface concept evidence outside the diagram first. If a concept later changes diagram labels or wiring, version the spec and revalidate generated diagrams; never append claims to an old approved SVG.

## Example distinctions

- Qwen3-0.6B-Base has a published, single-source `qk_norm: true` from its own weights even though the supplied gallery row omits QK-Norm. Track our evidence rather than reproduce the reference list.
- Kimi-K3 and Kimi-K3.1-Pro have published latent-MoE widths, attention residual block sizes, output gates and MLA NoPE flags. Their short convolution lives inside the KDA path. Their published activation is SiTU, not SiLU.
- Qwen3.8-Flash-Next has a raw `ple_embed_dim` setting and its primary card describes gated residuals and QSA. These are extraction leads in a held architecture, not permission to publish its graph.
- Mistral-Large-4 has a published shared-KV count plus a raw index-sharing schedule. Those are different concepts; its graph/cache remain held.

## Primary sources

- [RMSNorm definition](https://arxiv.org/abs/1910.07467)
- [RoPE definition](https://arxiv.org/abs/2104.09864)
- [QK normalization paper](https://arxiv.org/abs/2010.04245) — a conceptual reference; modern variants can use a different norm implementation.
- Model-specific evidence: `results/concept-coverage.json` includes approved values, statuses and links pinned to each source commit. Raw config leads are explicitly separate.

No website, extractor, record, gold-label, provider-budget or diagram files were changed for this audit. Existing local redesign changes are retained.
