# Model discovery and weekly generation policy

This is the proposed watcher/generation policy, incorporating the user's preferred labs and supplied Gemini quota screenshots on 13 November 2026. It is not implemented or scheduled yet.

## Separate the generator from the models being documented

Gemini is the layout/explanation service. The gallery subjects are open-weight models with inspectable primary architecture evidence. For example, Gemini may draw a DeepSeek diagram; the DeepSeek model itself is not loaded for inference.

Recommended generator: `gemini-4-hydrogen`, using ordinary text generation to produce validated SVG code or a constrained layout specification. The official [model catalog](https://aistudio.google.com/docs/models) lists its model ID, and [pricing](https://aistudio.google.com/docs/pricing#gemini-4-hydrogen) lists free text input/output. This recommendation follows the current coding capabilities and free access, not an ArchWatch SVG quality evaluation. Start with the existing four-model specifications and compare fidelity/layout before enabling it unattended.

Configured free fallback candidate: `gemini-3.8-flash`. Never silently switch to paid models. The deterministic SVG renderer remains available when generation fails or quotas are exhausted. Image-generation models are not required for SVG code output.

## Preferred publisher pool

These Hub namespaces were checked for existing repositories through the official Hub API on 13 November 2026. The allowlist establishes discovery priority, not automatic factual correctness or a guarantee that every release is supported by the extractor.

| Publisher | Primary Hugging Face namespace |
|---|---|
| DeepSeek | [deepseek-ai](https://huggingface.co/deepseek-ai) |
| Alibaba Cloud / Qwen | [Qwen](https://huggingface.co/Qwen) |
| Mistral | [mistralai](https://huggingface.co/mistralai) |
| Liquid AI | [LiquidAI](https://huggingface.co/LiquidAI) |
| NVIDIA | [nvidia](https://huggingface.co/nvidia) |
| Google | [google](https://huggingface.co/google) |
| Meta | [meta-llama](https://huggingface.co/meta-llama) |
| Microsoft | [microsoft](https://huggingface.co/microsoft) |
| Moonshot AI | [moonshotai](https://huggingface.co/moonshotai) |
| Z.ai | [zai-org](https://huggingface.co/zai-org) |

Keep aliases explicit and review new namespaces before treating them as official. Start with language models and multimodal text backbones; speech, embedding, image and video models need separate schema support. Closed/API-only models can have a source-linked overview but cannot receive tensor-verified architecture status without released evidence.

Bootstrap the gallery with at least one representative architecture from each preferred lab, rather than waiting only for new releases. Liquid AI and Moonshot already have live-checked representatives. The next core candidates to inspect are [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and [Mistral-Large-4-1T-A52B](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B), whose repository IDs were confirmed through the Hub API. These are proposed extraction targets, not records already validated by ArchWatch. Rotate Google, NVIDIA, Z.ai, Meta and Microsoft into subsequent core slots, subject to primary-source readiness and extractor support.

## Discovery pool beyond the allowlist

Inspect the global [trending text-generation list](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending), image-text-to-text releases and newly published primary repositories. Trending is a discovery signal, not a quality score or evidence of architectural novelty.

Initial candidates worth inspecting outside the core pool include [Meituan LongCat-2.5](https://huggingface.co/meituan-longcat/LongCat-2.5) and [Baidu ERNIE-5.5-Flash-Base](https://huggingface.co/baidu/ERNIE-5.5-Flash-Base). Their cards/configs/reports must pass the same extraction/review process; this note does not validate their architecture or benchmark claims.

A useful counterexample is [Nous Arete-27B](https://huggingface.co/NousResearch/Arete-27B), currently found in the trending results: its card states there was no architectural modification to its Qwen base. A verified matching architecture can share that base diagram rather than consume a new generation slot.

## Admission and ranking

1. Require an identifiable primary release, inspectable config and available architecture evidence. Read tensor headers only; do not download payloads or execute remote modeling code.
2. Fetch/verify the source record before spending a generation call. Unsupported or conflicting required architecture fields go to review; popularity cannot bypass publication checks.
3. Group conversions and fine-tunes by canonical family and a hash of the published architecture specification. A name match alone is insufficient. Changed dimensions/layer patterns get a distinct specification even within one family.
4. Prioritize explicit user picks, new architecture families, and meaningful changes from known predecessors. Then use trending position, recent activity, downloads and likes as supporting signals, with freshness and source completeness as tie-breakers.
5. Suggested starting popularity filter for outside discovery: 1,000 downloads OR top-50 trending. Preferred-lab new releases and explicit user picks bypass that popularity filter, but never bypass evidence checks. This avoids excluding a newly released model just because downloads have not accumulated.
6. Record a reason for every selected, deferred or rejected candidate, including unsupported-family and quota deferrals. Keep failed candidates retryable.

These are initial policy choices, not empirically optimized ranking weights. No benchmark-quality claim follows from being selected.

## Cadence and budget

- Cheap metadata discovery can run daily once the watcher is tested. Store each repository's last observed commit separately from its last successful processed commit so failed updates are not lost.
- Run the Gemini queue once per week, initially Sunday morning Asia/Karachi, after a successful manual batch and explicit user approval of the schedule. This is a weekly series of standard requests, not Google's separately priced Batch API.
- At most five changed architectures per batch: three preferred-publisher slots and two outside slots. Unused capacity can transfer between pools. Avoid filling slots with duplicates; cap a publisher at two slots initially and rotate overdue families.
- One generation and at most two repair requests per architecture: at most 15 requests for five models, leaving five of the screenshot's 20 daily requests as headroom. Keep reviews and explanations on the separate `gemini-3.5-flash-lite` model (500/day); the diagram model must not grade its own work. All diagram generation, repair and fallback calls count toward the 15-request cap. An HTTP request is not proof that a job succeeded.
- Start no more than four requests per minute and enforce the observed token limit; honor retry/backoff and defer to a later run on exhaustion. Existing account usage reduces available headroom.
- Cache by both source commits/evidence revision and published-specification hash plus generator/prompt version. Reuse the same diagram when the architecture is unchanged; refresh evidence/metadata without an LLM call when only those changed.
- Save provenance, validator outcome and attempts with the artifact. Publish only after deterministic content/SVG validation and review of unresolved discrepancies. Generator failures retain the deterministic rendering.

No cron or Codex task is created by this document. The watcher, queue, provider adapter and validation loop must be implemented and manually exercised first.

The 14 November amendments in [NEXT_STEPS.md](NEXT_STEPS.md) take precedence: test full-SVG and JSON-layout routes on all four existing records before selecting one, keep model names/limits in configuration, and save every generation/review round.
