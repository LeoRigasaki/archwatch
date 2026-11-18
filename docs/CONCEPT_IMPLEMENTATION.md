# Concept evidence and Compare implementation — 18 November 2026

This is local, uncommitted work on `main`. It extends the local redesign and pending trust/sorting integration. No PR was merged, no deployment or schedule was changed, and no Gemini generation request was made.

## What is now on the site

The catalog has **32 guides: all 28 requested concepts plus four existing companion topics** (linear attention, general sparse attention, Mamba-2 and KV cache). Every guide now contains motivation, a worked numerical or structural example, equation/notation, an original downloadable SVG, comparison pitfalls, verification guidance, a primary reference and a matching rule. The generated lesson articles contain 298–652 words, median 371, excluding their model-example sidebar. GQA has an independently checked 16/4/2 MiB MHA/GQA/MQA example.

Five guides have interactive teaching calculators: MHA, GQA, KV cache, sliding windows and RMSNorm. Three include minimal Python illustrations. Toy values are explicitly separated from checkpoint facts. Educational content never supplies architecture values to the extractor, verifier or model diagrams.

Model pages show concept associations with scope, exact supporting published values, per-field decisions and source links pinned to the model commit. Associations with any single-source supporting field remain labelled one source. Derived calculations do not receive extra independent-source credit. Every model has a downloadable `archwatch-concept-evidence/1` companion. It does not alter or re-label an approved generated SVG.

The Compare layout follows the supplied **ArchWatch site mockup.html**: two model cards with layer strips and source links, a working third-model card, Plain/Technical mode, “What changed”, a field table, difference-only filtering, share links/CSV, cache bars and optional detailed curves. Model changes, swaps and third-model removal update the URL and values. Invalid selections retain the prior comparison and show a message. Adaptive number precision distinguishes close parameter counts instead of displaying two identical rounded values beside a change badge. MLA rows show query heads and latent KV width rather than implying a conventional per-head KV cache.

The mockup's sample paragraph claimed a RoPE-to-NoPE change between the saved Kimi records. Both pinned records already state NoPE on that path. The implementation uses the computed record differences and does not import that claim. Cache bars include the calculator's fixed and windowed terms, unlike the mockup's growing-only illustration. Content hashes version CSS, scripts and comparison JSON so updated markup does not run stale behavior.

## Matching coverage and source enrichment

Of the 28 requested concepts:

- **16** have passing matches in at least one of the 13 published models.
- **6** have signals only in held records: IndexShare, KV Sharing, Looped Transformer, MTP, PLE and Qwen Sparse Attention.
- **6** have no qualified current-record match: Attention Budgeting, CCA, CSA/HCA, Gated Residuals, mHC and PolyNorm. They have full educational guides, not invented model associations.

The model inventory remains 13 published and 13 held. Raw discovery-pool repositories and fine-tunes are not admitted by this feature.

`tools/enrich_concepts.py` adds optional facts from exact cached primary snapshots without network access. It enriched five records:

| Model | Added optional evidence |
|---|---|
| LFM3-1.7B-Base | Decoder RMSNorm type and module/formula locations |
| DeepSeek-V2-Lite | Decoder RMSNorm type and module/formula locations |
| Qwen3.8-Flash-Next, held | Dedicated PLE width and source layer IDs |
| Mistral-Large-4, held | Explicit indexer shared modes and KV source mapping |
| GLM-5.2, held | Complete index source mapping |

The normal extractor/scanner/verifier also recognize these optional fields for future pipeline runs. Additional conservative handlers retain explicit normalization names, compression variants, residual-mechanism identifiers, attention-head allocations and CCA settings. They do not reinterpret an epsilon, a generic stream multiplier or a generic top-k as a named mechanism. A handler's existence does not qualify every upstream family's implementation or cache topology.

RMSNorm scanning recognizes a narrow AST pattern for squared mean, inverse RMS, multiplication and a returned result, wired into a used decoder normalization module. It rejects centered operations, dead results, overwritten statistics/modules and unresolved branches. It never imports or executes remote code. This is static, single-source evidence, not a claim of complete control-flow verification or coverage of every equivalent implementation.

## Files and contracts

- `config/concepts.yaml`: canonical names, aliases, definitions, references, scopes and deterministic rules.
- `config/concept-lessons.yaml`: original worked examples, notation, pitfalls and illustration labels.
- `src/archwatch/concepts.py`: matching, verifier-value consistency, publication/derivative gate and versioned evidence companion.
- `src/archwatch/concept_extract.py`: optional configuration facts and conservative RMSNorm AST recognition.
- `src/archwatch/sources.py`: shared commit-pinned provenance links.
- `src/archwatch/concept_diagrams.py`: original teaching SVGs.
- `build_site.py`, `web/`: public pages and working interactions.

The existing diagram specification and five approved generated diagrams remain unchanged. Concepts are shown with evidence outside the drawing. A future change that incorporates new claims into SVG labels or wiring requires a new drawing contract and fresh validation, not edits to a previously approved SVG.

## Validation

**318 tests pass.** Ruff lint/formatting, both JavaScript syntax checks and Git whitespace checks pass. Real Chrome exercises comparison mode, adding/removing/swapping models, invalid choices, filtering, exact rounded differences, downloads and teaching calculators. Desktop and mobile views were inspected.

The offline build generates 50 HTML pages and 32 teaching SVGs. **1,231 local resource/page links were checked with no missing targets.** Consecutive builds preserve bytes and modification times. Re-running pinned-source enrichment produces no changes. Existing published values are unchanged; only optional concept facts were added. Gold labels, frozen first-run evidence, approved generated SVGs and provider call ledgers are untouched.

Detailed coverage and validation: `results/concept-implementation.json`. This is implementation and source-consistency evidence, not a new independently labelled held-out accuracy score. Human review and broader unseen-family evaluation remain open.

## Preview

```sh
uv run --frozen python build_site.py
uv run --frozen python -m http.server 8834 --bind 127.0.0.1 --directory site
```

Open <http://127.0.0.1:8834/compare/index.html> and <http://127.0.0.1:8834/concepts/gqa.html>.
