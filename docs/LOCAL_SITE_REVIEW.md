# Local website review — updated 18 November 2026

All changes are uncommitted on local `main`, based on `e7bb20b`. Nothing was pushed, merged or deployed. The existing remote PRs are unchanged. The build uses saved records; no provider calls or model backfill were run.

The 18 November follow-up adds full teaching content for all 32 guides and implements the supplied Compare mockup. [Current implementation, validation and limits](CONCEPT_IMPLEMENTATION.md) supersede the initial guide and comparison details below.

## What to review

- A new visual identity: warm paper, forest green and orange, a layer-stack mark, Manrope body text, Bricolage Grotesque headings and IBM Plex Mono values. Light and dark themes remain available.
- A homepage driven by the inventory, including the latest published model, 13 explorable models, 13 held models and ten publishers currently represented in tracked records. These are generated counts, not constants in the page code.
- Publisher filtering, mobile navigation and collapsed mobile filters. At 390 × 844, the first model begins about 764 px down the page. Cards and the table retain the same sort order, including deterministic name tie-breaks.
- Select two or three cards to compare. The selection tray creates a shareable comparison URL. Comparisons can export their displayed fields to CSV with repository IDs, pinned commits and primary-file links. A rejected clipboard operation shows a failure message.
- Ten original concept guides: MHA, GQA/MQA, MoE, MLA, sliding windows, linear attention, sparse attention, Mamba-2, short convolution and KV cache. Each has an explanation, a conceptual flow, memory implications, source-reading advice, a primary reference and matching published examples. Membership comes from approved fields.
- An evidence/coverage page with field-check counts, layer composition, plain hold reasons and clear limits. Coverage counts are not accuracy scores.
- Each model has a published-only JSON download, source links pinned to its commit, an expandable architecture diagram and an SVG download. No held architecture values are exported.
- The model memory planner accepts device GB, concurrent sequences and a planning headroom percentage. Weights are shared; persistent state is multiplied by the sequence count. Windowed cache and recurrent state retain their existing calculation rules. Weight precision does not change cache precision. This remains a theoretical estimate, not a runtime benchmark.
- An RSS feed of published analyses. Feed dates come from `analyzed_at`, not Hub creation time or inferred release dates.

ArchWatch's implementation emphasizes an evidence workflow: trace a claim to pinned primary files, distinguish source agreement from computed values, inspect the publication holds, compare supported designs, and plan memory under explicit assumptions. The visual system, guide explanations, flow layouts, pages and interactions here were written for ArchWatch.

Concept sources live in `config/concepts.yaml`: the original Attention, GQA, Switch Transformers, DeepSeek-V2, Mistral, linear-attention and Mamba-2 papers, plus primary DeepSeek/Liquid model files and Hugging Face cache documentation. These educational references do not feed the model extractor or gold labels.

## Trust changes included locally

The changes from open PRs #40 and #41 were applied to the working tree, without making commits. The 16 changes to frozen first-run result records in #40 were excluded. Live metadata uses `hub_created_at`; card-stated release dates remain separate. Counting-convention reports and the corrected commit map are included.

Additional fixes reject invalid calendar dates, bind ordered transitions and deltas to the correct explainer field, reject a reversed direction, link single-file safetensors evidence to the actual pinned file, and fetch full Git history in CI. A SHA256 manifest checks the existing gold labels and frozen first-run artifacts, including in a shallow checkout. It supplements the history-based guard.

Static CSS, shared behavior and comparison behavior now live in `web/` rather than inside the Python builder. `config/concepts.yaml` holds the educational content. Model architecture values still come only from approved record fields and the calculator.

## Validation

- Full offline suite: **318 passed**. The explainer regressions also passed after the final validator adjustment.
- Ruff lint, formatting and Git whitespace checks passed.
- Actual Chrome flow coverage: mobile menu/Escape, collapsed filters, publisher filtering, empty search, reset, selection limit, three-way comparison, clipboard rejection, CSV export, sequence and device inputs, invalid inputs, diagram dialog/Escape and a parseable SVG download.
- Screenshots inspected at 1440 × 1000 and 390 × 844, in light and dark themes. No horizontal overflow in the checked home, model, comparison, guide and evidence pages.
- 50 generated HTML pages; 1,231 local resource/page links checked, none missing.
- Two consecutive builds produced byte-identical files with unchanged modification times.
- Existing gold labels and `results/newest-first-run/` are unchanged from `HEAD`. Existing architecture values are unchanged; five records gained optional concept evidence, alongside the Hub creation metadata rename.
- Machine-readable evidence: `results/local-site-review.json`.

## Run locally

```sh
uv run --frozen python build_site.py
uv run --frozen python -m http.server 8834 --bind 127.0.0.1 --directory site
```

Open <http://127.0.0.1:8834/>. `site/` is generated and ignored; review/commit the source files if you choose to keep this version.

## Still open

Support for the held architectures, a model-page reading-mode toggle, independently human-reviewed gold labels and broader held-out evaluation. This pass does not add model records, regenerate Gemini diagrams or qualify memory figures against hardware.
