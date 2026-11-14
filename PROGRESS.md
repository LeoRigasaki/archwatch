# ArchWatch progress checklist

As of **14 November 2026**, after the watcher and source-driven cleanup merges, manual discovery acceptance and the ten-publisher backfill. This checklist describes verified implementation and remaining acceptance; the original day numbers are planning milestones.

## Implemented and validated

- [x] Prototype integrated into `src/archwatch/`, tests into `tests/`, four JSON records and two successor YAML records into `data/models/`.
- [x] Config extractor: nested text configs, hybrid/dense/MoE and latent MoE, MLA/DSA/KDA, layer lists and one-based indexing.
- [x] LFM effective FFN normalization: explicit auto-adjust rule converts 12,288 to 8,192; agrees with released tensors; regression cases added.
- [x] Weight counting: exact supported-format shape totals, components, scale exclusion and active-count conventions.
- [x] AST scanner with file/line evidence; downloaded code is never executed.
- [x] Cache calculator with GQA, sliding window, short-conv, MLA, DSA and KDA rules and hand-math tests.
- [x] Verifier with explicit verified/single-source/conflict states and conflict withholding.
- [x] Published-only SVG/site inputs; a verified weight-kind layer pattern is required. Missing required inputs and unsupported site cache rules are rejected.
- [x] Site cache estimates and field comparisons recomputed from published values; raw facts, raw parameter counts and stored cache/diff artifacts cannot override them.
- [x] Field and tensor-group predecessor diffs; exact **178,526,208-parameter** DSA-indexer addition retained.
- [x] Two-model local site and standalone SVG outputs rebuilt. Imported PNGs remain historical previews.
- [x] Metadata refreshed once per analysis; later reads use the same commit and revision-specific cache. Two-run/new-commit regression added.
- [x] Header reads reject full-body range responses and oversized headers.
- [x] uv lockfile, Python 3.12 pin, ruff, pre-commit and offline PR CI configured.
- [x] Source ZIP exporter excludes environments, runtime caches, egg-info and private references, including accidentally tracked runtime files.
- [x] **114 offline tests pass**, including publication boundaries, family layouts, normalization, predecessor artifact consistency, refresh, packaging, AST, independent labels and hand-math formulas.
- [x] Fresh primary-source run reproduces all four commits and parameter totals. Of the original 81 checks, only the corrected LFM FFN verdict changes.
- [x] GitHub repository configured at `LeoRigasaki/archwatch`.
- [x] Foundations merged in [PR #1](https://github.com/LeoRigasaki/archwatch/pull/1) after [CI passed](https://github.com/LeoRigasaki/archwatch/actions/runs/43489941541) on its exact head commit; main CI also passed.
- [x] `needs-review` and `auto-merge` labels created; repository setting permits Actions to open PRs. Actual auto-merge remains disabled.
- [x] The [manual workflow](https://github.com/LeoRigasaki/archwatch/actions/runs/43490777972) succeeded from merged main: four records analyzed, 38 tests passed, zero review fields and [update PR #3](https://github.com/LeoRigasaki/archwatch/pull/3) opened with `auto-merge`. That no-op PR was subsequently closed without merging; see the correction below.
- [x] The initial run exposed duplicate Git auth headers; [compatibility PR #2](https://github.com/LeoRigasaki/archwatch/pull/2) passed CI and merged before the successful rerun.
- [x] Watcher branch rebased onto no-op correction main `a232073`; bounded discovery/analysis implementation and tests added.
- [x] Acceptance evidence and four-record equality checks recorded in [github-workflow-acceptance.json](results/github-workflow-acceptance.json).
- [x] Progress/result paths remain relative; research reference remains ignored.

## Decisions recorded, implementation still open

- [x] Preferred publishers: DeepSeek, Alibaba/Qwen, Mistral, Liquid AI, NVIDIA, Google, Meta, Microsoft, Moonshot and Z.ai.
- [x] Keep an outside-discovery route for emerging models, with architecture novelty and source readiness checked before diagram generation.
- [x] Proposed generator: Gemini 4 Hydrogen, with Gemini 3.8 Flash as an explicit free fallback. Account quotas are transcribed from the supplied screenshots.
- [x] Proposed weekly batch: at most five changed architectures, initially three preferred-lab slots and two outside slots; no regeneration for unchanged specifications.
- [x] Codex recommended as optional reviewer; explanations available for any pair on demand, predecessor pairs in automated batches.
- [x] Discovery/analysis selection implemented and live-tested.
- [ ] Provider generation/review policies, route comparison and weekly queue remain unimplemented.

## Still open

- [x] Watcher: ten core publishers, global trending/recent discovery, Hub/config/safetensors admission, verified architecture dedupe, bounded retries, retained work and `data/seen.json`.
- [x] Same-org family predecessor selection with explicit date/catalog evidence and ambiguity handling; both existing live pairs reproduced.
- [x] Manual workflow modes `fixed`, `discover` and `process`; default remains the fixed smoke test and cron stays disabled.
- [ ] Typed/Pydantic schema and broader family/config acceptance coverage.
- [x] Ten development gold labels manually transcribed from primary sources, frozen before family fixes; per-field published exact match reported. Every label is explicitly `human_reviewed: false`.
- [ ] Human review and unseen 30-model evaluation; calculator/citation metrics and the **95% held-out config-field target** remain unmeasured.
- [ ] Gemini validated spec-to-SVG generation/repair loop and response caching; provider quota controls.
- [ ] LLM code reader with structured output and source-line citations.
- [ ] Report finder/retrieval, report agent, page citations and citation validator.
- [ ] LangGraph orchestration and Langfuse traces.
- [ ] GitHub review issue queue and recorded human decisions.
- [ ] Diff-constrained explanations; family/predecessor selection v1 is implemented, with broader historical coverage still open.
- [ ] Controlled scheduled publishing/merging; cron remains disabled until explicit user approval.
- [ ] MCP server (`search_models`, `get_model`, `compare`) and client tests.
- [ ] Full collection/filter/compare UI, own RSS, final model pages and production deployment.
- [ ] **50+ model backfill**, independent evaluation results and demo recording.

The refresh workflow supports a default fixed-list smoke test plus explicit manual discovery/processing modes. Its cron remains disabled; neither a labeled PR nor zero conflicts constitutes automatic merge authorization.

## Milestone map

| Day(s) | Current state |
|---|---|
| 1 | uv/ruff/pytest/pre-commit/PR CI configured and CI passed remotely; typed schema remains open |
| 2 | Watcher/seen state implemented; bounded live discovery and four-model processing/restart checks pass |
| 3–4 | Core extractor implemented; four live models are narrower than the planned family acceptance sets |
| 5 | Cache formulas and hand-math tests implemented; broader/runtime validation remains |
| 6 | Phase target: ten independent gold labels; none written yet |
| 7 | AST scanner and line evidence implemented |
| 8 | LLM reader missing |
| 9 | Verifier and consumer publication boundary implemented; confidence remains heuristic |
| 10–12 | Gold evaluation, metrics, tracing and failure analysis open |
| 13–15 | Report retrieval and page-citation validation open |
| 16–18 | Deterministic diffs exist; bounded predecessor discovery implemented; explanations and issue queue open |
| 19–22 | Two-model site, deterministic SVG and cache UI exist; full collection/any-pair UI and RSS open |
| 23 | Manual fixed-list workflow passed and opened an update PR; discovery implemented; controlled scheduled publishing remains open |
| 24–25 | MCP, 50+ backfill, formal evaluation, production deployment and demo open |

## Evidence and boundaries

The refreshed records contain **61 verified checks, 67 single-source checks and zero conflicts**. Added checks cover inputs previously consumed by renderers without a publication decision. More checks do not establish higher accuracy; the independent gold set still contains zero models.

| Model | Exact parameter total |
|---|---:|
| Kimi-K3 | 2,779,931,837,184 |
| Kimi-K3.1-Pro | 2,780,110,363,392 |
| LFM2.5-1.2B-Instruct | 1,170,340,608 |
| LFM3-1.7B | 1,728,801,536 |

The offline suite checks behavior without network access. The separate live run validates current commit-pinned fetching for these four checkpoints. Pattern agreement is at tensor-kind granularity; identically shaped sliding/global attention still needs config/code evidence. Single-source values are explicitly labeled and permitted by current policy. Cache outputs estimate cache state, not total runtime VRAM.

Evidence: [results](results/README.md), [offline validation](results/prototype-validation.json), [live comparison](results/live-validation.json), [portable snapshot](docs/progress-snapshot.json), [LFM investigation](docs/LFM_FINDINGS.md).

## Private reference boundary

- [x] Legacy hosting policy checked owner-only; unauthenticated page, diagram and model JSON returned HTTP 401 in the recorded access review.
- [x] Legacy hosted link omitted from README and absent from the portfolio projects checked during the import.
- [x] Third-party diagrams/gallery remain ignored private research, outside production records and the gold set.

Evidence: `docs/deployment-access-review.json`. Hosting was not changed or re-audited in this foundations patch.

Next after backfill review: compare the two Gemini diagram routes, following [NEXT_STEPS.md](docs/NEXT_STEPS.md) using the [selection policy](docs/MODEL_SELECTION.md). The original four diagnostic records include held Liquid Instruct releases; choose eligible primary records for rendering rather than bypassing their publication gate.

The implementation amendments approved on 14 November are recorded in [NEXT_STEPS.md](docs/NEXT_STEPS.md): CI-gated foundations merge and manual workflow acceptance first; then separate watcher, route-test, Gemini-loop, explainer and ten-model gold-set PRs. Reviews/explanations use Flash Lite, diagram routes must be compared, and cron requires explicit approval.

## No-op update correction (14 November)

- [x] Closed PR #3 without merging and deleted `daily-update`; its changes were timestamps and equal-count ordering only.
- [x] Tensor groups and parameter-component ties use count plus name ordering; code/file scans, representative shapes and cache rows have stable ordering. Layer sequence order remains architectural data.
- [x] JSON comparison ignores only `analyzed_at`, preserving the saved record/date when unchanged. YAML, SVG and HTML writes preserve existing files when unchanged.
- [x] Cached end-to-end regression checks identical bytes and modification times, changing-input updates, and cross-process hash-seed stability. All 41 tests pass.
- [x] Two local four-model runs leave every artifact byte and modification time unchanged after the one-time ordering migration.
- [x] [No-op correction PR #4](https://github.com/LeoRigasaki/archwatch/pull/4) merged after green CI. Both [acceptance run 1](https://github.com/LeoRigasaki/archwatch/actions/runs/43492918439) and [run 2](https://github.com/LeoRigasaki/archwatch/actions/runs/43493166837) succeeded with operation `none` and no PR; watcher work resumed only afterward.

## Watcher validation (14 November)

- 66 offline tests pass, including admission, duplicate fine-tunes/conversions, unknown-mechanism review, pinned revisions, retries, changed commits, predecessor ambiguity, pool limits, dropped listings and no-op restart behavior.
- A temporary live output root processed all four known models, selected both expected predecessor pairs, and remained byte/mtime-identical on a second pass.
- A separate configured live discovery inspected 80 candidates: 37 pending analysis, five held for access/source review, 32 rejected and six popularity-deferred. These are discovery outcomes, not verified architectures or quality rankings. The scan did not publish any new model.
- The checked-in seen state seeds only the four existing records. Gemini requests, weekly generation, independent gold labels and the collection UI remain open.
- Evidence: [watcher live validation](results/watcher-live-validation.json), [no-op acceptance](results/no-op-acceptance.json), and [operation/limitations](docs/WATCHER.md).

## Source-driven cleanup before backfill

- [x] Watcher PR #5 merged after green CI; discover-mode run [43499469228](https://github.com/LeoRigasaki/archwatch/actions/runs/43499469228) succeeded and opened [PR #6](https://github.com/LeoRigasaki/archwatch/pull/6) with `needs-review`. Its discovery state is not merged or treated as verified model data.
- [x] Gallery inventory, title, introduction and pinned source links derive from model records and seen state; no named demo-model list remains in `build_site.py`.
- [x] Family rules added for Mistral, Gemma, Llama, Phi, GLM and Nemotron.
- [x] Hub-tagged fine-tunes, adapters, merges and quantized derivatives are rejected before a processing slot; historical derived records are not dedupe authorities for primary releases.
- [x] Actual config reads are tracked, with every unread leaf key reported by dotted path (including nested dictionaries/lists). Read coverage is not a claim of semantic verification.
- [x] Activation card names are parsed and pass through a publication decision before SVG rendering. Cards support HTML, Markdown pipe tables and key/value bullets, excluding fenced examples.
- [x] 83 tests pass locally. The fresh four-record rebuild exposed an Instruct card/config context disagreement now visible to the verifier; it is held, not described as a lab defect.
- [x] Cleanup [PR #7](https://github.com/LeoRigasaki/archwatch/pull/7) merged after [green CI](https://github.com/LeoRigasaki/archwatch/actions/runs/43502432850) before starting the backfill branch.

## Ten-publisher backfill

- [x] Processed all ten primary releases in `config/backfill.yaml` in two bounded batches of five. Each resolved on the Hub with pinned config, card and safetensors headers; all ten meet the publication gate with no review fields. No weight payloads or modeling code were executed.
- [x] Added tested rules for Mistral all-layer SWA; Gemma periodic/global pattern and post-norms; Phi parallel ungated FFN; GLM fused gate/up; Nemotron-H standalone Mamba-2/MLP/attention blocks; DeepSeek expert names; LFM static SiLU; Qwen/Meta FFN gates; Kimi packed gate evidence.
- [x] Added Mamba-2 mathematical cache-state bounds and separate standalone/parallel diagrams. These estimates do not claim runtime/kernel qualification.
- [x] Ten manually entered primary-source gold labels committed before family fixes (`fad28a0`), with pinned links and shape samples. Assistant curation is explicit; no label claims human review.
- [x] Development exact match improved from **117/144 to 144/144** published fields, reported per field and per model in [results/gold-exact-match.md](results/gold-exact-match.md). Missing/held records are failures, not dropped cases. This is not held-out accuracy.
- [x] Existing approved Hugging Face login supports gated Meta/Google sources. Authorization is stripped across hosts/downgrades; bounded range/header checks remain enforced and tested.
- [x] Added explicit `--reprocess` for unchanged source commits after schema/rule changes. Ordinary repeat backfill reports all ten unchanged; **39 artifact byte streams and mtimes remain identical**, including seen state/site.
- [x] Fixed the automatic PR review's predecessor consistency finding: reanalysis refreshes existing JSON/YAML/SVG formats, architecture signatures and seen state together, while preserving source-pinned lineage. A regression checks the former stale-YAML/SVG case and subsequent no-op behavior.
- [x] Gallery now derives 11 eligible model views from 13 saved records across ten publishers. Historical Liquid derivatives remain held and diagnostically available.
- [x] Browser text geometry checks found zero text/text overlaps and zero labels outside the canvas across all 11 diagrams. Phi/Nemotron layouts were visually inspected. Sliders render cache estimates from published inputs.
- [x] Backfill PR #8 merged after green CI; outdated discovery PR #6 closed without merging and `daily-update` deleted. Cron and recurring generation remain disabled.

Evidence: [processing](results/backfill-processing.json), [repeat no-op](results/backfill-no-op.json), [gold method](evals/README.md), [family rules](docs/FAMILY_RULES.md).

## Newest-release phase

- [x] Froze seven new primary-source gold labels before the first run; preserved code/source commits, label hashes and original records. Strict first-run exact match: **97/139 (69.8%)**.
- [x] Kept two already-used newest models outside the held-out denominator. Meta access arrived mid-run; its separate late-access baseline is **11/16**, with the curation timing disclosed.
- [x] Added tested native FP4/FP8 counting, metadata exclusion, main/auxiliary tensor isolation, fused expert layouts, modern attention/family facts and Llama4 MoE scheduling.
- [x] Added bounded manual newest/predecessor processing with diagnostic records, explicit source/access review and stable repeat behavior.
- [x] Adaptation reaches **138/139** plus **16/16** on Meta; DiffusionGemma's card/header parameter-scope disagreement remains withheld. These are not new held-out results.
- [x] All ten selected newest models analyzed; seven remain held for new graph/cache qualification. Llama3.2 Vision predecessor access and Mistral Large3 native config support remain unresolved.
- [x] Current inventory: 26 records, 12 eligible gallery views, 28 tracked sources; 114 offline tests pass.
- [ ] Diagram route comparison on the four requested established records, in a separate PR. Gemini credentials are configured but no generation loop/explainer is authorized.
- [ ] Optional Discord watcher notification, in its own small PR; credential stored without printing it.

Evidence: [newest protocol and review limits](docs/NEWEST_BACKFILL.md), [first-run score](results/newest-first-run/heldout-exact-match.md), [adaptation score](results/newest-adaptation.md).
