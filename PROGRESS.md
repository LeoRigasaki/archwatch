# ArchWatch progress checklist

As of **14 November 2026**, after the prototype import, foundations merge and successful manual GitHub Actions acceptance. This checklist describes verified implementation and remaining acceptance; the original day numbers are planning milestones.

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
- [x] **41 offline tests pass**, including publication boundaries, normalization, refresh, packaging, AST and original formulas/diffs.
- [x] Fresh primary-source run reproduces all four commits and parameter totals. Of the original 81 checks, only the corrected LFM FFN verdict changes.
- [x] GitHub repository configured at `LeoRigasaki/archwatch`.
- [x] Foundations merged in [PR #1](https://github.com/LeoRigasaki/archwatch/pull/1) after [CI passed](https://github.com/LeoRigasaki/archwatch/actions/runs/43489941541) on its exact head commit; main CI also passed.
- [x] `needs-review` and `auto-merge` labels created; repository setting permits Actions to open PRs. Actual auto-merge remains disabled.
- [x] The [manual workflow](https://github.com/LeoRigasaki/archwatch/actions/runs/43490777972) succeeded from merged main: four records analyzed, 38 tests passed, zero review fields and [update PR #3](https://github.com/LeoRigasaki/archwatch/pull/3) opened with `auto-merge`. That PR remains open; the label does not merge it.
- [x] The initial run exposed duplicate Git auth headers; [compatibility PR #2](https://github.com/LeoRigasaki/archwatch/pull/2) passed CI and merged before the successful rerun.
- [x] Watcher branch `feat/model-watcher` prepared from merged main `f93be2c`; watcher implementation has not started.
- [x] Acceptance evidence and four-record equality checks recorded in [github-workflow-acceptance.json](results/github-workflow-acceptance.json).
- [x] Progress/result paths remain relative; research reference remains ignored.

## Decisions recorded, implementation still open

- [x] Preferred publishers: DeepSeek, Alibaba/Qwen, Mistral, Liquid AI, NVIDIA, Google, Meta, Microsoft, Moonshot and Z.ai.
- [x] Keep an outside-discovery route for emerging models, with architecture novelty and source readiness checked before diagram generation.
- [x] Proposed generator: Gemini 4 Hydrogen, with Gemini 3.8 Flash as an explicit free fallback. Account quotas are transcribed from the supplied screenshots.
- [x] Proposed weekly batch: at most five changed architectures, initially three preferred-lab slots and two outside slots; no regeneration for unchanged specifications.
- [x] Codex recommended as optional reviewer; explanations available for any pair on demand, predecessor pairs in automated batches.
- [ ] Implement and manually test these selection/provider policies; documentation is not a running integration.

## Still open

- [ ] Watcher: lab allowlist, global discovery, popularity filters with new-release exceptions, dedupe, retry state and `data/seen.json`.
- [ ] Typed/Pydantic schema and broader family/config acceptance coverage.
- [ ] Independent **30-model gold set**, beginning with 15 hand-labeled models; config/calculator/citation metrics and the **95% config-field target**.
- [ ] Gemini validated spec-to-SVG generation/repair loop and response caching; provider quota controls.
- [ ] LLM code reader with structured output and source-line citations.
- [ ] Report finder/retrieval, report agent, page citations and citation validator.
- [ ] LangGraph orchestration and Langfuse traces.
- [ ] GitHub review issue queue and recorded human decisions.
- [ ] Automatic family/predecessor selection and diff-constrained explanations.
- [ ] Controlled scheduled publishing/merging; cron remains disabled until explicit user approval.
- [ ] MCP server (`search_models`, `get_model`, `compare`) and client tests.
- [ ] Full collection/filter/compare UI, own RSS, final model pages and production deployment.
- [ ] **50+ model backfill**, independent evaluation results and demo recording.

The existing refresh workflow uses a fixed watchlist and can be dispatched manually. Its cron remains disabled; neither a labeled PR nor zero conflicts constitutes automatic merge authorization.

## Milestone map

| Day(s) | Current state |
|---|---|
| 1 | uv/ruff/pytest/pre-commit/PR CI configured and CI passed remotely; typed schema remains open |
| 2 | Watcher/seen state next |
| 3–4 | Core extractor implemented; four live models are narrower than the planned family acceptance sets |
| 5 | Cache formulas and hand-math tests implemented; broader/runtime validation remains |
| 6 | First 15 gold labels missing |
| 7 | AST scanner and line evidence implemented |
| 8 | LLM reader missing |
| 9 | Verifier and consumer publication boundary implemented; confidence remains heuristic |
| 10–12 | Gold evaluation, metrics, tracing and failure analysis open |
| 13–15 | Report retrieval and page-citation validation open |
| 16–18 | Deterministic diffs exist; predecessor discovery, explanations and issue queue open |
| 19–22 | Two-model site, deterministic SVG and cache UI exist; full collection/any-pair UI and RSS open |
| 23 | Manual fixed-list workflow passed and opened an update PR; discovery and controlled scheduled publishing remain open |
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

Next: build the watcher on `feat/model-watcher`, following [NEXT_STEPS.md](docs/NEXT_STEPS.md) using the [selection policy](docs/MODEL_SELECTION.md).

The implementation amendments approved on 14 November are recorded in [NEXT_STEPS.md](docs/NEXT_STEPS.md): CI-gated foundations merge and manual workflow acceptance first; then separate watcher, route-test, Gemini-loop, explainer and ten-model gold-set PRs. Reviews/explanations use Flash Lite, diagram routes must be compared, and cron requires explicit approval.

## No-op update correction (14 November)

- [x] Closed PR #3 without merging and deleted `daily-update`; its changes were timestamps and equal-count ordering only.
- [x] Tensor groups and parameter-component ties use count plus name ordering; code/file scans, representative shapes and cache rows have stable ordering. Layer sequence order remains architectural data.
- [x] JSON comparison ignores only `analyzed_at`, preserving the saved record/date when unchanged. YAML, SVG and HTML writes preserve existing files when unchanged.
- [x] Cached end-to-end regression checks identical bytes and modification times, changing-input updates, and cross-process hash-seed stability. All 41 tests pass.
- [x] Two local four-model runs leave every artifact byte and modification time unchanged after the one-time ordering migration.
- [ ] Merge this correction on green CI, then dispatch `daily.yml` twice and require no update PR from either run before resuming the watcher.
