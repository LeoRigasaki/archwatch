# ArchWatch progress

As of **18 November 2026**. The dated sections under [History](#history) are kept as they were written.

## Local review pending

Uncommitted on local `main`: the #40/#41 trust and sorting changes (excluding frozen first-run record edits), an original website redesign, 32 full concept guides with worked examples and original illustrations, evidence/coverage page, two/three-model selection tray, published-only JSON/CSV exports, diagram expansion/download, custom device/sequence memory planning and an analysis RSS feed. The live deployment and remote PRs are unchanged. See [local review notes](docs/LOCAL_SITE_REVIEW.md) and [validation](results/local-site-review.json).

Local validation: **318 tests passed**, Ruff passed, 1,231 local links checked, repeated builds preserved bytes and modification times. Gold labels and frozen first-run evidence are unchanged; existing published values remain unchanged, with optional concept facts added from pinned configs/code.

Concept follow-up: all 28 requested concepts plus four companion topics now have full guides; 16 requested concepts have matches in published records, six only in held records, and six await qualified record evidence. Compare follows the supplied mockup with working Plain/Technical mode, model cards, filtering and cache bars. See [implementation and limits](docs/CONCEPT_IMPLEMENTATION.md).

## Project status

- **Site:** live at [archwatch.vercel.app](https://archwatch.vercel.app), deployed by Vercel on every merge to `main`, built offline from committed records (`site/` is not committed). 13 published models, an explore page with filters, one page per model, and a compare tool for 2-3 models with shareable links.
- **Records:** 26 records, 28 tracked repositories. Seven newest architectures and two fine-tunes are held; fine-tunes are out of scope and not listed.
- **Watcher:** daily GitHub Action (no LLM calls). Predecessors must share the family and size class (total within 2x, same dense/MoE kind, base with base).
- **Diagrams:** Monday/Wednesday/Friday Gemini runs, at most 5 models and 15 calls, each checked by code. 5 published models show an approved Gemini diagram; the rest show a code-drawn one. Outputs and the call ledger are kept in a needs-review PR and an artifact before the tests run.
- **Counting convention 2** (`config/counting-conventions.yaml`): total excludes MTP modules (published as `params_mtp`); active excludes unselected experts, MTP, lookup tables and the input embedding, keeps the output head.
- **Dates:** `hub_created_at` is the Hub creation time; a release date is shown only when the model's own card states one.
- **Explainer:** notes are checked field by field (each claim's before and after belong to that field); none is on the site yet.
- **Tests:** the locally integrated version passes 318 offline tests, including real Chrome checks; this count includes changes still awaiting review.

## Evaluation

| Set | Score | What it is |
|---|---:|---|
| Development gold (10 models) | 144/144 | Adaptation score on labels frozen before family fixes |
| Newest held-out, first run (7 models) | 97/139 (69.8%) | The held-out result; frozen in `results/newest-first-run/` |
| Same labels, current records as published (convention 2) | 134/139 | 4 misses are MTP modules now reported separately; 1 is DiffusionGemma's held total |
| Same labels, current records mapped to convention 1 | 138/139 | Like for like with the labels (total = published total + MTP) |

The gold labels are never edited; `tools/evaluate_gold.py --convention 1` maps records back to the labels' convention. Reports: [convention 2](results/heldout-current-v2.md), [convention 1](results/heldout-current-v1.md). No label is human-reviewed, and the held-out 30-model evaluation and the 95% target remain open.

## Open work

- [ ] Drawing and cache rules for the held architectures (compressed/sparse attention with hyper-connections, shared KV, chunked attention, block diffusion, recurrent layers).
- [ ] Explainer notes on the site, regenerated under the change filter and the field-bound validator.
- [x] Concept guides and an analysis RSS feed implemented locally, awaiting review.
- [x] Plain/Technical comparison mode implemented locally.
- [ ] Reading-mode toggle for standalone model pages.
- [ ] Human-reviewed held-out evaluation set.
- [ ] Typed schema; LLM code/report readers with citations; review issue queue; MCP server.

<a id="history"></a>
# History

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

## Reference boundary

- [x] Third-party diagrams remain outside production records and the gold set.
- [x] An external reference sheet may be used as a comparison only: named at run time, downloaded into the ignored
  cache, never committed, and never a source for published fields or gold labels.

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
- [x] Ten manually entered primary-source gold labels committed before family fixes (`0fa673f`), with pinned links and shape samples. Assistant curation is explicit; no label claims human review.
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
- [x] Current inventory: 26 records, 12 eligible gallery views, 28 tracked sources; 116 offline tests pass.
- [ ] Diagram route comparison on the four requested established records, in a separate PR. Gemini credentials are configured but no generation loop/explainer is authorized.
- [x] Optional Discord watcher notification implemented in its own branch; credential stored without printing it. Awaiting PR review/merge before workflow activation.

Evidence: [newest protocol and review limits](docs/NEWEST_BACKFILL.md), [first-run score](results/newest-first-run/heldout-exact-match.md), [adaptation score](results/newest-adaptation.md).

## Optional Discord notification

- [x] Ephemeral watcher report records actual artifact content changes and current-run model events; no-change runs send nothing.
- [x] One bounded text message includes processed/review models, access-request Hub links and the PR URL when available. No attachments, embeds or mentions.
- [x] Missing `DISCORD_WEBHOOK_URL` skips silently; one failed attempt logs a generic redacted line and does not fail the workflow. The secret is scoped to the notification step only.
- [x] 128 tests pass, including formatting, no-op/missing-secret silence, webhook errors, CLI report/no-op behavior and workflow secret scope. Fake transport only; no test message sent to Discord.
- [ ] Merge notification PR and validate delivery during a deliberately changed manual watcher run. Cron remains disabled.

Evidence: [notification operation and limits](docs/DISCORD_NOTIFICATIONS.md), `tests/test_notifications.py`.
