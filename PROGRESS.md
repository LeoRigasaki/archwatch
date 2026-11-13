# ArchWatch progress checklist

As of **13 November 2026**, after importing and testing the user-supplied `archwatch.zip`.

The earlier checklist described the newly created folder and missed this implementation. The supplied prototype is now the primary code baseline. This is approximately the **Day 9 deterministic-core stage**, with watcher/tooling/evaluation gaps still open and later diff/SVG/site features already present. It is not a claim that nine complete daily milestones or the full five-week plan have passed acceptance.

## Imported and validated

- [x] Application code integrated into `src/archwatch/`.
- [x] Supplied tests integrated into `tests/`; meaningful offline migration, AST, verifier, diff, SVG and Hub-boundary checks added.
- [x] **Four JSON records and two YAML records** integrated into `data/models/`.
- [x] Original record-generated SVG/PNG diagrams integrated into `diagrams/`.
- [x] CLI, fixed watchlist, static site builder and supplied daily workflow integrated.
- [x] Minimal editable src-layout packaging added, with dependencies installed in a local Python 3.12 environment.
- [x] **22 offline tests pass**; the two-model page and both SVGs rebuild from saved records.
- [x] Cache root updated for the src layout; future file/header requests now use the recorded commit and revision-specific cache paths.
- [x] Weight-header requests reject full-body responses and unsupported header lengths.
- [x] Progress/result paths made relative; generated results are in `results/`.

## Implemented prototype components

- [x] **Config extractor:** nested `text_config`, dense/hybrid/MoE, latent MoE, MLA, DSA, KDA, layer patterns and one-based layer-list handling.
- [x] **Weight counting:** exact shape-based totals for supported formats, scale exclusion, component counts and explicit active-parameter conventions. The archive contains counts for all four saved models.
- [x] **AST analyzer:** class/pattern scanning with `file#line` evidence, without executing downloaded code. The separate LLM reader is open.
- [x] **Calculator:** GQA, sliding-window, short-conv, MLA, DSA and KDA cache formulas, hand-math tests and explicit assumptions.
- [x] **Verifier:** verified/single-source/conflict states, confidence thresholds and conflicts excluded from the published field dictionary.
- [x] **Diff:** deterministic field diff and tensor-group diff for explicitly supplied predecessor pairs.
- [x] **DSA result:** saved Kimi records account for an exact **178,526,208-parameter** indexer addition.
- [x] **SVG renderer v1:** architecture diagrams generated from normalized records.
- [x] **Static demo:** two successor models, evidence tables, predecessor/tensor diffs and interactive cache estimates.
- [x] **Workflow file:** scheduled/manual fixed-watchlist pipeline and labeled PR creation written. It has not been run or connected to a GitHub repository.

## Still open

- [ ] Watcher: lab allowlist, new-model discovery, download filtering, dedupe and `data/seen.json`.
- [ ] Tooling completion: locked dependencies/uv project workflow, ruff, pre-commit and dedicated PR CI.
- [ ] Typed/Pydantic schema validation and broader config/family acceptance coverage.
- [ ] Independent **30-model gold set**, exact-match/calculator/citation metrics and the **95% config-field target**.
- [ ] LLM code reader with structured output and line citations.
- [ ] Model-card-to-report finder, report agent/retrieval, page citations and citation validator.
- [ ] LangGraph orchestration and Langfuse traces.
- [ ] GitHub issue review queue and human decisions.
- [ ] Automatic family/predecessor selection and constrained LLM change notes.
- [ ] Strict publication checks in all site/SVG consumers, including conflicted/incomplete target records and unsupported cache cases.
- [ ] Repository connection and a real run of the supplied workflow; explicit auto-merge controls beyond a label.
- [ ] MCP server with `search_models`, `get_model`, `compare` and client tests.
- [ ] Full model-list/filter/compare UI for the new primary-source records, own RSS and final model pages.
- [ ] **50+ verified-model backfill**, gold-set results, production deployment and demo recording.

## Corrected milestone map

| Day(s) | Status | Evidence and remaining acceptance |
|---|---|---|
| 1 | Partial | Installable src package, pytest, README and workflow file exist; locking, ruff, pre-commit, typed schema and PR CI remain |
| 2 | Open | Fixed watchlist only; watcher/seen state not built |
| 3–4 | Core implemented | Dense/hybrid/MoE/nested config extraction works in the prototype; saved coverage is four models, not the planned 5/10 acceptance sets |
| 5 | Implemented prototype | Cache formulas and hand-math tests exist; broader family/runtime coverage remains |
| 6 | Open | First 15 independently hand-labeled gold models missing |
| 7 | Implemented | Static AST scanner and file/line citations |
| 8 | Open | LLM code reader missing |
| 9 | Implemented prototype | Agreement/conflict confidence and record publication withholding; consumer-side hardening remains |
| 10–12 | Open | Gold-set eval harness, formal metrics, tracing and failure analysis missing; existing counts are prototype evidence |
| 13–15 | Open | Report finder, retrieval agent and page-citation validator missing |
| 16 | Diff implemented | Field and tensor-group diffs; automatic predecessor selection missing |
| 17 | Open | Diff-only LLM change notes and hallucination checks missing |
| 18 | Open | Review fields exist in records; GitHub issue queue missing |
| 19 | Partial | Own two-model static page builds; full collection filtering still to integrate |
| 20 | SVG v1 implemented | Own generated diagrams; final standalone model pages/broader rendering still open |
| 21 | Partial | Deterministic predecessor diff exists; arbitrary-pair compare UI for the new records remains |
| 22 | Partial | Interactive cache calculator exists; own RSS missing |
| 23 | Written, unrun | Daily/manual fixed-list workflow opens labeled PRs; repository, discovery and controlled merge behavior remain |
| 24 | Open | MCP server/client tests missing |
| 25 | Open | Four saved records are not a 50+ verified backfill; final deployment/eval/demo work remains |

## Results and validation boundaries

The imported records contain **60 verified checks, 20 single-source checks and one conflict**. Those are the prototype's rule-based statuses, not a gold-set accuracy score. The newly run test suite is offline; it validates behavior, formulas, saved diffs, conflict withholding, generated SVGs, revision URLs and bounded header reads. It does not re-download the original model sources or headers.

The saved Kimi parameter totals are 2,779,931,837,184 and 2,780,110,363,392; their difference is exactly the recorded DSA-indexer group addition. The LFM totals are 1,170,340,608 and 1,728,801,536. Reproducible summaries are generated by `tools/summarize_prototype.py` into `results/` and `docs/progress-snapshot.json`.

`verify.py` withholds conflicting fields in records. The general SVG/site code still falls back to raw facts for some inputs, so that consumer boundary needs hardening before general publishing. Current displayed successor records have zero conflicts. Unknown layer/cache cases also need explicit release handling.

## LFM findings retained

- The imported `LiquidAI/LFM2.5-1.2B-Instruct` record has a held FFN-width conflict: config 12,288 versus tensor-shape 8,192.
- The earlier primary-source audit found model-card/config context inconsistencies for the explicitly named `LFM2.5-1.2B-Thinking` and `LFM2.5-350M` variants. This is a separate finding, not the same checkpoint or field.
- Details: `docs/LFM_FINDINGS.md`.

## Private reference deployment

- [x] Hosting policy checked: owner only, no other users/editors/groups/external visitors.
- [x] Unauthenticated requests to the page, a diagram asset and model JSON each returned **HTTP 401**.
- [x] Legacy deployment link removed from this project's README; none was found in the checked portfolio projects.
- [x] Legacy gallery/third-party diagrams retained only as ignored private research references, not production records or the independent gold set.

Evidence: `docs/deployment-access-review.json`. No new public deployment or GitHub workflow run was performed during integration.

## Files to start from

- `src/archwatch/`: integrated implementation.
- `tests/`: supplied and added offline tests.
- `data/models/`: four original JSON records and two original YAML records.
- `site/index.html`, `diagrams/`: rebuilt original prototype outputs.
- `results/README.md`, `results/prototype-validation.json`: generated counts and test results.
- `docs/prototype-import.json`: archive hash and relative import manifest.
- `docs/progress-snapshot.json`: corrected portable project status.

Next work is the **watcher and remaining tooling/evaluation foundations**, followed by the LLM/report/review stages. The implemented extractor/AST/calculator/verifier should be extended rather than rebuilt from Day 1.
