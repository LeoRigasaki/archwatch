# Implementation sequence approved on 14 November 2026

One PR per step. Update `PROGRESS.md` as each step lands. Keys belong in ignored local `.env` files and GitHub Actions secrets, never source control. Free tiers only. Do not add co-author attribution to commits or PRs. Cron stays commented out until the user explicitly approves it, including after successful manual runs.

## 1. Foundations merge and manual acceptance

Push `fix/prototype-foundations`, open a PR, and merge only after its new CI passes at the current head SHA. Create the `needs-review` and `auto-merge` labels before manually dispatching `daily.yml` from merged main. Confirm that the workflow completes and opens an update PR; a label does not enable auto-merge. Report the merge, CI and workflow/PR links before implementing later steps. Create the watcher branch from merged main.

The patch fixes LFM effective FFN normalization, refreshes repo metadata per analysis, and gates rendering/comparisons on published values and a verified weight-kind layer pattern. It includes uv locking, ruff, pre-commit, offline PR CI and a source-only ZIP exporter. The four source commits and parameter totals are unchanged; the original 81 core verdicts reproduce except for the corrected LFM FFN field.

## 2. Watcher PR

Use the ten core publishers and outside discovery in [MODEL_SELECTION.md](MODEL_SELECTION.md). Before queueing, resolve each repository on the Hub and require `config.json` plus safetensors files. Skip conversions and fine-tunes whose normalized architecture matches an existing record, with an explicit reason. Unsupported or conflicting required facts go to review rather than diagram generation.

Track last-observed and last-successful commits, processing outcomes and retry state in `data/seen.json`. Choose predecessors from the same organization and family using the previous release, with deterministic tie/ambiguity handling and source dates. Keep all file/header reads pinned to the resolved commit. Test discovery, admission, duplicates, predecessor choice, commit updates, failures, rate limits and restart behavior. Keep cron disabled pending user approval.

## 3. Diagram route-test PR

Test both routes against all four existing published records:

- A: Gemini writes the full SVG.
- B: Gemini writes a small JSON layout specification (nodes, groups, labels and edges); Python draws the SVG.

Gemini only arranges published data. Automatically check every architectural label/number against the published fields, text-box overlap, canvas bounds and layer totals. Record scores, invalid outputs and request counts in `results/diagram-route-test.md`; select the route from measured results. A tie or invalid routes must not be described as a winner without evidence. Both routes and validators need tests. Count all generation/repair requests against the configured budget.

## 4. Gemini loop and weekly queue PR

Store model names and limits in configuration, not implementation code:

- Diagrams: `gemini-4-hydrogen`; free fallback `gemini-3.8-flash`, each screenshot quota 20 requests/day.
- Reviews and explainer text: `gemini-3.5-flash-lite`, screenshot quota 500 requests/day. The diagram generator never grades its own work; automatic validators remain authoritative.

Implement a weekly cap of five architectures, one generation plus at most two repairs each: 15 diagram requests total, including fallback requests. Cache by published-spec hash, evidence revisions and prompt/model version; honor RPM/TPM/RPD and existing use. Defer on exhaustion and never switch to paid models. Use standard requests rather than the separately priced provider Batch API.

Save every round: `spec.yaml`, `v1.svg`, `v1.review.json`, `v2.svg`, and subsequent attempts, including route-B layout JSON when applicable. Preserve the deterministic renderer as fallback. Test budget/retry accounting, validation, caching, artifact persistence and failure/restart behavior. Do not enable cron without approval.

## 5. Explainer PR

For each new model with a predecessor, generate what-changed text from the computed diff JSON only, using the configured review/text model. Reject publication if any number in the text is absent from the diff. Test valid summaries, invented numbers, missing predecessor, empty diffs and quota failures. Any-pair explanations remain an on-demand feature; automatic batches use predecessor pairs.

## 6. Gold-set PR

As core models are processed, manually inspect their primary `config.json`, safetensors headers and model card to write independent YAML labels in `evals/gold/`. Never derive labels from pipeline output or another gallery. Target ten models in this phase, with pinned source links and explicit uncertainty. Report per-field exact match in `results/`; agreement counts are not gold-set accuracy. The longer-term independent set can expand to 30 later. Add evaluation/fixture tests and prevent pipeline output from silently becoming ground truth.

The watcher is implemented and locally/live tested (see [WATCHER.md](WATCHER.md)). Route evaluation, provider adapter, weekly queue, explainer and the ten-model gold set remain unimplemented. Current provider evidence is in [PROVIDER_LIMITS.md](PROVIDER_LIMITS.md).
