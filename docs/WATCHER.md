# Watcher operation

The watcher uses the official Hub API and the configurable ten-publisher/outside policy in `config/watcher.yaml`. It does not invoke Gemini, create a weekly generation job, download weight payloads or execute modeling code.

```bash
# Discover and checkpoint admitted analysis candidates.
uv run --frozen python watch.py
# Analyze at most five admitted candidates (initial allocation: three core, two outside).
uv run --frozen python watch.py --process
# A controlled explicit selection, bypassing popularity but never source/publication checks.
uv run --frozen python watch.py --repo LiquidAI/LFM3-1.7B-Base --process
# Revisit held records after investigating access, unsupported rules or retry exhaustion.
uv run --frozen python watch.py --process --retry-review
# Deliberately reanalyze unchanged commits after extractor/schema improvements.
uv run --frozen python watch.py --repo microsoft/phi-2 --process --reprocess
```

`--root` selects a separate output folder for a live smoke test. The checked-in `data/seen.json` contains the original records plus the ten-publisher primary backfill. Discovery-only PR #6 remains a separate unmerged observation set; no outside model from that scan is published by this backfill.

## Admission and publication

Listings are discovery hints. Each candidate must resolve on the Hub, have root `config.json` and safetensors files, and expose a readable JSON config before entering the pending analysis state. Quantization-only GGUF/ONNX repositories are rejected. Access-gated sources are held for review; this code does not accept licenses or bypass gates.

Explicit Hub `base_model` finetune/adapter/merge/quantized relations are rejected before analysis slots. Historical derived records are excluded from the dedupe authority and gallery. Approved gated sources use an existing local Hub credential, with credential stripping on cross-host redirects; the code does not approve access gates.

Analysis uses the exact admitted commit for config, README, code and tensor headers. Required architecture fields and layer-pattern publication checks still apply. Only successful records receive generated YAML/SVG outputs. The gallery inventory is built from records and seen state; a richer collection UI is a later step.

A diagram-eligible architecture signature includes published structural fields, normalized primary config (including unknown keys) and exact custom source code. Packaging dtype/quantization/provenance keys do not create a new architecture. Matching fine-tunes and conversions are marked `duplicate`, linked to the existing record and get no new record or diagram. Matching published dimensions with different unmodeled config/code go to review rather than silently deduplicating or drawing an invented mechanism. This is conservative and can hold a harmless packaging/code difference for investigation.

## Predecessors

Choose the latest earlier Hub creation date among inspected same-organization, same-family candidates and recorded history. Family rules/overrides are configurable; separate VL/Omni/Coder name streams are kept apart. Missing dates and equally recent alternatives produce an explicit unresolved reason. The primary backfill retains Kimi-K3.1-Pro -> Kimi-K3. The historical LFM Instruct pair is retained as diagnostic history but excluded from new primary-release predecessor selection.

Hub creation time is a release-order proxy, and discovery is bounded. This is not a guarantee of complete historical release coverage. The selected ID, source commit and selection reason remain in the record for inspection. Family overrides or broader backfill can correct a sparse/ambiguous catalog. A predecessor that changes commit between selection and analysis defers the target instead of mixing revisions.

## State and retries

`data/seen.json` separates `last_observed_commit` from `last_successful_commit`. `pending` means source admission passed, not architecture verification. Other states are `recorded`, `ready_for_diagram`, `duplicate`, `review`, `rejected`, `deferred` and `error`. Diagram eligibility is not a completed LLM generation job; the weekly queue remains a separate step.

State is checkpointed atomically and written only when changed. Pending/errors survive disappearance from the newest/trending lists; previously seen repos are revisited for upstream commits. With more than one run's bound, half the admission slots revisit retained work/history using an unpersisted UTC-day rotation; the rest allow new discovery. No cursor/timestamp-only updates are committed. Retry counts prioritize unattempted pending work and move repeated failures to review after three attempts at a commit. New commits reset the attempt budget.

Requests honor bounded retries and Retry-After values; longer deferrals persist across restart. Discovery failures and per-model errors remain visible in the report/state. Per-model failures do not discard other candidates' successful checkpoints. A wholly failed discovery returns a nonzero CLI exit; a completed batch's zero exit does not mean every model passed verification.

## Manual GitHub workflow

`daily.yml` has manual modes `fixed` (default smoke test), `discover`, and `process`. Generated update PRs are labeled `needs-review` if the state or records contain review/error items. Neither the `auto-merge` label nor zero review items merges a PR. Merge accepted state/data PRs before relying on that state in subsequent Actions runs.

Cron remains commented out until the user explicitly approves it. No scheduled task was created. Provider generation, route testing and explanations remain unimplemented. Ten development gold labels and their evaluation are documented in `evals/README.md`; human review and held-out evaluation remain open.

Validation: `tests/test_watcher.py`, `results/watcher-live-validation.json` and `results/no-op-acceptance.json`. Behavior tests and live source agreement are not independent gold-set accuracy scores.
