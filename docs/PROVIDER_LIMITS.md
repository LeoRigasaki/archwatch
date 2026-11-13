# Provider constraints checked on 13 November 2026

## Gemini API

The user supplied the account's AI Studio quota screenshots on 13 November 2026. The relevant text-model limits are transcribed below; this is screenshot evidence, not an authenticated API measurement.

| Model | RPM | TPM | RPD | Proposed use |
|---|---:|---:|---:|---|
| Gemini 4 Hydrogen | 5 | 250,000 | 20 | Primary layout/SVG-code generator |
| Gemini 3.8 Flash | 5 | 250,000 | 20 | Explicit free fallback candidate |
| Gemini 3.5 Flash Lite | 15 | 250,000 | 500 | Required separate model for reviews and explainer text |
| Gemini 3.1 Flash Lite | 15 | 250,000 | 500 | Available alternative |
| Gemini 2.5 Flash | 5 | 250,000 | 20 | Older available alternative |

Exclude screenshot entries with zero capacity. Google's [pricing page](https://aistudio.google.com/docs/pricing#gemini-4-hydrogen) confirms free text input/output for `gemini-4-hydrogen`, and its [model catalog](https://aistudio.google.com/docs/models) confirms the model ID. The recommendation is provisional until the four existing architecture specifications have been evaluated for content fidelity and layout quality; no generator call has been made yet.

Free access depends on model and project. Limits are project-scoped, not per API key; daily request quotas reset at midnight Pacific time. Recheck active limits before the first run. [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), [active project quotas](https://aistudio.google.com/rate-limit).

Use a weekly application queue of at most five changed architectures, up to three requests each, with a 15-request total cap and conservative request/token throttling. Existing usage can reduce headroom. Use standard requests: the separately priced provider Batch API is not required. Cache outputs and defer jobs on exhaustion; do not upgrade or fall back to paid models. See [MODEL_SELECTION.md](MODEL_SELECTION.md).

## Groq API

The published free table currently lists `openai/gpt-oss-20b`, `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` at **30 RPM, 1,000 RPD, 8,000 TPM and 200,000 TPD** each. Limits apply at organization level; account-specific exceptions and separate input/output limits can apply. Inspect the organization's limits page and response headers. A token budget can be exhausted before the request-count limit. [Official rate limits](https://console.groq.com/docs/rate-limits), [organization limits](https://console.groq.com/settings/limits).

Use compact verified diffs for explanations, output caps, caching, and `retry-after` handling. These published values are a planning reference, not a measurement of this account.

## Codex scheduled tasks

Project-scoped desktop tasks need the computer powered on, the app running and the project present on disk. They can use the local project or a worktree. The official task page does not establish a universal active-task maximum, so none is assumed here. [Official scheduled-task documentation](https://learn.chatgpt.com/docs/automations?surface=app).

Usage depends on plan, model and task, with local/cloud work sharing allowances and possible weekly limits. Check the account's usage dashboard for its current allowance. [Official pricing and usage](https://learn.chatgpt.com/docs/pricing).

Recommendation: GitHub Actions owns the daily deterministic refresh. Codex can review a bounded diff or review queue optionally; it is not a required dependency of the unattended data pipeline. No scheduled task was created by this documentation update.
