# Optional Discord watcher summary

This integration sends one text-only summary after a manual `discover` or `process` run changes watcher artifacts. No schedule is enabled. `DISCORD_WEBHOOK_URL` is an optional environment variable; a missing variable skips all report reads and network calls silently. The credential is stored in ignored local `.env` and GitHub secrets, never in a report, command-line argument or tracked file.

`watch.py --report PATH` compares content hashes of `data/seen.json`, model records and diagrams before/after its run. Caches, report files and modification times do not count. The ephemeral report carries only this run's changed model events. Old review items are not repeatedly announced on no-op runs. In Actions the report lives under `$RUNNER_TEMP`, outside the checkout.

The message gives processed/review/access-blocked/queued counts, names and Hub links for processed/review cases, and a “request access here” Hub link for gated/HTTP 401/403 cases. Predecessors implicitly analyzed/refreshed during successor processing are included as processing outcomes. If the PR action has a PR URL, it is included. Large lists are shortened to fit one message under 2,000 characters, retaining the PR link. These are watcher outcomes, not a claim that held architectures have been verified.

Only the notification step receives the webhook secret. It runs after the PR action, even if another step failed after the watcher wrote its report. A missing/unreadable report or webhook failure logs one generic line without exception details or the URL and exits successfully. Delivery is best-effort: there is one HTTP attempt and no retry, since retrying a timed-out request could duplicate a message. Redirects are refused. No attachments are sent; embeds and mentions are disabled. Payload/confirmation behavior follows the [official Discord execute-webhook API](https://docs.discord.com/developers/resources/webhook#execute-webhook).

For a deliberate local send after a changed watcher run:

```bash
uv run --frozen python watch.py --process --report /tmp/archwatch-report.json
uv run --env-file .env --frozen python tools/notify_discord.py --report /tmp/archwatch-report.json
```

Tests use a fake transport; development/CI does not send test messages to the channel. Cron and automatic merges remain disabled.
