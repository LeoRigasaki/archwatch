"""Optional, best-effort watcher summaries. No credentials enter saved reports."""

import hashlib
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .watcher import valid_repo

LOG = logging.getLogger(__name__)
FAILURE_MESSAGE = "Discord notification failed; watcher results are unaffected."
PROCESSED = {"recorded", "duplicate", "ready_for_diagram"}


def snapshot(root):
    """Hash only watcher artifacts, excluding caches and ephemeral run reports."""
    root = Path(root)
    paths = [root / "data/seen.json"]
    paths += sorted((root / "data/models").glob("*.*"))
    paths += sorted((root / "diagrams").glob("*.svg"))
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    state = root / "data/seen.json"
    return dict(files=files, models=json.loads(state.read_text()).get("models", {}) if state.exists() else {})


def annotate_report(root, report, before):
    """Attach actual content changes and only the model events from this run."""
    after = snapshot(root)
    changed = {key for key in before["files"] | after["files"] if before["files"].get(key) != after["files"].get(key)}
    events = []
    for repo in sorted(before["models"] | after["models"] | report["outcomes"]):
        stem = repo.replace("/", "__")
        model_files = {f"data/models/{stem}.json", f"data/models/{stem}.yaml", f"diagrams/{stem}.svg"}
        if before["models"].get(repo) == after["models"].get(repo) and not model_files.intersection(changed):
            continue
        entry = after["models"].get(repo, {})
        reason = entry.get("reason", "")
        events.append(
            dict(
                id=repo,
                status=entry.get("status", report["outcomes"].get(repo, "removed")),
                processed=report["outcomes"].get(repo) in PROCESSED,
                access_blocked=bool(re.search(r"\b(?:401|403)\b|gated|access.blocked", reason, re.I)),
            )
        )
    return report | dict(changed=bool(changed), model_events=events)


def format_message(report, pr_url=""):
    if not report.get("changed"):
        return None
    events = sorted((e for e in report.get("model_events", []) if valid_repo(e.get("id", ""))), key=lambda e: e["id"])
    blocked = [e for e in events if e.get("access_blocked")]
    processed = [e for e in events if e.get("processed")]
    review = [e for e in events if e["status"] in {"review", "error"}]
    queued = [e for e in events if e["status"] == "pending"]
    lines = [
        "ArchWatch watcher update",
        f"Processed: {len(processed)} | Sent to review: {len(review)} | Access blocked: {len(blocked)} | Queued: {len(queued)}",
    ]
    if report.get("discovery_errors"):
        lines.append(f"Discovery errors: {len(report['discovery_errors'])} (see run log)")
    # Keep the PR link intact even when a large discovery result must be shortened.
    suffix = ""
    if re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/\d+", pr_url):
        suffix = f"\nPR: <{pr_url}>"
    details = []
    for e in blocked:
        repo = e["id"]
        details.append(f"{repo} — [request access here](<https://huggingface.co/{repo}>)")
    for label, items in (("Processed", processed), ("Review", review)):
        for e in items:
            if e not in blocked:
                details.append(f"{label}: {e['id']} — <https://huggingface.co/{e['id']}>")
    for index, detail in enumerate(details):
        text = "\n".join(lines + [detail])
        if len(text + suffix) > 1900:
            lines.append(f"… {len(details) - index} more model details; see the PR or run log.")
            break
        lines.append(detail)
    return "\n".join(lines) + suffix


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def notify(report, pr_url="", *, environ=None, opener=None):
    """Send at most once. Missing secrets/no changes are silent; failures are redacted."""
    if not (os.environ if environ is None else environ).get("DISCORD_WEBHOOK_URL"):
        return False  # unconfigured: do not even format the report
    try:
        content = format_message(report, pr_url)
    except Exception:
        LOG.warning(FAILURE_MESSAGE)
        return False
    return send_text(content, environ=environ, opener=opener) if content is not None else False


def send_text(content, *, environ=None, opener=None):
    """Post one message to the configured Discord webhook. Missing secret is silent; failures are redacted."""
    webhook = (os.environ if environ is None else environ).get("DISCORD_WEBHOOK_URL")
    if not webhook:
        return False
    try:
        parsed = urlsplit(webhook)
        if (
            parsed.scheme != "https"
            or parsed.netloc not in {"discord.com", "discordapp.com"}
            or not re.fullmatch(r"/api/(?:v\d+/)?webhooks/\d+/[A-Za-z0-9._-]+", parsed.path)
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid webhook")
        request = urllib.request.Request(
            webhook + "?wait=true",
            data=json.dumps(dict(content=content[:1900], allowed_mentions=dict(parse=[]), flags=4)).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "ArchWatch"},
            method="POST",
        )
        send = opener or urllib.request.build_opener(NoRedirect()).open
        with send(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise OSError("Webhook rejected")
        return True
    except Exception:
        LOG.warning(FAILURE_MESSAGE)
        return False
