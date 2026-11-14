"""Best-effort post-watcher summary; credentials come from the environment only."""

import argparse
import json
import logging
import os
from pathlib import Path

from archwatch.notifications import FAILURE_MESSAGE, notify


def main():
    # No report reads or output when this optional integration is unconfigured.
    if not os.environ.get("DISCORD_WEBHOOK_URL"):
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text())
    except Exception:
        logging.warning(FAILURE_MESSAGE)
        return
    notify(report, os.environ.get("ARCHWATCH_PR_URL", ""))


if __name__ == "__main__":
    main()
