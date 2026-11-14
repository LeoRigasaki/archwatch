"""Discover Hub models; optionally analyze admitted candidates. No LLM calls or schedule."""

import argparse
import json
from pathlib import Path

from archwatch.watcher import load_policy, run

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "config/watcher.yaml")
    parser.add_argument("--repo", action="append", help="Inspect only these explicit repository IDs (repeatable)")
    parser.add_argument("--process", action="store_true", help="Analyze up to the configured per-run limit")
    parser.add_argument("--retry-review", action="store_true", help="Recheck held records after rule changes")
    parser.add_argument(
        "--reprocess", action="store_true", help="Reanalyze unchanged admitted commits after schema/rule changes"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Output root, e.g. a temporary directory for a live smoke test"
    )
    args = parser.parse_args()
    report = run(
        args.root,
        load_policy(args.policy),
        repos=args.repo,
        process=args.process,
        retry_review=args.retry_review,
        reprocess=args.reprocess,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    # Per-model failures are persisted for review/retry, not discarded by failing
    # the entire batch before its state can be proposed in a PR.
    if report["discovery_errors"] and not report["outcomes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
