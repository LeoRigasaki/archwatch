"""Run ArchWatch on a list of models, each paired with its predecessor, and write records.

    python run.py moonshotai/Kimi-K3.1-Pro:moonshotai/Kimi-K3 LiquidAI/LFM3-1.7B:LiquidAI/LFM2.5-1.2B-Instruct
"""
import json
import sys
from pathlib import Path

import yaml

from archwatch import pipeline

OUT = Path(__file__).parent / "data" / "models"


def main(pairs: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        repo, _, pred = pair.partition(":")
        print(f"== {repo}")
        rec = pipeline.analyze(repo)
        if pred:
            print(f"   predecessor {pred}")
            old = pipeline.analyze(pred)
            rec["predecessor"] = {"id": pred, "source_commit": old["source_commit"]}
            rec["diff_vs_predecessor"] = pipeline.diff(rec, old)
            rec["tensor_diff_vs_predecessor"] = pipeline.tensor_diff(repo, pred)
            (OUT / f"{pred.replace('/', '__')}.json").write_text(json.dumps(old, indent=1, default=str))
        name = repo.replace("/", "__")
        (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1, default=str))
        slim = {k: rec[k] for k in ("id", "source_commit", "hub_created", "license", "analyzed_at", "published")}
        slim["review_queue"] = [{"field": r["field"], "candidates": r["candidates"]} for r in rec["review_queue"]]
        slim["unchecked_notes"] = rec["unchecked_notes"]
        (OUT / f"{name}.yaml").write_text(yaml.safe_dump(slim, sort_keys=False, allow_unicode=True))
        print(f"   published {len(rec['published'])} fields, {len(rec['review_queue'])} sent to review")


if __name__ == "__main__":
    main(sys.argv[1:])
