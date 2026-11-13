"""Run ArchWatch on a list of models, each paired with its predecessor, and write records.

python run.py moonshotai/Kimi-K3.1-Pro:moonshotai/Kimi-K3 LiquidAI/LFM3-1.7B:LiquidAI/LFM2.5-1.2B-Instruct
"""

import sys
from pathlib import Path

from archwatch import pipeline
from archwatch.artifacts import save_record

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
            save_record(old, OUT)
        save_record(rec, OUT, with_yaml=True, diagrams=OUT.parents[1] / "diagrams")
        print(f"   published {len(rec['published'])} fields, {len(rec['review_queue'])} sent to review")


if __name__ == "__main__":
    main(sys.argv[1:])
