"""Write 'what changed vs predecessor' notes from the diff JSON with the configured review model.

    uv run --frozen python tools/explain.py moonshotai/Kimi-K3.1-Pro LiquidAI/LFM3-1.7B ...
    uv run --frozen python tools/explain.py --recheck   # re-run checks on saved notes, no calls

Saves results/explainer/<stem>.json (input, input hash, every attempt, checks). Not shown on the site yet.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archwatch.explainer import check, diff_input, format_numbers, input_hash, word_limit  # noqa: E402

PROMPTS = ROOT / "config/explainer"
OUT = ROOT / "results/explainer"


def explain(repo, config, client, *, out=None):
    settings = config["explainer"]
    record = json.loads((ROOT / "data/models" / (repo.replace("/", "__") + ".json")).read_text())
    data = diff_input(record)
    diff = json.dumps({k: data[k] for k in ("changes", "weights")}, indent=1, ensure_ascii=False)
    limit = word_limit(data, settings)
    system = (PROMPTS / "system_instructions.txt").read_text().format(max_words=limit)
    path = (out or OUT) / (repo.replace("/", "__") + ".json")
    saved = json.loads(path.read_text()) if path.exists() else None
    if saved and saved["input_sha256"] == input_hash(data) and saved["passed"]:
        return saved  # unchanged diff, passing note: no call
    attempts = []
    for n in range(1 + settings["repairs"]):
        if n == 0:
            prompt = (
                (PROMPTS / "prompt.txt").read_text().format(old=data["predecessor"], new=data["successor"], diff=diff)
            )
        else:
            prompt = (
                (PROMPTS / "repair_prompt.txt")
                .read_text()
                .format(
                    failures="\n".join("- " + p for p in attempts[-1]["check"]["problems"]),
                    old=data["predecessor"],
                    new=data["successor"],
                    diff=diff,
                    text=attempts[-1]["text"],
                )
            )
        text, meta = client.generate(system, prompt)
        text = format_numbers(" ".join(text.split()))
        result = check(text, data, max_words=limit)
        attempts.append(dict(text=text, check=result, usage=meta.get("usage", {}), model=meta.get("model")))
        if result["passed"]:
            break
    final = attempts[-1]
    saved = dict(
        model=repo,
        predecessor=data["predecessor"],
        input=data,
        input_sha256=input_hash(data),
        text=final["text"],
        passed=final["check"]["passed"],
        attempts=attempts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    if args.recheck:
        for path in sorted(OUT.glob("*.json")):
            saved = json.loads(path.read_text())
            limit = word_limit(saved["input"], config["explainer"])
            print(saved["model"], check(saved["text"], saved["input"], max_words=limit))
        return
    from archwatch.gemini import Gemini, local_env

    local_env(ROOT / ".env")
    client = Gemini(config, ROOT / config["explainer"]["ledger"], role="review", section="explainer")
    for repo in args.models:
        saved = explain(repo, config, client)
        print(f"{repo}: {'PASS' if saved['passed'] else 'FAIL'} after {len(saved['attempts'])} call(s)")
        print("  " + saved["text"])


if __name__ == "__main__":
    main()
