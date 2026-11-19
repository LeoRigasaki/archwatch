"""Read modelling code with an LLM for architectures the deterministic rules cannot read. Manual runs only.

    uv run --frozen python tools/read_code.py --plan          # build prompts, no calls
    uv run --frozen python tools/read_code.py                 # call, verify, write results/code-reader/

Every claim is verified in code (cited lines, config keys, tensor shapes, layer pattern). The second-opinion model
is called only when claims conflict. Results are for review: nothing is written to data/models or the site.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archwatch import code_reader as cr  # noqa: E402
from archwatch import extract, hub  # noqa: E402
from archwatch.gemini import ProviderError, local_env  # noqa: E402


def inputs(repo, section, prompt):
    info = hub.repo_info(repo)
    config = hub.get_json(repo, "config.json", revision=info["sha"])
    if not isinstance(config, dict):
        raise SystemExit(f"{repo}: no config.json")
    code = cr.fetch_code(repo, info, config, section.get("code_sources"))
    tensors = hub.safetensors_headers(repo, info=info)
    try:  # whatever the rules can read is the reference the claims must not contradict
        rule_facts = {k: v[0] for k, v in extract.extract(config).items() if not k.startswith("_")}
    except Exception:  # noqa: BLE001 - the rules failing is why this reader runs at all
        rule_facts = {}
    return info, config, code, tensors, rule_facts, cr.prompt_text(prompt, code["files"], config)


def decide(first: list[dict], second: list[dict] | None) -> dict:
    """Per field: verified only when one value survives every check (and, if asked, both models agree)."""
    fields = {}
    for name in sorted({r["field"] for r in first + (second or [])}):
        ok1 = {
            json.dumps(r["value"], sort_keys=True) for r in first if r["field"] == name and r["verdict"] == "verified"
        }
        rows2 = [r for r in second or [] if r["field"] == name]
        ok2 = {json.dumps(r["value"], sort_keys=True) for r in rows2 if r["verdict"] == "verified"}
        conflicted = any(r["field"] == name and r["conflicts"] for r in first)
        if second is not None and conflicted:
            good = len(ok1 | ok2) == 1 and ok1 and ok2
            why = "both models verified the same value" if good else "models disagree or a claim failed its checks"
        else:
            good = len(ok1) == 1 and not conflicted
            why = "verified" if good else "failed a check or conflicts"
        value = json.loads(next(iter(ok1 | ok2))) if good else None
        fields[name] = dict(verdict="verified" if good else "review", value=value, why=why)
    return fields


def report(repo, info, code, result) -> str:
    lines = [f"# {repo}", "", f"Commit `{info['sha']}`. Code: {code['source']}."]
    for flag in code["flags"]:
        lines.append(f"- Flag: {flag}")
    lines += ["", "| Field | Value | Verdict | Model | Cited lines | Problems |", "|---|---|---|---|---|---|"]
    for row in result["claims"]:
        value = json.dumps(row["value"])
        value = value if len(value) <= 40 else value[:37] + "..."
        issues = "; ".join(row["problems"] + row["conflicts"]) or "none"
        lines.append(
            f"| {row['field']} | `{value}` | {row['verdict']} | {row['model']} | "
            f"{row['file']}:{row['line_start']}-{row['line_end']} | {issues} |"
        )
    lines += ["", "Field decisions:", ""]
    for name, d in result["fields"].items():
        lines.append(
            f"- {name}: **{d['verdict']}** {json.dumps(d['value']) if d['value'] is not None else ''} ({d['why']})"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="build prompts and stop before any call")
    parser.add_argument("--repo", action="append", help="limit to these repos (default: config list)")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    section = config["code_reader"]
    prompt = cr.load_prompt(ROOT / section["prompt"])
    repos = args.repo or section["models"]
    out = ROOT / section["output"]
    calls = 0
    local_env(ROOT / ".env")
    run_id = os.environ.get("GITHUB_RUN_ID") or "local"
    for repo in repos:
        info, cfg, code, tensors, rule_facts, text = inputs(repo, section, prompt)
        print(
            f"{repo}: code {code['source']}, {len(text.encode()):,} prompt bytes ({prompt['version']}), flags {code['flags']}"
        )
        if args.plan or not code["files"]:
            continue
        rows, second = [], None
        try:
            reader = cr.CodeReader(config, role=section["first"], run_id=run_id)
            if calls >= section["max_calls_per_run"]:
                raise ProviderError("per-run call cap reached")
            try:
                claims, meta = reader.read(repo, info["sha"], prompt, text)
            finally:
                calls += reader.calls  # a failed call was still reserved in the ledger and counts
            rows = [dict(r, model=meta["model"]) for r in cr.verdicts(claims.claims, code, cfg, tensors, rule_facts)]
            if any(r["conflicts"] for r in rows) and calls < section["max_calls_per_run"]:
                other = cr.CodeReader(config, role=section["second_opinion"], run_id=run_id)
                try:
                    claims2, meta2 = other.read(repo, info["sha"], prompt, text, purpose="second opinion")
                finally:
                    calls += other.calls
                second = [
                    dict(r, model=meta2["model"]) for r in cr.verdicts(claims2.claims, code, cfg, tensors, rule_facts)
                ]
        except ProviderError as error:
            print(f"  {repo}: {error}")
            if not rows:
                continue
        result = dict(
            repo=repo,
            commit=info["sha"],
            code_source=code["source"],
            prompt_version=prompt["version"],
            code_origin=code["origin"],
            flags=code["flags"],
            claims=rows + (second or []),
            fields=decide(rows, second),
            published=False,  # never published from here; a person reviews first
        )
        out.mkdir(parents=True, exist_ok=True)
        stem = repo.replace("/", "__")
        (out / f"{stem}.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
        (out / f"{stem}.md").write_text(report(repo, info, code, result))
        verified = sum(d["verdict"] == "verified" for d in result["fields"].values())
        print(f"  {len(result['claims'])} claims, {verified}/{len(result['fields'])} fields verified")
    print(f"calls this run: {calls} (cap {section['max_calls_per_run']})")


if __name__ == "__main__":
    main()
