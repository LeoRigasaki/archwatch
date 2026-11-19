"""Score the LLM code reader on published models whose fields the deterministic rules already know.

    uv run --frozen python tools/eval_code_reader.py --plan   # build prompts, no calls
    uv run --frozen python tools/eval_code_reader.py          # one call per (model, reader), then score

Per claim: is the value correct (against our published record), is the citation valid (file, line range, exact
quote), do the config keys agree, did the weight check pass (or n/a for fields with no weight). The key safety
number is "passes every check but wrong": the verifier must not let wrong values through.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from read_code import inputs  # noqa: E402

from archwatch import code_reader as cr  # noqa: E402
from archwatch.gemini import ProviderError, local_env  # noqa: E402

NO_REFERENCE = {"layer_types", "mixer_type"}  # our record uses ArchWatch kind names; not comparable word for word


def record(repo: str) -> dict:
    return json.loads((ROOT / "data/models" / (repo.replace("/", "__") + ".json")).read_text())


def truth(repo: str) -> dict:
    return record(repo)["published"]


def after_cutoff(repo: str, reader: dict) -> bool | None:
    """Did the model reach the Hub after the reader's stated training cutoff (so it cannot have memorised it)?"""
    cutoff = reader.get("knowledge_cutoff")
    released = (record(repo).get("hub_created_at") or "")[:10]
    return released > str(cutoff) if cutoff and released else None


def cutoff_cell(r) -> str:
    return {True: f"yes ({r['released']})", False: f"no ({r['released']})"}.get(r.get("after_cutoff"), "unknown")


def sample_size(results, config) -> str:
    scored = [r for r in results if "scored" in r]
    models = sorted({r["repo"] for r in scored})
    ref = sum(1 for r in scored for s in r["scored"] if s["value_correct"] is not None)
    after = sorted({r["repo"].split("/")[1] for r in scored if r["after_cutoff"]})
    cutoffs = sorted(
        {f"{m['id']} {m['knowledge_cutoff']}" for m in config["models"].values() if m.get("knowledge_cutoff")}
    )
    return (
        f"Sample size: {len(models)} models and {ref} claimed fields with a reference value across {len(scored)} "
        f"readings; {len(after)} model(s) released after the reader's cutoff ({', '.join(after) or 'none'}). "
        f"Cutoff from the model card: {', '.join(cutoffs)} (the card adds that some domains end in January 2025). "
        "Too small to generalize: these numbers decide whether to try the reader on new models, nothing more."
    )


def correct(field, value, reference):
    if field in NO_REFERENCE or field not in reference or reference[field] is None:
        return None
    ref = reference[field]
    if isinstance(ref, str) and isinstance(value, str):
        return value.strip().lower() == ref.strip().lower()
    return value == ref


def score(rows, reference):
    out = []
    for r in rows:
        checks_pass = not (r["citation"] or r["agreement"] or r["weights"] or r["conflicts"])
        ok = correct(r["field"], r["value"], reference)
        out.append(
            dict(
                field=r["field"],
                value=r["value"],
                reference=reference.get(r["field"]),
                value_correct=ok,
                citation_valid=not r["citation"],
                config_agrees=not r["agreement"],
                weight_check=r["weight_status"],
                passes_checks=checks_pass,
                verdict=r["verdict"],
                problems=r["problems"] + r["conflicts"],
                cited=f"{r['file']}:{r['line_start']}-{r['line_end']}",
                module_paths=r["module_paths"],
            )
        )
    return out


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


def summary(scored):
    ref = [s for s in scored if s["value_correct"] is not None]
    shape = [s for s in scored if s["weight_check"] != "n/a"]
    wrong_pass = [s for s in ref if s["passes_checks"] and not s["value_correct"]]
    return dict(
        claims=len(scored),
        value_correct=pct(sum(s["value_correct"] for s in ref), len(ref)),
        citation_valid=pct(sum(s["citation_valid"] for s in scored), len(scored)),
        config_agrees=pct(sum(s["config_agrees"] for s in scored), len(scored)),
        weight_check=pct(sum(s["weight_check"] == "passed" for s in shape), len(shape)),
        passes_checks=pct(sum(s["passes_checks"] for s in scored), len(scored)),
        wrong_but_passes=len(wrong_pass),
        verified=sum(s["verdict"] == "verified" for s in scored),
    )


def report(results, prompt, config) -> str:
    lines = [
        "# Code reader evaluation",
        "",
        f"Prompt `{prompt['version']}` (sha256 `{prompt['sha256'][:12]}`). Reference: our published records for models "
        "whose fields the deterministic rules already read. Nothing here is published.",
        "",
        "Columns: **value correct** = claimed value equals our published value (claims whose field has a reference); "
        "**citation valid** = cited file is fetched modelling code, the line range exists and the quote equals those "
        "lines; **config agrees** = cited config keys exist and agree; **weight check** = the cited line builds a module "
        "whose tensors, mapped from the code's own module tree, have consistent shapes (fields with a weight only); "
        "**passes all checks** = every check above with no conflict (the version cap is reported separately); "
        "**wrong but passes** = passes every check yet disagrees with our record (the verifier must keep this at 0).",
        "",
        "| Model | Reader | Code | Released after reader's training cutoff | Claims | Value correct | Citation valid | "
        "Config agrees | Weight check | Passes all checks | Wrong but passes | Verified |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r['repo'].split('/')[1]} | {r['reader']} | {r.get('code', '')} | {cutoff_cell(r)} | "
                f"error: {r['error']} | | | | | | | |"
            )
            continue
        s = r["summary"]
        lines.append(
            f"| {r['repo'].split('/')[1]} | {r['reader']} | {r['code']} | {cutoff_cell(r)} | {s['claims']} | "
            f"{s['value_correct']} | "
            f"{s['citation_valid']} | {s['config_agrees']} | {s['weight_check']} | {s['passes_checks']} | "
            f"{s['wrong_but_passes']} | {s['verified']} |"
        )
    lines.append("")
    for reader in sorted({r["reader"] for r in results}):
        for label, keep in (("all models", None), ("released after its cutoff", True), ("released before it", False)):
            group = [
                r for r in results if r["reader"] == reader and "scored" in r and keep in (None, r["after_cutoff"])
            ]
            if not group:
                lines.append(f"- **{reader}, {label}:** no scored models")
                continue
            t = summary([s for r in group for s in r["scored"]])
            lines.append(
                f"- **{reader}, {label}** ({len(group)} models, {t['claims']} claims): value correct "
                f"{t['value_correct']}; citation valid {t['citation_valid']}; weight check {t['weight_check']}; "
                f"passes all checks {t['passes_checks']}; wrong but passes {t['wrong_but_passes']}."
            )
    lines += ["", sample_size(results, config)]
    lines += ["", "## Per field", ""]
    fields = sorted({s["field"] for r in results if "scored" in r for s in r["scored"]})
    readers = sorted({r["reader"] for r in results if "scored" in r})
    lines += [
        "| Field | " + " | ".join(f"{m}: correct / passes" for m in readers) + " |",
        "|---|" + "---|" * len(readers),
    ]
    for f in fields:
        cells = []
        for m in readers:
            rows = [s for r in results if r["reader"] == m and "scored" in r for s in r["scored"] if s["field"] == f]
            ref = [s for s in rows if s["value_correct"] is not None]
            cells.append(
                f"{sum(s['value_correct'] for s in ref)}/{len(ref)} / {sum(s['passes_checks'] for s in rows)}/{len(rows)}"
            )
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    lines += ["", "## Claims that failed or disagree with the record", ""]
    for r in results:
        for s in r.get("scored", []):
            if s["value_correct"] is False or not s["passes_checks"]:
                why = "; ".join(s["problems"]) or "passes every check but disagrees with the record"
                lines.append(
                    f"- {r['repo'].split('/')[1]} / {r['reader']} / {s['field']} = `{json.dumps(s['value'])[:40]}` "
                    f"(record `{json.dumps(s['reference'])[:40]}`), {s['cited']}: {why[:220]}"
                )
    return "\n".join(lines) + "\n"


def save(ev, results, prompt, config):
    out = ROOT / ev["output"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(dict(prompt=prompt["version"], results=results), indent=1, default=str) + "\n"
    )
    (ROOT / ev["report"]).write_text(report(results, prompt, config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    section, ev = config["code_reader"], config["code_reader"]["eval"]
    prompt = cr.load_prompt(ROOT / section["prompt"])
    local_env(ROOT / ".env")
    run_id = os.environ.get("GITHUB_RUN_ID") or "local"
    results, calls = [], 0
    for role in ev["readers"]:
        reader_id = config["models"][role]["id"]
        for repo in ev["models"]:
            info, cfg, code, tensors, _, text = inputs(repo, section, prompt)
            print(f"{reader_id} {repo}: {code['source']}, {len(text.encode()):,} bytes, flags {code['flags']}")
            if args.plan:
                continue
            entry = dict(
                repo=repo,
                reader=reader_id,
                code=code["source"],
                flags=code["flags"],
                released=(record(repo).get("hub_created_at") or "")[:10],
                after_cutoff=after_cutoff(repo, config["models"][role]),
            )
            if calls >= ev["max_calls_per_run"]:
                results.append(dict(entry, error="eval call cap reached"))
                continue
            reader = cr.CodeReader(config, role=role, run_id=run_id)
            try:
                claims, meta = reader.read(repo, info["sha"], prompt, text, purpose="eval")
            except ProviderError as error:
                results.append(dict(entry, error=str(error)))
                print(f"   {error}", flush=True)
                save(ev, results, prompt, config)
                continue
            finally:
                calls += reader.calls
            rows = cr.verdicts(claims.claims, code, cfg, tensors, {})  # judged on its own: rules not consulted
            scored = score(rows, truth(repo))
            results.append(dict(entry, scored=scored, summary=summary(scored), cached=meta["cached"]))
            print(f"   {summary(scored)}", flush=True)
            save(ev, results, prompt, config)  # after every call, so an interrupted run keeps what it paid for
    if args.plan:
        return
    out = ROOT / ev["output"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(dict(prompt=prompt["version"], results=results), indent=1, default=str) + "\n"
    )
    (ROOT / ev["report"]).write_text(report(results, prompt, config))
    print(f"calls this run: {calls} (cap {ev['max_calls_per_run']})")


if __name__ == "__main__":
    main()
