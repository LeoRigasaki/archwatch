"""Spec -> Gemini SVG -> code checks -> up to N repairs, saved per model under results/diagram-gen/.

    uv run --frozen python tools/diagram_gen.py --specs-only     # build specs, no model calls
    uv run --frozen python tools/diagram_gen.py                  # full run (needs GEMINI_API_KEY)
    uv run --frozen python tools/diagram_gen.py --recheck        # re-run checks on saved SVGs, no model calls

Saved SVGs are reused, so an interrupted run resumes without repeating calls.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archwatch.diagram_checks import extract_svg, review  # noqa: E402
from archwatch.diagram_spec import build_spec, dump  # noqa: E402

PROMPTS = ROOT / "config/diagram-gen"
OUT = ROOT / "results/diagram-gen"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def run_model(repo, config, client, *, recheck=False):
    settings = config["diagram_gen"]
    record = json.loads((ROOT / "data/models" / (repo.replace("/", "__") + ".json")).read_text())
    spec = build_spec(record)
    spec_text = dump(spec, record)
    folder = OUT / repo.replace("/", "__")
    write(folder / "spec.yaml", spec_text)
    if client is None and not recheck:
        return dict(model=repo, spec_only=True)
    system = (PROMPTS / "system_instructions.txt").read_text()
    attempts = []
    svg = None
    for n in range(1, settings["repairs"] + 2):
        svg_path = folder / f"v{n}.svg"
        if svg_path.exists():
            svg = svg_path.read_text()
        elif recheck or client is None:
            break
        else:
            if n == 1:
                prompt = (
                    (PROMPTS / "generate_prompt.txt")
                    .read_text()
                    .format(spec=spec_text, num_layers=spec["model"]["layers"])
                )
            else:
                failures = "\n".join(f"- {f}" for f in attempts[-1]["failures"])
                prompt = (PROMPTS / "repair_prompt.txt").read_text().format(failures=failures, spec=spec_text, svg=svg)
            write(folder / f"v{n}.prompt.txt", prompt)
            try:
                text, meta = client.generate(system, prompt)
            except Exception as error:  # provider errors are recorded, never retried silently
                write(folder / f"v{n}.error.json", json.dumps(dict(error=str(error)), indent=2) + "\n")
                attempts.append(dict(version=n, error=str(error)))
                break
            write(folder / f"v{n}.response.txt", text)
            write(folder / f"v{n}.metadata.json", json.dumps(meta, indent=2) + "\n")
            try:
                svg = extract_svg(text)
            except ValueError:
                svg = text
            write(svg_path, svg)
        result = review(svg, spec, settings, screenshot=folder / f"v{n}.png")
        write(folder / f"v{n}.review.json", json.dumps(result, indent=2) + "\n")
        attempts.append(dict(version=n, passed=result["passed"], failures=result["failures"]))
        print(f"{repo} v{n}: {'PASS' if result['passed'] else str(len(result['failures'])) + ' problem(s)'}")
        if result["passed"]:
            break
    final = next((a for a in reversed(attempts) if "passed" in a), None)
    return dict(
        model=repo,
        attempts=len(attempts),
        final_version=final and final["version"],
        passed=bool(final and final["passed"]),
        remaining_failures=final["failures"] if final else None,
        errors=[a["error"] for a in attempts if "error" in a],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs-only", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    client = None
    if not args.specs_only and not args.recheck:
        from archwatch.gemini import Gemini, local_env

        local_env(ROOT / ".env")
        client = Gemini(config, OUT / "request-ledger.json")
    rows = [run_model(m, config, client, recheck=args.recheck) for m in args.models or config["diagram_gen"]["models"]]
    if not args.specs_only:
        write(OUT / "summary.json", json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
