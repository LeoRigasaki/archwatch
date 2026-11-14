"""One-shot A/B experiment. No repairs, fallback, scheduled queue or explainer."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from archwatch.artifacts import write_if_changed
from archwatch.diagram_routes import browser_geometry, inspect_svg, passed, prompt, render_layout, specification
from archwatch.gemini import Gemini, ProviderError, local_env

ROOT = Path(__file__).resolve().parents[1]


def unfence(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def report(root, config, rows):
    expected = 2 * len(config["route_test"]["models"])
    complete = len(rows) == expected and all(r.get("generated") and r.get("scored") for r in rows)
    totals = {}
    checks = (
        "labels_and_numbers_from_published",
        "layer_sum_matches",
        "no_text_overlap",
        "everything_inside_canvas",
        "label_coverage",
        "nodes_and_edges_match",
        "all_labels_visible",
        "readable_font_sizes",
        "no_node_overlap",
        "canvas_within_experiment_bounds",
        "labels_attached_to_nodes",
        "text_fits_nodes",
    )
    for route in ("A", "B"):
        items = [r for r in rows if r["route"] == route]
        totals[route] = dict(
            passed=sum(r.get("passed", False) for r in items),
            cases=len(items),
            check_passes=sum(sum(r.get(k) is True for k in checks) for r in items),
            output_tokens=sum(r.get("metadata", {}).get("usage", {}).get("candidatesTokenCount", 0) for r in items),
        )
    winner = None
    if complete and max(t["passed"] for t in totals.values()) > 0:
        a, b = totals["A"], totals["B"]

        def key(t):
            return (t["passed"], t["check_passes"], -t["output_tokens"])

        winner = "A" if key(a) > key(b) else "B" if key(b) > key(a) else "tie"
    summary = dict(
        protocol="Eight first-response candidates; same fixed facts and graph, one model; no repairs or fallback.",
        generator=config["models"]["diagram"]["id"],
        complete=complete,
        totals=totals,
        winner=winner,
        cases=rows,
        selection_rule="Most complete passes, then most individual check passes, then fewer output tokens; declared before generation.",
    )
    lines = [
        "# Diagram route comparison",
        "",
        summary["protocol"],
        "",
        f"Generator: `{summary['generator']}` from `config/providers.yaml`. Reviews are deterministic; no model grades its own output.",
        "",
        "| Model | Route | Facts | Layer sum | No text overlap | Inside canvas | Coverage | Full pass | Output tokens |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]

    def mark(value):
        return "pass" if value is True else "fail"

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["id"],
                    row["route"],
                    *[
                        mark(row.get(k))
                        for k in (
                            "labels_and_numbers_from_published",
                            "layer_sum_matches",
                            "no_text_overlap",
                            "everything_inside_canvas",
                            "label_coverage",
                            "passed",
                        )
                    ],
                    str(row.get("metadata", {}).get("usage", {}).get("candidatesTokenCount", 0)),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Selection rule: " + summary["selection_rule"],
        "",
        f"Result: **{winner if winner else 'no passing case; no route selected' if complete else 'incomplete; no route selected'}**.",
        "",
        "Each case saves its frozen spec, prompt, raw response, SVG, review JSON and provider usage. Route B also saves its JSON layout. Geometric dimensions are not architecture facts; visible labels must match source-linked published values, allowing colon whitespace and numeric thousands separators. Input facts, node membership and edges are fixed for both routes.",
        "",
        "The LFM3-1.7B record is the requested historical record, used only for this private experiment; its current watcher derivative status is not overridden or newly admitted. The other subjects are the requested established records, not the newest unqualified layouts.",
        "",
        "These are four models and one sample per route. The checks test factual control, required-label coverage, layer totals and browser geometry, not exhaustive semantic architecture correctness or general aesthetic quality. A route failing validation is not publishable. No output is added to the public gallery by this experiment. The generation/repair loop and explainer remain disabled.",
    ]
    accounting = root / "results/diagram-route-test/request-accounting.json"
    if accounting.exists():
        summary["request_accounting"] = json.loads(accounting.read_text())
        usage = summary["request_accounting"]
        lines += [
            "",
            f"Request accounting: **{usage['reserved_requests']} reserved attempts**, {usage['successful_candidates']} successful candidates, {usage['http_503_failures']} HTTP 503 failures. Failed requests count against the 15-request weekly cap. Manual resumption used the same prompts and reused completed outputs; zero architecture repairs, fallback or model-review calls.",
        ]
    if complete and winner in ("A", "B"):
        lines += [
            "",
            f"A passed {totals['A']['passed']}/4 cases ({totals['A']['check_passes']}/48 individual checks); B passed {totals['B']['passed']}/4 ({totals['B']['check_passes']}/48). Output tokens: A {totals['A']['output_tokens']:,}, B {totals['B']['output_tokens']:,}. The selected route wins the predeclared token tie-break; this is not evidence of better visual quality.",
            "",
            "## Saved previews",
            "",
            "| Model | Full SVG (A) | JSON + Python (B) |",
            "|---|---|---|",
        ]
        for repo in config["route_test"]["models"]:
            path = "diagram-route-test/" + repo.replace("/", "__")
            lines.append(f"| {repo} | [preview]({path}/A/v1.png) | [preview]({path}/B/v1.png) |")
    for row in rows:
        if row.get("error") or not row.get("passed"):
            lines += [
                "",
                f"## {row['id']} — Route {row['route']}",
                "",
                "```json",
                json.dumps({k: v for k, v in row.items() if k not in ("metadata", "id", "route")}, indent=2),
                "```",
            ]
    write_if_changed(root / "results/diagram-route-test.md", "\n".join(lines) + "\n")
    write_if_changed(root / "results/diagram-route-test.json", json.dumps(summary, indent=2) + "\n")
    if winner in ("A", "B"):
        choice = dict(
            schema_version=1,
            route=winner,
            generator=summary["generator"],
            evidence="results/diagram-route-test.md",
            generation_loop_enabled=False,
            explainer_enabled=False,
        )
        write_if_changed(root / "config/diagram-route-choice.yaml", yaml.safe_dump(choice, sort_keys=False))
    return summary


def run(root, config, *, generate=False, score=False, provider=None):
    experiment = config["route_test"]
    if len(experiment["models"]) != 4 or experiment["repairs"] != 0 or experiment["fallback_enabled"]:
        raise ValueError("This experiment requires four subjects, no repairs and no fallback")
    if generate:
        local_env(root / ".env")
        provider = provider or Gemini(config, root / "cache/gemini-route-requests.json")
        meta = provider.preflight()
        if "generateContent" not in meta.get("supportedGenerationMethods", []):
            raise ProviderError("Configured model does not support this adapter")
        write_if_changed(
            root / "results/diagram-route-test/provider-preflight.json",
            json.dumps(
                {k: meta.get(k) for k in ("name", "supportedGenerationMethods", "inputTokenLimit", "outputTokenLimit")},
                indent=2,
            )
            + "\n",
        )
    rows = []
    for repo in experiment["models"]:
        record = json.loads((root / "data/models" / (repo.replace("/", "__") + ".json")).read_text())
        spec = specification(record)
        for route in ("A", "B"):
            folder = root / "results/diagram-route-test" / repo.replace("/", "__") / route
            folder.mkdir(parents=True, exist_ok=True)
            specpath = folder / "spec.yaml"
            if specpath.exists() and yaml.safe_load(specpath.read_text())["spec_hash"] != spec["spec_hash"]:
                raise ValueError("Frozen experiment inputs changed; use a separately versioned experiment")
            write_if_changed(specpath, yaml.safe_dump(spec, sort_keys=False, allow_unicode=True))
            request = prompt(spec, route)
            write_if_changed(folder / "prompt.txt", request + "\n")
            response_path = folder / "v1.response.txt"
            if generate and not response_path.exists():
                print(f"Generating {repo} route {route}", flush=True)
                try:
                    text, metadata = provider.generate(request, json_output=route == "B")
                except ProviderError as error:
                    error_number = len(list(folder.glob("request*.error.json"))) + 1
                    write_if_changed(
                        folder / f"request-{error_number}.error.json",
                        json.dumps(dict(error=str(error)), indent=2) + "\n",
                    )
                    report(root, config, rows)
                    raise
                metadata.update(spec_hash=spec["spec_hash"], prompt_hash=hashlib.sha256(request.encode()).hexdigest())
                write_if_changed(response_path, text + "\n")
                write_if_changed(folder / "v1.metadata.json", json.dumps(metadata, indent=2) + "\n")
                print(f"Received {repo} route {route}", flush=True)
            row = dict(
                id=repo,
                route=route,
                generated=response_path.exists(),
                reviewer="deterministic validators",
                spec_hash=spec["spec_hash"],
            )
            if response_path.exists():
                row["scored"] = bool(score)
                row["metadata"] = json.loads((folder / "v1.metadata.json").read_text())
                if (
                    row["metadata"]["spec_hash"] != spec["spec_hash"]
                    or row["metadata"]["prompt_hash"] != hashlib.sha256(request.encode()).hexdigest()
                ):
                    raise ValueError("Frozen generation inputs changed; cached responses cannot be reused")
                try:
                    output = unfence(response_path.read_text())
                    if route == "B":
                        layout = json.loads(output)
                        write_if_changed(folder / "v1.layout.json", json.dumps(layout, indent=2) + "\n")
                        svg = render_layout(layout, spec)
                    else:
                        svg = output
                    row.update(inspect_svg(svg, spec))
                    write_if_changed(folder / "v1.svg", svg + "\n")
                    if score:
                        row.update(
                            browser_geometry(
                                svg, spec, channel=experiment["browser_channel"], screenshot=folder / "v1.png"
                            )
                        )
                    row["passed"] = passed(row)
                except Exception as error:
                    row.update(passed=False, error=type(error).__name__ + ": " + str(error))
                write_if_changed(folder / "v1.review.json", json.dumps(row, indent=2) + "\n")
            rows.append(row)
    return report(root, config, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "config/providers.yaml")
    parser.add_argument(
        "--generate", action="store_true", help="Make the eight bounded one-shot calls (cached cases are reused)"
    )
    parser.add_argument("--score", action="store_true", help="Measure safe SVGs in the configured installed browser")
    args = parser.parse_args()
    result = run(args.root, yaml.safe_load(args.config.read_text()), generate=args.generate, score=args.score)
    print(
        json.dumps({"complete": result["complete"], "winner": result["winner"], "totals": result["totals"]}, indent=2)
    )


if __name__ == "__main__":
    main()
