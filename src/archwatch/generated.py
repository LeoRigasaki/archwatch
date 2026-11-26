"""Generated (Gemini-drawn) diagrams for the gallery, used only when they passed every code check.

A generated diagram is published to diagrams/generated/<stem>.svg with a .json sidecar holding the spec hash
it was drawn from, its review result and its own hash. The gallery shows it only if all of these still hold
for the record being rendered; otherwise it falls back to the code-drawn diagram.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import yaml

from .diagram_checks import static_checks
from .diagram_spec import TEMPLATES, SpecError, build_spec, dump

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path("diagrams/generated")


def spec_hash(spec: dict, record: dict) -> str:
    """Hash of the drawn spec body (the header carries the source commit, which does not change the drawing)."""
    return hashlib.sha256(dump(spec, record).split("\n\n", 1)[1].encode()).hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def publish(root: Path, record: dict, svg_path: Path) -> dict:
    """Copy a generated SVG that passed every check into diagrams/generated with its provenance."""
    review = json.loads(svg_path.with_name(svg_path.stem + ".review.json").read_text())
    if not review.get("passed"):
        raise ValueError("Only a diagram that passed every code check can be published")
    drawn_from = (svg_path.parent / "spec.yaml").read_text()
    template = next((t for t in TEMPLATES if dump(build_spec(record, t), record) == drawn_from), None)
    if template is None:
        raise ValueError("The diagram was drawn from a different spec than the record's current one")
    spec = build_spec(record, template)
    stem = record["id"].replace("/", "__")
    out = root / FOLDER
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(svg_path, out / f"{stem}.svg")
    meta = dict(
        model=record["id"],
        source_commit=record["source_commit"],
        spec_sha256=spec_hash(spec, record),
        spec_template=template,
        svg_sha256=_sha((out / f"{stem}.svg").read_bytes()),
        source=str(svg_path.relative_to(root)),
        checks_passed=sorted(review["checks"]),
    )
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def select(record: dict, root: Path = ROOT) -> tuple[str, dict] | None:
    """Return (svg, meta) when a published generated diagram still passes every check for this record."""
    stem = record["id"].replace("/", "__")
    svg_path, meta_path = root / FOLDER / f"{stem}.svg", root / FOLDER / f"{stem}.json"
    if not (svg_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    svg = svg_path.read_text()
    try:
        spec = build_spec(record, meta.get("spec_template", "v1"))  # the template it was drawn from
    except SpecError:
        return None
    if meta.get("spec_sha256") != spec_hash(spec, record) or meta.get("svg_sha256") != _sha(svg.encode()):
        return None
    settings = yaml.safe_load((root / "config/providers.yaml").read_text())["diagram_gen"]
    if not all(c["passed"] for c in static_checks(svg, spec, settings).values()):
        return None
    return svg, meta
