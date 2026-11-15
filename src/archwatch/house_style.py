"""The approved house-style reference diagram: ArchWatch's own output only, with checked provenance."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

REFERENCE_DIR = Path("docs/diagram-kit")
NAMES = ("reference.svg", "reference.png", "reference.json")
ALLOWED_SOURCES = ("results/diagram-gen/", "results/diagram-gen-v1/", "results/diagram-gen-v2/")


class ReferenceError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approve(root: Path, source: str, *, approved_on: str | None = None) -> dict:
    """Promote an ArchWatch-generated diagram (and its saved PNG render) to the house-style reference."""
    if not source.startswith(ALLOWED_SOURCES) or not source.endswith(".svg"):
        raise ReferenceError("Only ArchWatch's own generated diagrams under results/diagram-gen*/ may be references")
    svg, png = root / source, (root / source).with_suffix(".png")
    if not svg.exists() or not png.exists():
        raise ReferenceError("Reference SVG and its PNG render must both exist")
    review = svg.with_name(svg.stem + ".review.json")
    if not review.exists() or not json.loads(review.read_text()).get("passed"):
        raise ReferenceError("Only a diagram that passed every code check may become the reference")
    out = root / REFERENCE_DIR
    shutil.copyfile(svg, out / "reference.svg")
    shutil.copyfile(png, out / "reference.png")
    meta = dict(
        origin="archwatch-generated",
        source=source,
        approved_on=approved_on or dt.date.today().isoformat(),
        svg_sha256=_sha(out / "reference.svg"),
        png_sha256=_sha(out / "reference.png"),
        note="House style only: match this layout and styling, not its content.",
    )
    (out / "reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load(root: Path) -> tuple[str, bytes]:
    """Return (svg text, png bytes) after checking the reference is ArchWatch's own, unmodified output."""
    out = root / REFERENCE_DIR
    try:
        meta = json.loads((out / "reference.json").read_text())
    except (OSError, ValueError) as error:
        raise ReferenceError("No approved house-style reference (docs/diagram-kit/reference.json)") from error
    if meta.get("origin") != "archwatch-generated" or not str(meta.get("source", "")).startswith(ALLOWED_SOURCES):
        raise ReferenceError("House-style reference is not an ArchWatch-generated diagram")
    if _sha(out / "reference.svg") != meta.get("svg_sha256") or _sha(out / "reference.png") != meta.get("png_sha256"):
        raise ReferenceError("House-style reference files differ from their approved hashes")
    return (out / "reference.svg").read_text(), (out / "reference.png").read_bytes()
