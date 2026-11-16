"""The one layer-type palette (config/site.yaml `layer_colors`), shared by the site and the code-drawn diagrams."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "config" / "site.yaml"
FALLBACK = "#999999"


@lru_cache(maxsize=1)
def colors() -> dict[str, str]:
    return dict(yaml.safe_load(CONFIG.read_text())["layer_colors"])


def color(kind: str) -> str:
    return colors().get(kind, FALLBACK)
