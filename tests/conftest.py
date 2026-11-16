"""site/ is a build output (Vercel builds it; it is not committed). Build it once so tests read fresh pages."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def built_site():
    spec = importlib.util.spec_from_file_location("site_build_for_tests", ROOT / "build_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
