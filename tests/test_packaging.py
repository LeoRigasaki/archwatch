"""Source archives exclude runtime state even if accidentally tracked."""

import importlib.util
import subprocess
import zipfile
from pathlib import Path


def test_source_zip_excludes_environments_caches_and_egg_info(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "archwatch_packager", Path(__file__).resolve().parents[1] / "tools/package_source.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in [
        "src/archwatch/__init__.py",
        ".venv/bin/python",
        ".pytest_cache/state",
        "src/archwatch.egg-info/PKG-INFO",
        ".env",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    subprocess.run(["git", "add", "-f", "."], cwd=tmp_path, check=True)
    output = tmp_path / "dist/source.zip"
    assert module.package(tmp_path, output) == 1
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["archwatch/src/archwatch/__init__.py"]
