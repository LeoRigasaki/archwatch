"""Build a source ZIP from Git's tracked and non-ignored files, without runtime state."""

import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cache",
    "research-reference",
    ".import-staging",
    "build",
    "dist",
}


def source_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    files = []
    for name in set(result.stdout.decode().split("\0")) - {""}:
        path = root / name
        parts = Path(name).parts
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in parts):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix in (".pyc", ".zip"):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        files.append(path)
    return sorted(files)


def package(root: Path, output: Path) -> int:
    files = source_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "archwatch/" + path.relative_to(root).as_posix())
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/archwatch-source.zip")
    args = parser.parse_args()
    count = package(ROOT, args.output)
    print(f"Packaged {count} source files: {args.output} ({args.output.stat().st_size:,} bytes)")
