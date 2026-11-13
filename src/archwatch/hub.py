"""Hugging Face Hub access: cached file downloads and safetensors headers via HTTP range requests.

Nothing here downloads model weights. A safetensors file starts with an 8-byte little-endian header
length followed by a JSON header that lists every tensor's dtype and shape, so two small range
requests per shard are enough to know the exact shape of every weight in the model.
"""

from __future__ import annotations

import json
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi

CACHE = Path(__file__).resolve().parents[2] / "cache"
BASE = "https://huggingface.co"
MAX_HEADER_BYTES = 64 * 1024 * 1024


def _cache_dir(repo: str) -> Path:
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        if headers and "Range" in headers:
            if r.status != 206:
                raise ValueError("Server ignored the byte range; refusing to download a weight body")
            start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
            expected = end - start + 1
            data = r.read(expected + 1)
            if len(data) != expected:
                raise ValueError("Weight-header range returned an unexpected byte count")
            return data
        return r.read()


def get_text(repo: str, path: str, *, revision: str | None = None) -> str | None:
    """Return a repo file's text (cached), or None if the file does not exist."""
    revision = revision or repo_info(repo)["sha"]
    directory = _cache_dir(repo) / revision
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / path.replace("/", "__")
    if f.exists():
        return f.read_text()
    try:
        data = _get(f"{BASE}/{repo}/resolve/{revision}/{path}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    f.write_bytes(data)
    return data.decode("utf-8", errors="replace")


def get_json(repo: str, path: str, *, revision: str | None = None) -> dict | None:
    t = get_text(repo, path, revision=revision)
    return json.loads(t) if t is not None else None


def repo_info(repo: str, *, refresh: bool = False, api=None) -> dict:
    """Commit sha, creation date, files and license for a model repo (cached)."""
    f = _cache_dir(repo) / "_repo_info.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text())
    info = (api or HfApi()).model_info(repo, files_metadata=False)
    out = {
        "sha": info.sha,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "files": sorted(s.rfilename for s in (info.siblings or [])),
        "license": (info.card_data or {}).get("license") if info.card_data else None,
        "license_name": (info.card_data or {}).get("license_name") if info.card_data else None,
        "downloads": info.downloads,
        "likes": info.likes,
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": sorted(getattr(info, "tags", None) or []),
        "base_model": (info.card_data or {}).get("base_model") if info.card_data else None,
        "base_model_relation": (info.card_data or {}).get("base_model_relation") if info.card_data else None,
    }
    temporary = f.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(out, indent=1))
    temporary.replace(f)
    return out


def _header(repo: str, fname: str, revision: str | None = None) -> dict:
    url = f"{BASE}/{repo}/resolve/{revision or repo_info(repo)['sha']}/{fname}"
    n = struct.unpack("<Q", _get(url, {"Range": "bytes=0-7"}))[0]
    if not 2 <= n <= MAX_HEADER_BYTES:
        raise ValueError("Safetensors header length is outside the supported range")
    h = json.loads(_get(url, {"Range": f"bytes=8-{7 + n}"}))
    h.pop("__metadata__", None)
    return h


def safetensors_headers(repo: str, *, info: dict | None = None) -> dict[str, dict]:
    """Merged {tensor_name: {dtype, shape}} across every safetensors shard (cached)."""
    info = info or repo_info(repo)
    directory = _cache_dir(repo) / info["sha"]
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / "_tensors.json"
    if f.exists():
        return json.loads(f.read_text())
    shards = sorted(x for x in info["files"] if x.endswith(".safetensors") and "/" not in x)
    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(lambda s: _header(repo, s, info["sha"]), shards))
    merged: dict[str, dict] = {}
    for p in parts:
        for k, v in p.items():
            merged[k] = {"dtype": v["dtype"], "shape": v["shape"]}
    f.write_text(json.dumps(merged))
    return merged
