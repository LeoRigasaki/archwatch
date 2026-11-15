"""Hugging Face Hub access: cached file downloads and safetensors headers via HTTP range requests.

Nothing here downloads model weights. A safetensors file starts with an 8-byte little-endian header
length followed by a JSON header that lists every tensor's dtype and shape, so two small range
requests per shard are enough to know the exact shape of every weight in the model.
"""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import HfApi, get_token

CACHE = Path(__file__).resolve().parents[2] / "cache"
BASE = "https://huggingface.co"
MAX_HEADER_BYTES = 64 * 1024 * 1024


class _SourceRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        target, original = urlsplit(newurl), urlsplit(req.full_url)
        if redirected and (target.hostname != original.hostname or target.scheme != "https"):
            redirected.remove_header("Authorization")
        return redirected


def _cache_dir(repo: str) -> Path:
    d = CACHE / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, headers: dict | None = None) -> bytes:
    request_headers = dict(headers or {})
    target = urlsplit(url)
    token = get_token() if target.scheme == "https" and target.hostname == "huggingface.co" else None
    if token:
        request_headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=request_headers)
    open_url = urllib.request.build_opener(_SourceRedirect()).open if token else urllib.request.urlopen
    with open_url(req, timeout=120) as r:
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


TP_SHARD = re.compile(r"-TP-\d+\.safetensors$")


def weight_shards(repo: str, info: dict) -> list[str]:
    """Root safetensors files that hold the weights.

    A Mistral-native consolidated*.safetensors export duplicates the HF-format shards (under native tensor names)
    when an index lists those shards; it is not counted twice. Every other root file is kept, including extra
    components the index does not list (e.g. a separate gate network).
    """
    shards = sorted(x for x in info["files"] if x.endswith(".safetensors") and "/" not in x)
    if "model.safetensors.index.json" in info["files"]:
        index = get_json(repo, "model.safetensors.index.json", revision=info["sha"]) or {}
        listed = set((index.get("weight_map") or {}).values())
        if listed and listed <= set(shards):
            shards = [x for x in shards if x in listed or not x.startswith("consolidated")]
    return shards


def safetensors_headers(repo: str, *, info: dict | None = None) -> dict[str, dict]:
    """Merged {tensor_name: {dtype, shape}} across the weight shards (cached).

    Tensor-parallel exports (-TP-NNN files) store slices of the same tensor name; the merged entry keeps the
    per-rank shape and records `tp_parts` so element counts cover every rank.
    """
    info = info or repo_info(repo)
    directory = _cache_dir(repo) / info["sha"]
    directory.mkdir(parents=True, exist_ok=True)
    f, used = directory / "_tensors.json", directory / "_shards.json"
    shards = weight_shards(repo, info)
    all_root = sorted(x for x in info["files"] if x.endswith(".safetensors") and "/" not in x)
    if f.exists():
        if used.exists():
            reusable = json.loads(used.read_text()) == shards
        else:  # caches written before shard selection read every root file and merged TP slices by name
            reusable = all_root == shards and not any(TP_SHARD.search(x) for x in shards)
            if reusable:
                used.write_text(json.dumps(shards))
        if reusable:
            return json.loads(f.read_text())
    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = list(pool.map(lambda s: (s, _header(repo, s, info["sha"])), shards))
    merged: dict[str, dict] = {}
    for shard, p in parts:
        for k, v in p.items():
            if k in merged and TP_SHARD.search(shard):
                merged[k]["tp_parts"] = merged[k].get("tp_parts", 1) + 1
            else:
                merged[k] = {"dtype": v["dtype"], "shape": v["shape"]}
    f.write_text(json.dumps(merged))
    used.write_text(json.dumps(shards))
    return merged
