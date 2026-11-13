"""A new analysis resolves a new commit; its individual reads stay pinned."""

import datetime as dt
import json
import struct
from types import SimpleNamespace

from archwatch import hub, pipeline


def test_analysis_refreshes_metadata_and_uses_revision_specific_files(tmp_path, monkeypatch):
    state = {"sha": "a" * 40, "calls": 0}
    metadata = SimpleNamespace(
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        last_modified=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        siblings=[SimpleNamespace(rfilename=x) for x in ("config.json", "README.md", "model.safetensors")],
        card_data={"license": "mit"},
        downloads=1,
        likes=1,
    )

    class API:
        def model_info(self, *args, **kwargs):
            state["calls"] += 1
            metadata.sha = state["sha"]
            return metadata

    tensors = {
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 8]},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [8, 8]},
        "model.layers.0.feed_forward.w1.weight": {"dtype": "BF16", "shape": [16, 8]},
    }
    payload = json.dumps(tensors).encode()
    requests = []

    def fetch(url, headers=None):
        requests.append(url)
        if headers:
            return struct.pack("<Q", len(payload)) if headers["Range"] == "bytes=0-7" else payload
        if url.endswith("config.json"):
            return json.dumps(
                {
                    "model_type": "toy",
                    "num_hidden_layers": 1,
                    "hidden_size": 8,
                    "vocab_size": 4,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "intermediate_size": 16,
                    "max_position_embeddings": 1024 if "/" + "a" * 40 + "/" in url else 2048,
                }
            ).encode()
        return b"# Toy model\n"

    monkeypatch.setattr(hub, "CACHE", tmp_path)
    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "_get", fetch)
    old = pipeline.analyze("lab/toy")
    state["sha"] = "b" * 40
    new = pipeline.analyze("lab/toy")
    assert old["source_commit"] == "a" * 40
    assert new["source_commit"] == "b" * 40
    assert old["published"]["context_length"] == 1024
    assert new["published"]["context_length"] == 2048
    assert state["calls"] == 2
    assert all("/resolve/main/" not in url for url in requests)
    assert (tmp_path / "lab__toy" / ("a" * 40) / "_tensors.json").is_file()
    assert (tmp_path / "lab__toy" / ("b" * 40) / "_tensors.json").is_file()
