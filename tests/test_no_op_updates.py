"""Exercise real cached extraction and artifact writes, including fresh timestamps."""

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from archwatch import codescan, hub, pipeline, weights

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_cached_pipeline_second_run_leaves_every_artifact_untouched(tmp_path, monkeypatch):
    cache, output = tmp_path / "cache", tmp_path / "output"
    files = ["model.safetensors", "modeling_z.py", "modeling_a.py", "README.md", "config.json"]
    revision = "a" * 40
    info = SimpleNamespace(
        sha=revision,
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        last_modified=None,
        siblings=[SimpleNamespace(rfilename=p) for p in files],
        card_data={"license": "mit"},
        downloads=1,
        likes=1,
    )

    class API:
        def model_info(self, *args, **kwargs):
            info.siblings.reverse()  # API listing order must not affect output.
            return info

    class Clock:
        calls = 0

        @classmethod
        def now(cls, timezone):
            cls.calls += 1
            return dt.datetime(2026, 1, 1, tzinfo=timezone) + dt.timedelta(minutes=cls.calls)

    monkeypatch.setattr(hub, "CACHE", cache)
    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected network read")))
    monkeypatch.setattr(pipeline, "dt", SimpleNamespace(datetime=Clock, timezone=dt.timezone))
    for repo in ("lab/old", "lab/new"):
        directory = cache / repo.replace("/", "__") / revision
        directory.mkdir(parents=True)
        config = dict(
            model_type="toy",
            num_hidden_layers=2,
            hidden_size=8,
            vocab_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=16,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
        )
        (directory / "config.json").write_text(json.dumps(config))
        (directory / "README.md").write_text("# Toy\n")
        for name in ("modeling_a.py", "modeling_z.py"):
            (directory / name).write_text("class ToyAttention:\n    pass\n")
        tensors = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 8]}}
        for layer in range(2):
            tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = {"dtype": "BF16", "shape": [8, 8]}
            tensors[f"model.layers.{layer}.feed_forward.w1.weight"] = {"dtype": "BF16", "shape": [16, 8]}
        if repo.endswith("new"):
            for name in ("z_norm", "a_norm"):
                tensors[f"model.layers.0.{name}.weight"] = {"dtype": "BF16", "shape": [8]}
        (directory / "_tensors.json").write_text(json.dumps(tensors))
    cli, site = module("run"), module("build_site")
    monkeypatch.setattr(cli, "OUT", output / "data/models")
    monkeypatch.setattr(site, "ROOT", output)
    cli.main(["lab/new:lab/old"])
    site.main()
    first = snapshot(output)
    pages = {"site/index.html", "site/style.css", "site/site.js", "site/compare/index.html"}
    assert pages <= set(first) and {"data/models/lab__new.json", "data/models/lab__new.yaml"} <= set(first)
    cli.main(["lab/new:lab/old"])
    site.main()
    assert Clock.calls == 4
    assert snapshot(output) == first  # bytes AND mtimes, not just parsed values
    # A real content change must still update the target and its derived artifacts.
    target = cache / "lab__new" / revision / "config.json"
    config = json.loads(target.read_text())
    config["max_position_embeddings"] = 2048
    target.write_text(json.dumps(config))
    cli.main(["lab/new:lab/old"])
    site.main()
    changed = snapshot(output)
    assert changed["data/models/lab__old.json"] == first["data/models/lab__old.json"]
    for name in ("data/models/lab__new.json", "data/models/lab__new.yaml", "diagrams/lab__new.svg"):
        assert changed[name][0] != first[name][0]


def test_ties_are_stable_across_process_hash_seeds():
    script = """import json
from archwatch import hub, pipeline
names = {"model.layers.0.z_norm.weight", "model.layers.0.a_norm.weight", "model.layers.0.m_norm.weight"}
hub.safetensors_headers = lambda repo: {n: {"dtype":"BF16", "shape":[8]} for n in names} if repo == "new" else {}
print(json.dumps(pipeline.tensor_diff("new", "old", top=2)))
"""
    outputs = [
        subprocess.check_output([sys.executable, "-c", script], env=dict(os.environ, PYTHONHASHSEED=seed))
        for seed in ("1", "17", "991")
    ]
    assert outputs[0] == outputs[1] == outputs[2]
    assert [row["pattern"] for row in json.loads(outputs[0])["added"]] == ["layers.*.a_norm", "layers.*.m_norm"]


def test_representative_shapes_and_component_ties_ignore_input_order():
    tensors = {
        "model.layers.1.feed_forward.w1.weight": {"dtype": "BF16", "shape": [32, 8]},
        "model.layers.0.feed_forward.w1.weight": {"dtype": "BF16", "shape": [16, 8]},
        "model.embed_tokens.weight": {"dtype": "BF16", "shape": [4, 8]},
        "model.lm_head.weight": {"dtype": "BF16", "shape": [4, 8]},
    }
    reverse = dict(reversed(list(tensors.items())))
    assert weights.shape_facts(tensors, None) == weights.shape_facts(reverse, None)
    assert list(weights.count(tensors, None)["by_component"]) == list(weights.count(reverse, None)["by_component"])
    files = {"modeling_z.py": "class ZAttention:\n pass", "modeling_a.py": "class AAttention:\n pass"}
    assert codescan.scan(files) == codescan.scan(dict(reversed(list(files.items()))))
