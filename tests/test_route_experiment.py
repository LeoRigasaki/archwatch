import importlib.util
import json
import shutil
from pathlib import Path

import yaml

from archwatch.diagram_routes import render_layout

ROOT = Path(__file__).resolve().parents[1]


def experiment():
    spec = importlib.util.spec_from_file_location("route_experiment", ROOT / "tools/diagram_route_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def preflight(self):
        return dict(name="fake", supportedGenerationMethods=["generateContent"])

    def generate(self, prompt, *, json_output=False):
        self.calls += 1
        spec = json.loads(prompt.split("FIXED SPECIFICATION:\n", 1)[1])
        positions = []
        y = 20
        for node in spec["nodes"]:
            positions.append(dict(id=node["id"], x=20, y=y, width=600, height=node["min_height"]))
            y += node["min_height"] + 30
        layout = dict(
            canvas=dict(width=650, height=y + 20),
            nodes=positions,
            groups=[],
            labels=[dict(id=label, node=node["id"]) for node in spec["nodes"] for label in node["labels"]],
            edges=[e["id"] for e in spec["edges"]],
        )
        output = json.dumps(layout) if json_output else render_layout(layout, spec)
        return output, dict(usage=dict(candidatesTokenCount=1))


def test_cached_experiment_never_repeats_generation_or_selects_before_geometry(tmp_path):
    cfg = yaml.safe_load((ROOT / "config/providers.yaml").read_text())
    folder = tmp_path / "data/models"
    folder.mkdir(parents=True)
    for repo in cfg["route_test"]["models"]:
        file = repo.replace("/", "__") + ".json"
        shutil.copyfile(ROOT / "data/models" / file, folder / file)
    provider = FakeProvider()
    module = experiment()
    first = module.run(tmp_path, cfg, generate=True, score=False, provider=provider)
    assert provider.calls == 8 and not first["complete"] and first["winner"] is None
    assert not (tmp_path / "config/diagram-route-choice.yaml").exists()
    snapshot = {
        str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (tmp_path / "results").rglob("*")
        if p.is_file()
    }
    module.run(tmp_path, cfg, generate=True, score=False, provider=provider)
    assert provider.calls == 8
    assert snapshot == {
        str(p.relative_to(tmp_path)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (tmp_path / "results").rglob("*")
        if p.is_file()
    }
