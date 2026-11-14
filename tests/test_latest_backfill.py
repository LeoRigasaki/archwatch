import copy
import importlib.util
import json
from pathlib import Path

from test_watcher import Source, files, model, policy

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("latest_backfill", ROOT / "tools/backfill_latest.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_review_record_survives_without_public_artifacts_and_restart_is_stable(tmp_path):
    old = model("lab/Family1", "2025-01-01T00:00:00Z")
    new = model("lab/Family2", "2026-01-01T00:00:00Z", sha="b" * 40)
    new["published"]["generation_type"] = "block_diffusion"
    source = Source([old, new])
    source.infos[old["id"]]["files"] = ["model.safetensors", "params.json"]
    manifest = dict(
        models=[
            dict(
                id=new["id"],
                source_commit=new["source_commit"],
                predecessor=old["id"],
                predecessor_commit=old["source_commit"],
                selection="Primary-source predecessor",
            )
        ]
    )
    report = runner().run(tmp_path, manifest, policy(), source=source)
    assert report["outcomes"][new["id"]]["status"] == "review"
    rec = json.loads((tmp_path / "data/models/lab__Family2.json").read_text())
    assert rec["published"]["context_length"] == 128000
    assert rec["predecessor_review"]["error"] == "ValueError"
    assert "tensor_diff_vs_predecessor" in rec
    assert not (tmp_path / "data/models/lab__Family2.yaml").exists()
    assert not list((tmp_path / "diagrams").glob("*.svg"))
    snapshot = files(tmp_path)
    assert runner().run(tmp_path, manifest, policy(), source=source) == report
    assert files(tmp_path) == snapshot


def test_upstream_change_after_freeze_never_analyzes_a_different_revision(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    selection = dict(id=rec["id"], source_commit="b" * 40, predecessor=None)
    report = runner().run(tmp_path, dict(models=[copy.deepcopy(selection)]), policy(), source=source)
    assert not source.analyzed
    assert report["outcomes"][rec["id"]]["status"] == "review"
    assert not list((tmp_path / "data/models").glob("*.json"))
