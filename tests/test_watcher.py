"""Watcher behavior against a fake Hub; no model payloads or network access."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from archwatch import hub, watcher
from archwatch.artifacts import save_record

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "data/models/LiquidAI__LFM2.5-1.2B-Instruct.json").read_text())


def policy():
    value = watcher.load_policy(ROOT / "config/watcher.yaml")
    value["core_publishers"] = ["lab"]
    return value


def model(repo, date, context=128000, sha="a" * 40):
    record = copy.deepcopy(BASE)
    record.pop("hub_metadata", None)  # Synthetic repositories have their own provenance.
    record.update(id=repo, source_commit=sha, hub_created=date, analyzed_at="2026-01-01T00:00:00+00:00")
    record["published"]["context_length"] = context
    record["facts"]["context_length"]["value"] = context
    for check in record["checks"]:
        if check["field"] == "context_length":
            check["value"] = context
            # This fixture's context is synthetic, not the donor's card claim.
            check.update(
                status="single source",
                published=True,
                confidence=0.7,
                candidates=[{"value": context, "evidence": "config:max_position_embeddings"}],
            )
    record["review_queue"] = [c for c in record["review_queue"] if c["field"] != "context_length"]
    # Tests flip ffn_gated themselves; the synthetic fixture starts without it, whatever the donor publishes.
    record["published"].pop("ffn_gated", None)
    record["checks"] = [c for c in record["checks"] if c["field"] != "ffn_gated"]
    return record


class Source:
    def __init__(self, records):
        self.records = {r["id"]: copy.deepcopy(r) for r in records}
        self.configs = {r["id"]: {"model_type": "toy", "context": r["published"]["context_length"]} for r in records}
        self.codes = {}
        self.infos = {
            r["id"]: {
                "sha": r["source_commit"],
                "created_at": r["hub_created"],
                "files": ["config.json", "model.safetensors"],
                "pipeline_tag": "text-generation",
                "downloads": 10000,
            }
            for r in records
        }
        self.analyzed, self.lookups, self.lists = [], [], []
        self.fail = None

    def listing(self, **kwargs):
        self.lists.append(kwargs)
        return [
            SimpleNamespace(id=repo)
            for repo in sorted(self.records)
            if not kwargs.get("author") or repo.startswith(kwargs["author"] + "/")
        ]

    def metadata(self, repo):
        self.lookups.append(repo)
        return copy.deepcopy(self.infos[repo])

    def config(self, repo, commit):
        return copy.deepcopy(self.configs[repo])

    def code(self, repo, commit, files):
        return self.codes.get(repo, {})

    def analyze(self, repo, info):
        self.analyzed.append(repo)
        if self.fail:
            error, self.fail = self.fail, None
            raise error
        return copy.deepcopy(self.records[repo])

    def tensor_diff(self, *args):
        return dict(added=[], removed=[], added_count=0, removed_count=0, added_total=0, removed_total=0)


def seed(root, record):
    path = root / "data/models" / (record["id"].replace("/", "__") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


def state(root, repo):
    return json.loads((root / "data/seen.json").read_text())["models"][repo]


def files(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_policy_has_all_ten_publishers_and_outside_discovery():
    configured = watcher.load_policy(ROOT / "config/watcher.yaml")
    assert len(configured["core_publishers"]) == 10
    assert configured["trending_limit"] >= 50
    source = Source([model("lab/Family1", "2025-01-01T00:00:00Z"), model("outside/Other1", "2025-01-02T00:00:00Z")])
    candidates, errors = watcher.discover(source, policy())
    assert not errors
    assert {c["pool"] for c in candidates} == {"core", "outside"}
    assert any(q["sort"] == "trending_score" and not q.get("author") for q in source.lists)
    assert any(q["sort"] == "created_at" and not q.get("author") for q in source.lists)


def test_analyzes_new_release_selects_predecessor_and_restart_is_no_op(tmp_path):
    old = model("lab/Family1", "2025-01-01T00:00:00Z")
    new = model("lab/Family2", "2026-01-01T00:00:00Z", 256000, "b" * 40)
    source = Source([old, new])
    seed(tmp_path, old)
    result = watcher.run(tmp_path, policy(), source=source, repos=[new["id"]], process=True)
    assert result["outcomes"][new["id"]] == "ready_for_diagram"
    entry = state(tmp_path, new["id"])
    assert entry["predecessor"] == old["id"]
    assert entry["last_successful_commit"] == "b" * 40
    record = json.loads((tmp_path / "data/models/lab__Family2.json").read_text())
    assert record["predecessor"]["id"] == old["id"]
    before = files(tmp_path)
    watcher.run(tmp_path, policy(), source=source, repos=[new["id"]], process=True)
    assert files(tmp_path) == before
    assert source.analyzed == [new["id"]]


def test_explicit_reprocess_refreshes_schema_at_same_commit_and_keeps_ready(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    source.records[rec["id"]]["config_coverage"] = {"consumed": ["context"], "unconsumed": []}
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True, reprocess=True)
    saved = json.loads((tmp_path / "data/models/lab__Family1.json").read_text())
    assert saved["config_coverage"]["consumed"] == ["context"]
    assert state(tmp_path, rec["id"])["status"] == "ready_for_diagram"
    assert source.analyzed == [rec["id"], rec["id"]]
    before = files(tmp_path)
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert files(tmp_path) == before


def test_reprocessing_successor_refreshes_predecessor_formats_and_seen_state(tmp_path):
    from archwatch.notifications import annotate_report, format_message, snapshot

    old = model("lab/Family1", "2025-01-01T00:00:00Z")
    new = model("lab/Family2", "2026-01-01T00:00:00Z", 256000, "b" * 40)
    save_record(old, tmp_path / "data/models", with_yaml=True, diagrams=tmp_path / "diagrams")
    source = Source([old, new])
    watcher.run(tmp_path, policy(), source=source, repos=[new["id"]], process=True)
    before = files(tmp_path)
    # A new tensor-layout rule learns gating at the SAME source commit.
    fixed = source.records[old["id"]]
    fixed["published"]["ffn_gated"] = True
    fixed["checks"].append(
        dict(
            field="ffn_gated",
            value=True,
            status="single source",
            published=True,
            confidence=0.7,
            candidates=[dict(value=True, evidence="weights:gate tensors")],
        )
    )
    notification_before = snapshot(tmp_path)
    result = watcher.run(tmp_path, policy(), source=source, repos=[new["id"]], process=True, reprocess=True)
    assert result["outcomes"][old["id"]] == "ready_for_diagram"
    content = format_message(annotate_report(tmp_path, result, notification_before))
    assert "Processed: 2" in content and "Processed: lab/Family1" in content and "Processed: lab/Family2" in content
    after = files(tmp_path)
    for name in ("data/models/lab__Family1.json", "data/models/lab__Family1.yaml", "diagrams/lab__Family1.svg"):
        assert before[name][0] != after[name][0], name
    old_state = state(tmp_path, old["id"])
    assert old_state["last_successful_commit"] == old["source_commit"]
    assert old_state["status"] == "ready_for_diagram"
    assert old_state["architecture_hash"] == watcher.architecture_hash(fixed, source.configs[old["id"]], {})
    watcher.run(tmp_path, policy(), source=source, repos=[old["id"], new["id"]], process=True)
    assert files(tmp_path) == after


@pytest.mark.parametrize("files_present", [[], ["config.json"], ["model.safetensors"], ["config.json", "model.gguf"]])
def test_admission_requires_existing_config_and_safetensors(tmp_path, files_present):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    source.infos[rec["id"]]["files"] = files_present
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert source.lookups == [rec["id"]]
    assert not source.analyzed
    assert state(tmp_path, rec["id"])["status"] == "rejected"
    assert not list((tmp_path / "data/models").glob("*.json"))


@pytest.mark.parametrize("variant", ["Instruct", "AWQ"])
def test_finetunes_and_conversions_with_matching_architecture_are_skipped(tmp_path, variant):
    old = model("lab/Family1", "2025-01-01T00:00:00Z")
    new = model("outside/Family1-" + variant, "2026-01-01T00:00:00Z", sha="b" * 40)
    seed(tmp_path, old)
    source = Source([old, new])
    source.configs[new["id"]].update(torch_dtype="float16", quantization_config={"bits": 4}, _name_or_path="copy")
    watcher.run(tmp_path, policy(), source=source, repos=[new["id"]], process=True)
    entry = state(tmp_path, new["id"])
    assert entry["status"] == "duplicate"
    assert entry["architecture_match"] == old["id"]
    assert len(list((tmp_path / "data/models").glob("*.json"))) == 1
    assert not list((tmp_path / "diagrams").glob("*.svg"))


def test_unknown_config_or_custom_code_difference_is_not_silently_deduplicated(tmp_path):
    old = model("lab/Family1", "2025-01-01T00:00:00Z")
    new = model("outside/Family2", "2026-01-01T00:00:00Z", sha="b" * 40)
    seed(tmp_path, old)
    source = Source([old, new])
    source.configs[new["id"]]["new_mechanism"] = True
    watcher.run(tmp_path, policy(), source=source, repos=[new["id"]], process=True)
    assert state(tmp_path, new["id"])["status"] == "review"
    assert not (tmp_path / "data/models/outside__Family2.json").exists()


def test_failures_do_not_advance_success_and_are_retryable(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    source.fail = OSError("temporary failure")
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    entry = state(tmp_path, rec["id"])
    assert entry["last_observed_commit"] == rec["source_commit"]
    assert entry["last_successful_commit"] is None
    assert entry["status"] == "error"
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert state(tmp_path, rec["id"])["last_successful_commit"] == rec["source_commit"]


def test_changed_commit_is_processed_but_observation_alone_is_not_success(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    source.infos[rec["id"]]["sha"] = "b" * 40
    source.records[rec["id"]]["source_commit"] = "b" * 40
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]])
    entry = state(tmp_path, rec["id"])
    assert entry["last_observed_commit"] == "b" * 40
    assert entry["last_successful_commit"] == "a" * 40
    assert entry["status"] == "pending"
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert state(tmp_path, rec["id"])["status"] == "recorded"  # evidence refresh, no diagram generation
    assert state(tmp_path, rec["id"])["last_successful_commit"] == "b" * 40


def test_unverified_pattern_stays_in_review(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    rec["published"].pop("layer_pattern_per_layer")
    next(c for c in rec["checks"] if c["field"] == "layer_pattern_per_layer")["status"] = "single source"
    watcher.run(tmp_path, policy(), source=Source([rec]), repos=[rec["id"]], process=True)
    assert state(tmp_path, rec["id"])["status"] == "review"
    assert state(tmp_path, rec["id"])["last_successful_commit"] is None
    assert not list((tmp_path / "diagrams").glob("*.svg"))


def test_predecessor_excludes_other_org_family_newer_and_ambiguous_dates():
    candidate = dict(id="lab/Family3", family="family", created_at="2026-03-01T00:00:00Z")
    catalog = {
        r["id"]: r
        for r in [
            dict(id="lab/Family1", family="family", created_at="2026-01-01T00:00:00Z"),
            dict(id="lab/Family2", family="family", created_at="2026-02-01T01:00:00+01:00"),
            dict(id="other/Family2", family="family", created_at="2026-02-28T00:00:00Z"),
            dict(id="lab/Other2", family="other", created_at="2026-02-28T00:00:00Z"),
            dict(id="lab/Family4", family="family", created_at="2026-04-01T00:00:00Z"),
        ]
    }
    sizes = {key: (1e9, False) for key in [*catalog, candidate["id"], "lab/Family2-alt"]}
    assert watcher.predecessor(candidate, catalog, sizes)[0]["id"] == "lab/Family2"
    catalog["lab/Family2-alt"] = dict(id="lab/Family2-alt", family="family", created_at="2026-02-01T00:00:00Z")
    assert watcher.predecessor(candidate, catalog, sizes)[0] is None
    assert "Ambiguous" in watcher.predecessor(candidate, catalog, sizes)[1]


def test_predecessor_must_be_the_same_size_class():
    candidate = dict(id="lab/Model3-1.7B", family="model", created_at="2026-03-01T00:00:00Z")
    catalog = {
        "lab/Model2-1.2B": dict(id="lab/Model2-1.2B", family="model", created_at="2026-01-01T00:00:00Z"),
        "lab/Model2-24B-MoE": dict(id="lab/Model2-24B-MoE", family="model", created_at="2026-02-01T00:00:00Z"),
        "lab/Model2-1B-MoE": dict(id="lab/Model2-1B-MoE", family="model", created_at="2026-02-15T00:00:00Z"),
    }
    sizes = {
        candidate["id"]: (1.73e9, False),
        "lab/Model2-1.2B": (1.17e9, False),
        "lab/Model2-24B-MoE": (23.97e9, True),  # newer, but 14x larger and MoE
        "lab/Model2-1B-MoE": (1.5e9, True),  # same size, other kind
    }
    assert watcher.predecessor(candidate, catalog, sizes)[0]["id"] == "lab/Model2-1.2B"
    del sizes["lab/Model2-1.2B"]  # size unknown: cannot be a predecessor
    pred, reason = watcher.predecessor(candidate, catalog, sizes)
    assert pred is None and "size class" in reason


def test_lfm2_5_24b_is_never_chosen_for_lfm3_1_7b():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    records = {
        json.loads(p.read_text())["id"]: json.loads(p.read_text())
        for p in (root / "data/models").glob("LiquidAI__*.json")
    }
    sizes = {key: watcher.size_class(rec) for key, rec in records.items()}
    catalog = {
        key: dict(
            id=key,
            family="lfm",
            created_at=rec["hub_created"],
            derived=bool(watcher.derivative_relation(rec.get("hub_metadata") or {})),
        )
        for key, rec in records.items()
    }
    pred, _ = watcher.predecessor(catalog["LiquidAI/LFM3-1.7B-Base"], catalog, sizes)
    assert pred is None  # LFM2.5-1.2B's base model is not in the catalog; fine-tunes compare only with fine-tunes
    pred, _ = watcher.predecessor(catalog["LiquidAI/LFM3-1.7B"], catalog, sizes)
    assert pred["id"] == "LiquidAI/LFM2.5-1.2B-Instruct"
    for target in ("LiquidAI/LFM3-1.7B-Base", "LiquidAI/LFM3-1.7B"):
        pred, _ = watcher.predecessor(catalog[target], catalog, sizes)
        assert pred is None or pred["id"] != "LiquidAI/LFM2.5-24B-A2B-Base"
        assert (records[target].get("predecessor") or {}).get("id") != "LiquidAI/LFM2.5-24B-A2B-Base"


def test_pool_budget_limits_processing_and_retains_unprocessed_candidates(tmp_path):
    records = [model(f"lab/Family{i}", f"2025-01-0{i}T00:00:00Z", 10000 * i, str(i) * 40) for i in range(1, 6)]
    records += [model(f"outside/Other{i}", f"2025-02-0{i}T00:00:00Z", 100000 * i, str(i) * 40) for i in range(1, 4)]
    source = Source(records)
    # Distinct families avoid extra predecessor analyses affecting this budget assertion.
    pol = policy()
    pol["family_overrides"] = {r["id"]: r["id"].split("/")[-1] for r in records}
    watcher.run(tmp_path, pol, source=source, repos=[r["id"] for r in records], process=True)
    assert len(source.analyzed) == 5
    assert sum(repo.startswith("lab/") for repo in source.analyzed) == 3
    assert sum(repo.startswith("outside/") for repo in source.analyzed) == 2
    assert sum(state(tmp_path, r["id"])["status"] == "pending" for r in records) == 3


def test_discovery_checks_missing_repositories_and_unsafe_ids(tmp_path):
    source = Source([])
    watcher.run(tmp_path, policy(), source=source, repos=["lab/missing"], process=True)
    assert state(tmp_path, "lab/missing")["status"] == "error"
    assert not source.analyzed
    with pytest.raises(ValueError, match="Invalid repository"):
        watcher.run(tmp_path, policy(), source=source, repos=["../../escape"])


class RateLimit(Exception):
    def __init__(self, retry):
        self.response = SimpleNamespace(status_code=429, headers={"retry-after": retry})


def test_rate_limit_retries_are_bounded_and_long_delays_survive_restart(tmp_path):
    waits, calls = [], []
    source = watcher.HubSource(policy(), api=object(), wait=waits.append)

    def transient():
        calls.append(True)
        if len(calls) == 1:
            raise RateLimit("2")
        return "ok"

    assert source.request(transient) == "ok"
    assert waits == [2]
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    fake = Source([rec])
    fake.fail = RateLimit("3600")
    watcher.run(tmp_path, policy(), source=fake, repos=[rec["id"]], process=True)
    before = files(tmp_path)
    result = watcher.run(tmp_path, policy(), source=fake, repos=[rec["id"]], process=True)
    assert result["outcomes"][rec["id"]] == "rate_limit_deferred"
    assert files(tmp_path) == before


def test_explicit_snapshot_reads_never_resolve_a_moving_head(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "CACHE", tmp_path)
    monkeypatch.setattr(hub, "repo_info", lambda *a, **k: (_ for _ in ()).throw(AssertionError("head resolved again")))
    urls = []

    def fetch(url, headers=None):
        urls.append(url)
        return b"{}"

    monkeypatch.setattr(hub, "_get", fetch)
    assert hub.get_json("lab/model", "config.json", revision="a" * 40) == {}
    assert "/resolve/" + "a" * 40 + "/" in urls[0]


def test_pending_and_previously_seen_repos_are_revisited_after_leaving_listings(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]])
    source.listing = lambda **kwargs: []
    watcher.run(tmp_path, policy(), source=source, process=True)
    assert state(tmp_path, rec["id"])["status"] == "ready_for_diagram"
    source.infos[rec["id"]]["sha"] = "b" * 40
    source.records[rec["id"]]["source_commit"] = "b" * 40
    watcher.run(tmp_path, policy(), source=source, process=True)
    assert state(tmp_path, rec["id"])["last_successful_commit"] == "b" * 40
    assert source.analyzed == [rec["id"], rec["id"]]


def test_per_commit_failure_budget_and_explicit_review_retry(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    for _ in range(3):
        source.fail = OSError("retryable")
        watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert state(tmp_path, rec["id"])["status"] == "review"
    assert len(source.analyzed) == 3
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True, retry_review=True)
    assert state(tmp_path, rec["id"])["status"] == "ready_for_diagram"


@pytest.mark.parametrize("status", [401, 403, 404])
def test_unreadable_or_missing_sources_are_not_queued(tmp_path, status):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    error = RuntimeError("access failure")
    error.code = status
    source.config = lambda *args: (_ for _ in ()).throw(error)
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert state(tmp_path, rec["id"])["status"] == ("rejected" if status == 404 else "review")
    assert not source.analyzed
    assert state(tmp_path, rec["id"])["last_successful_commit"] is None


def test_partial_listing_errors_are_visible_and_checkpoint_is_stable(tmp_path):
    rec = model("outside/Other1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    listing = source.listing

    def partial(**kwargs):
        if kwargs.get("author"):
            raise OSError("listing unavailable")
        return listing(**kwargs)

    source.listing = partial
    report = watcher.run(tmp_path, policy(), source=source)
    assert len(report["discovery_errors"]) == 2
    assert json.loads((tmp_path / "data/seen.json").read_text())["discovery_errors"]
    before = files(tmp_path)
    watcher.run(tmp_path, policy(), source=source)
    assert files(tmp_path) == before


def test_analysis_revision_mismatch_is_not_published(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    source.infos[rec["id"]]["sha"] = "b" * 40
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    assert state(tmp_path, rec["id"])["status"] == "error"
    assert state(tmp_path, rec["id"])["last_successful_commit"] is None
    assert not list((tmp_path / "data/models").glob("*.json"))


def test_unmodeled_mechanism_change_in_existing_repo_requires_review(tmp_path):
    rec = model("lab/Family1", "2025-01-01T00:00:00Z")
    source = Source([rec])
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    source.infos[rec["id"]]["sha"] = "b" * 40
    source.records[rec["id"]]["source_commit"] = "b" * 40
    # The historical pinned config remains available for bootstrap/signature comparison.
    original_config = source.config
    source.config = lambda repo, commit: (
        original_config(repo, commit) | ({"new_mechanism": True} if commit == "b" * 40 else {})
    )
    watcher.run(tmp_path, policy(), source=source, repos=[rec["id"]], process=True)
    entry = state(tmp_path, rec["id"])
    assert entry["status"] == "review"
    assert entry["last_successful_commit"] == "a" * 40
