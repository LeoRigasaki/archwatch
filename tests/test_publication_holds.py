import json
from pathlib import Path

import pytest
import yaml

from archwatch.publication import PublicationError, held, holds, render_values

ROOT = Path(__file__).resolve().parents[1]


def test_every_hold_has_a_reason_and_names_real_records():
    data = yaml.safe_load((ROOT / "config/publication.yaml").read_text())["holds"]
    assert set(data) == {"generation_types", "model_types"}
    for group in data.values():
        for key, entry in group.items():
            assert len(entry["reason"]) > 60, key
            for repo in entry["records"]:
                assert (ROOT / "data/models" / (repo.replace("/", "__") + ".json")).exists(), repo


def test_hold_list_matches_the_previous_hard_coded_list():
    h = holds()
    assert set(h["generation_types"]) == {"block_diffusion", "latent_recurrent"}
    assert set(h["model_types"]) == {
        "deepseek_v41",
        "deepseek_v4",
        "qwen4_exp",
        "mistral_large4",
        "llama4",
        "glm5_next",
        "gemma4",
    }


def test_held_records_raise_with_their_reason_and_others_do_not():
    for path in sorted((ROOT / "data/models").glob("*.json")):
        record = json.loads(path.read_text())
        reason = held(record["published"])
        listed = any(record["id"] in entry["records"] for group in holds().values() for entry in group.values())
        assert bool(reason) == listed, record["id"]
        if reason:
            with pytest.raises(PublicationError, match="not qualified"):
                render_values(record)


def test_zero_shared_experts_needs_both_config_and_weight_evidence():
    import json

    from archwatch.verify import verify

    rec = json.loads((ROOT / "data/models/LiquidAI__LFM2.5-24B-A2B-Base.json").read_text())
    facts = {k: (v["value"], v["evidence"]) for k, v in rec["facts"].items()}
    facts.pop("moe_shared_experts", None)
    wcount = rec["params"]["counted_from_weights"]

    def shared(flag, wc, wshape):
        out = verify(dict(facts, _config_names_shared_experts=(flag, "config:*")), wc, wshape, None, {"facts": {}}, {})
        return next((c for c in out if c["field"] == "moe_shared_experts"), None)

    zero = shared(False, wcount, {})
    assert zero["value"] == 0 and zero["status"] == "verified"
    assert shared(True, wcount, {}) is None  # the config names shared experts: no zero is inferred
    with_tensors = dict(wcount, by_component=dict(wcount["by_component"], moe_shared_experts=5))
    found = shared(False, with_tensors, {"shared_experts": 1})
    assert found is None or found["value"] != 0


def test_lfm3_base_is_released_with_its_moe_predecessor():
    import json

    pred = json.loads((ROOT / "data/models/LiquidAI__LFM2.5-24B-A2B-Base.json").read_text())
    assert pred["published"]["moe_shared_experts"] == 0
    base = json.loads((ROOT / "data/models/LiquidAI__LFM3-1.7B-Base.json").read_text())
    render_values(base)  # passes the publication gate
    assert "block_shapes" in base["published"]


def test_absent_tie_key_is_not_evidence_and_weights_decide():
    from archwatch.extract import extract_with_usage

    facts, _ = extract_with_usage({"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1, "vocab_size": 4})
    assert "tied_embeddings" not in facts
    rec = json.loads((ROOT / "data/models/google__gemma-3-1b-pt.json").read_text())
    check = next(c for c in rec["checks"] if c["field"] == "tied_embeddings")
    assert check["value"] is True and check["candidates"][0]["evidence"].startswith("weights:")
