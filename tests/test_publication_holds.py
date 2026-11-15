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
