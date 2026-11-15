import json
from pathlib import Path

import yaml

from archwatch.explainer import check, diff_input, format_numbers, input_hash, word_limit

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((ROOT / "config/providers.yaml").read_text())["explainer"]


def record(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def kimi():
    return diff_input(record("moonshotai__Kimi-K3.1-Pro"))


def test_digest_uses_only_the_diff_with_plain_labels_and_directions():
    data = kimi()
    assert set(data) == {"predecessor", "successor", "changes", "weights", "exact_values"}
    by = {c["what"]: c for c in data["changes"]}
    assert data["changes"][0]["what"] == "layer mix"  # most important architectural change first
    assert "model type" not in by
    assert by["sparse-attention token budget"]["before"] == "not stated for Kimi-K3"
    total = by["total parameters"]
    assert (total["before"], total["after"], total["direction"]) == ("2.7799T", "2.7801T", "up")  # never equal
    assert by["growing cache per token"]["after"] == "33 KB"
    assert data["weights"] == "5 weight groups added (179M parameters); none removed."


def test_fields_in_review_state_the_conflict():
    data = diff_input(record("LiquidAI__LFM3-1.7B"))
    context = next(c for c in data["changes"] if c["what"] == "context length")
    assert context["before"] == "in review for LFM2.5-1.2B-Instruct: config says 128,000, card says 32,768"


def test_namespace_rename_replaces_misleading_tensor_totals():
    data = diff_input(record("google__diffusiongemma-26B-A4B-it"))
    assert "namespace changed (language_model to decoder)" in data["weights"]
    assert "25,823,781,228" not in json.dumps(data)


def test_check_catches_invented_numbers_and_claims():
    data, limit = kimi(), word_limit(kimi(), SETTINGS)
    ok = "Total parameters went up from 2.7799T to 2.7801T, active parameters from 104.2B to 104.4B."
    assert check(ok, data, max_words=limit)["passed"]
    for bad, problem in [
        ("Total parameters rose by 178,526,208.", None),  # 178,526,208 is in the diff: allowed
        ("Total parameters rose by 0.006%.", "number 0.006"),
        ("Total parameters reached 2.9T.", "number 2.9T"),  # does not round from any diff value
        ("The cache grew to 34 KB.", "number 34KB"),
        ("The cache grew to 33 KB, with 179M parameters added.", None),
        ("It was introduced, having been absent.", 'awkward or raw wording "having been"'),
        ("The new model is faster.", 'claim word "faster"'),
        ("The dsa_index_topk is 2,048.", 'raw field or tensor name "dsa_index_topk"'),
        ("The head dim is 128.", 'awkward or raw wording "head dim"'),
        ("It adds layers.*.self_attn.indexer.wk tensors.", "raw field or tensor name"),
        ("Five tensor groups were added.", 'number word "five"'),
        ("- Total parameters changed.", "plain prose only"),
    ]:
        result = check(bad, data, max_words=limit)
        if problem is None:
            assert result["passed"], result
        else:
            assert any(problem in p for p in result["problems"]), (bad, result["problems"])


def test_direction_words_must_match_every_change_in_the_sentence():
    data = diff_input(record("LiquidAI__LFM3-1.7B"))
    limit = word_limit(data, SETTINGS)
    wrong = "Attention heads decreased from 32 to 16, and head size from 64 to 128."
    result = check(wrong, data, max_words=limit)
    assert any('"decreased" contradicts head size' in p for p in result["problems"])
    right = "Attention heads went down from 32 to 16, while head size went up from 64 to 128."
    assert check(right, data, max_words=limit)["passed"]


def test_word_limit_scales_with_changes_and_numbers_are_only_formatted():
    data = kimi()
    expected = SETTINGS["max_words_base"] + SETTINGS["max_words_per_change"] * len(data["changes"])
    assert word_limit(data, SETTINGS) == min(SETTINGS["max_words_cap"], expected)
    assert format_numbers("from 27648 to 33792 bytes, 2048.") == "from 27,648 to 33,792 bytes, 2048."
    assert format_numbers("total 2780110363392.") == "total 2,780,110,363,392."


def test_saved_examples_pass_and_match_their_current_input():
    saved = sorted((ROOT / "results/explainer").glob("*.json"))
    assert len(saved) == 3
    for path in saved:
        note = json.loads(path.read_text())
        data = diff_input(record(note["model"].replace("/", "__")))
        assert note["input_sha256"] == input_hash(data) and note["passed"]
        assert check(note["text"], data, max_words=word_limit(data, SETTINGS))["passed"]


def test_explainer_is_not_on_the_site_yet():
    assert "explainer" not in (ROOT / "build_site.py").read_text()
