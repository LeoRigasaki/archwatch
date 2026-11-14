"""Regression: position-encoding labels in code-drawn diagrams come only from published facts."""

import json
from pathlib import Path

from archwatch import diagram

ROOT = Path(__file__).resolve().parents[1]


def rec(stem):
    return json.loads((ROOT / "data/models" / f"{stem}.json").read_text())


def test_kimi_nope_mla_is_never_labelled_rope():
    for stem in ("moonshotai__Kimi-K3.1-Pro", "moonshotai__Kimi-K3"):
        record = rec(stem)
        assert record["published"]["mla_use_nope"] is True
        svg = diagram.render(record)
        assert "RoPE" not in svg
        assert "NoPE" in svg


def test_rope_mla_keeps_its_rope_label():
    v = {"mla_kv_lora_rank": 512, "mla_qk_rope_head_dim": 64}
    assert diagram._mla_cache_line(v).endswith("64 RoPE")
    assert "NoPE" in diagram._mla_cache_line(dict(v, mla_use_nope=True))


def test_attention_without_published_rope_gets_no_rope_label():
    svg = diagram.render(rec("nvidia__Nemotron-H-8B-Base-8K"))
    assert "RoPE" not in svg
    assert diagram._rope_line({}) == "position encoding not published"
    assert diagram._rope_line({"rope_theta": 500000.0}) == "RoPE theta 500,000"
    assert diagram._rope_line({"rope_theta": 1e6, "partial_rotary_factor": 0.5}) == "partial RoPE 0.5, theta 1,000,000"
