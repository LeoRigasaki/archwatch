"""The config's nominal FFN width and its instantiated width are distinct."""

import pytest

from archwatch import extract, verify


@pytest.mark.parametrize(
    "adjust,multiplier,width,expected",
    [
        (True, 1.0, 12288, 8192),
        (False, 1.0, 12288, 12288),
        (True, 1.25, 12288, 10240),
        (True, 1.0, 12290, 8448),
        (True, None, 12290, 8193),
    ],
)
def test_lfm_effective_width(adjust, multiplier, width, expected):
    config = {
        "model_type": "lfm2",
        "num_hidden_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "intermediate_size": width,
        "block_auto_adjust_ff_dim": adjust,
        "block_ffn_dim_multiplier": multiplier,
        "block_multiple_of": 256,
    }
    facts = extract.extract(config)
    assert facts["ffn_intermediate"][0] == expected
    if adjust:
        assert facts["ffn_intermediate_raw"][0] == width


def test_lfm_adjusted_config_and_weights_agree():
    facts = extract.extract(
        {
            "model_type": "lfm2",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "intermediate_size": 12288,
            "block_auto_adjust_ff_dim": True,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
        }
    )
    decision = verify.check(
        "ffn_intermediate",
        [
            {"value": facts["ffn_intermediate"][0], "evidence": facts["ffn_intermediate"][1]},
            {"value": 8192, "evidence": "weights:feed_forward.w1 rows"},
        ],
    )
    assert decision["status"] == "verified"
    assert decision["published"] is True
