"""Hand-checked numbers. If any of these change, the extractor or calculator changed meaning."""
import json
from pathlib import Path

from archwatch import calc, card, extract, weights

DATA = Path(__file__).resolve().parent.parent / "data" / "models"


def test_gqa_and_sliding_and_conv_cache():
    v = {"layer_type_counts": {"full_attention": 6, "sliding_attention": 5, "short_conv": 13},
         "num_kv_heads": 4, "head_dim": 128, "num_attention_heads": 16, "num_layers": 24,
         "hidden_size": 2048, "sliding_window": 256, "short_conv_kernel": 7, "context_length": 262144}
    m = calc.cache_model(v)
    assert m["grows_per_token_bytes"] == 2 * 4 * 128 * 2 * 6          # 12,288 B
    assert m["constant_bytes"] == 256 * 2 * 4 * 128 * 2 * 5 + 2048 * 7 * 2 * 13


def test_mla_dsa_and_kda_cache():
    v = {"layer_type_counts": {"kda_linear": 69, "gated_dsa_mla": 24}, "num_kv_heads": 96, "head_dim": 192,
         "v_head_dim": 128, "num_attention_heads": 96, "num_layers": 93, "mla_kv_lora_rank": 512,
         "mla_qk_rope_head_dim": 64, "dsa_index_head_dim": 128, "linear_attn_num_heads": 96,
         "linear_attn_head_dim": 128, "linear_attn_conv_kernel": 4, "context_length": 1048576}
    m = calc.cache_model(v)
    assert m["grows_per_token_bytes"] == (512 + 64 + 128) * 2 * 24    # 33,792 B
    assert m["constant_bytes"] == (96 * 128 * 128 * 4 + 3 * 96 * 128 * 3 * 2) * 69
    assert m["naive_mha_per_token_bytes"] == 93 * 96 * (192 + 128) * 2


def test_one_based_layer_lists():
    cfg = {"text_config": {"num_hidden_layers": 4, "hidden_size": 8, "num_attention_heads": 2,
                           "linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}}}
    f = extract.extract(cfg)
    assert f["layer_types"][0] == ["kda_linear", "kda_linear", "kda_linear", "full_attention_mla"]
    assert f["layer_index_base_in_config"][0] == 1


def test_packed_4bit_counts_two_params_per_byte_and_skips_scales():
    t = {"model.layers.1.mlp.experts.0.w1.weight_packed": {"dtype": "U8", "shape": [10, 4]},
         "model.layers.1.mlp.experts.0.w1.weight_scale": {"dtype": "U8", "shape": [10, 1]},
         "model.embed_tokens.weight": {"dtype": "BF16", "shape": [5, 8]}}
    c = weights.count(t, 4)
    assert c["total"] == 10 * 4 * 2 + 5 * 8


def test_card_rounding_tolerance():
    assert card.parse_number("2.8T") == (2.8e12, 0.05e12)
    assert card.parse_number("1048576") == (1048576, 0)


def test_records_match_cards_when_present():
    for name in ("moonshotai__Kimi-K3.1-Pro", "LiquidAI__LFM3-1.7B"):
        f = DATA / f"{name}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            assert not [c for c in rec["checks"] if c["status"] == "conflict"], name
