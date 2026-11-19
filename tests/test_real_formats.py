"""Formats read by deterministic rules, each checked on the model's real files at a pinned commit.

Fixtures under tests/fixtures/real/<repo>@<commit>/ are the Hub files at that commit (config.json or
params.json, README, modelling code, index files) plus the safetensors headers read with range requests.
Files whose licences do not allow redistribution here (NVIDIA's model files, including modelling code) are
listed in sources.json as {url at the pinned commit, sha256}: tests download them into cache/ (gitignored) and
verify the hash. The Hub client is pointed at the fixture folder; nothing else touches the network.
"""

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

import pytest

from archwatch import extract, hub, pipeline, weights

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "real"


def fixture_dir(repo):
    (match,) = FIXTURES.glob(repo.replace("/", "__") + "@*")
    return match


DOWNLOADS = Path(__file__).resolve().parents[1] / "cache" / "test-fixtures"


def sourced(folder, name):
    """A file that is not redistributed here: fetched from its pinned Hub URL into cache/ and hash-checked."""
    manifest = folder / "sources.json"
    entry = json.loads(manifest.read_text()).get(name) if manifest.exists() else None
    if entry is None:
        return None
    target = DOWNLOADS / folder.name / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(entry["url"], timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise AssertionError(f"{name}: downloaded bytes do not match the pinned sha256")
        target.write_bytes(data)
    data = target.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], f"{name}: cached copy changed"
    return data.decode("utf-8")


def read(folder, name):
    plain, packed = folder / name, folder / (name + ".gz")
    if plain.exists():
        return plain.read_text()
    if packed.exists():
        return gzip.decompress(packed.read_bytes()).decode()
    return sourced(folder, name)


@pytest.fixture(scope="session", autouse=True)
def prefetched():
    """Download every manifest-listed file once, in parallel (Ultra lists 113 per-shard index files)."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [
        (f, n)
        for f in FIXTURES.iterdir()
        if (f / "sources.json").exists()
        for n in json.loads((f / "sources.json").read_text())
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda job: sourced(*job), jobs))


@pytest.fixture
def offline(monkeypatch):
    def use(repo):
        folder = fixture_dir(repo)
        info = json.loads((folder / "repo_info.json").read_text())
        assert folder.name.endswith("@" + info["sha"])
        monkeypatch.setattr(hub, "get_text", lambda r, path, revision=None: read(folder, path))
        tensors = folder / "tensors.json.gz"
        monkeypatch.setattr(
            hub,
            "safetensors_headers",
            lambda r, info=None: hub.weight_shards(r, info) and json.loads(gzip.decompress(tensors.read_bytes())),
        )
        return pipeline.analyze(repo, info=info), info

    return use


def check(record, field):
    return next(c for c in record["checks"] if c["field"] == field)


def test_gpt2_legacy_keys_and_transposed_conv1d_weights(offline):
    record, _ = offline("openai-community/gpt2-xl")
    pub = record["published"]
    assert record["facts"]["num_layers"] == {"value": 48, "evidence": "config:n_layer"}
    assert record["facts"]["hidden_size"]["evidence"] == "config:n_embd"
    assert record["facts"]["num_attention_heads"]["evidence"] == "config:n_head"
    assert record["facts"]["context_length"]["evidence"] == "config:n_positions"
    assert pub["num_layers"] == 48 and pub["hidden_size"] == 1600 and pub["num_attention_heads"] == 25
    assert pub["head_dim"] == 64 and pub["context_length"] == 1024 and pub["attention_type"] == "MHA"
    # Conv1D stores [in, out]: c_fc is [1600, 6400], so the inner width is the last dimension
    assert pub["ffn_intermediate"] == 6400 and pub["ffn_gated"] is False
    assert check(record, "ffn_intermediate")["status"] == "verified"
    # the [1, 1, 1024, 1024] attn.bias causal masks are buffers: 1,557,611,200 is GPT-2 XL's parameter count
    assert pub["params_total"] == 1_557_611_200 and pub["tied_embeddings"] is True
    assert record["review_queue"] == []


def test_gpt2_causal_mask_buffers_are_not_parameters():
    mask = {"shape": [1, 1, 1024, 1024], "dtype": "F32"}
    assert weights.logical_elements("h.0.attn.bias", mask) is None
    assert weights.logical_elements("h.0.attn.c_attn.bias", {"shape": [4800], "dtype": "F32"}) == 4800
    assert weights._layer_idx("h.17.mlp.c_fc.weight") == 17


def test_nemotron3_nano_moe_blocks_in_the_pattern_string(offline):
    record, _ = offline("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    pub = record["published"]
    assert pub["layer_type_counts"] == {"mamba2": 23, "moe": 23, "full_attention": 6}
    assert check(record, "layer_pattern_per_layer")["status"] == "verified"  # E blocks hold mixer.experts tensors
    assert pub["moe_num_experts"] == 128 and pub["moe_active_experts"] == 6 and pub["moe_shared_experts"] == 1
    assert pub["moe_expert_intermediate"] == 1856 and pub["ffn_gated"] is False  # up -> relu2 -> down
    assert pub["dense_layers"] == 0  # Mamba and attention blocks have no FFN; there are no MLP blocks
    assert record["review_queue"] == []


def test_nemotron3_super_latent_moe_and_mtp_pattern(offline):
    record, _ = offline("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    pub = record["published"]
    assert pub["moe_latent_dim"] == 1024 and check(record, "moe_latent_dim")["status"] == "verified"
    assert record["facts"]["moe_latent_dim"]["evidence"] == "config:moe_latent_size"
    # mtp_hybrid_override_pattern "*E": one attention block and one MoE block, stored as mtp.layers.0/1
    assert record["facts"]["mtp_block_types"]["value"] == ["full_attention", "moe"]
    assert check(record, "mtp_block_count")["status"] == "verified" and pub["mtp_block_count"] == 2
    counted = record["params"]["counted_from_weights"]
    assert pub["params_mtp"] == counted["mtp_params"] > 0
    assert counted["by_component"]["auxiliary"] == counted["mtp_params"]  # MTP blocks are not decoder weights
    assert counted["by_component"]["moe_latent_projection"] > 0


def test_nemotron3_ultra_block_list_large_config_and_per_shard_indexes(offline):
    record, info = offline("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4")
    pub = record["published"]
    # no num_hidden_layers key: the depth is the length of layers_block_type
    assert record["facts"]["num_layers"] == {"value": 108, "evidence": "derived:len(layers_block_type)"}
    assert pub["layer_type_counts"] == {"mamba2": 48, "moe": 48, "full_attention": 12}
    assert check(record, "layer_pattern_per_layer")["status"] == "verified"
    assert pub["moe_latent_dim"] == 2048  # experts are NVFP4: [5120, 1024] bytes hold 2048 values per row
    assert record["facts"]["mtp_block_types"]["evidence"] == "config:mtp_layers_block_type"
    assert pub["mtp_block_count"] == 2 and pub["params_mtp"] > 0
    # the 6.9 MB config: its per-tensor quantization map is one consumed setting, not thousands of loose keys
    unconsumed = record["config_coverage"]["unconsumed"]
    assert not any(p.startswith("quantization_config.quantized_layers.") for p in unconsumed) and len(unconsumed) < 80
    # 113 per-shard weight maps: listed in the sources, and they name exactly the shards that were counted
    per_shard = hub.per_shard_indexes(info)
    assert len(per_shard) == 113 and set(per_shard) <= set(record["source_files"])
    assert set(hub.shard_index_map("x", info).values()) == set(hub.weight_shards("x", info))
    assert 545e9 < pub["params_total"] < 555e9 and record["review_queue"] == []


def test_per_shard_index_that_names_another_shard_is_rejected(monkeypatch):
    info = {"sha": "a", "files": ["model-00001-of-00002.json", "model-00001-of-00002.safetensors"]}
    monkeypatch.setattr(
        hub, "get_json", lambda r, p, revision=None: {"weight_map": {"x": "model-00002-of-00002.safetensors"}}
    )
    with pytest.raises(ValueError, match="another shard"):
        hub.weight_shards("r", info)


def test_nemotron35_super_vl_llm_config_and_vision_code_scope(offline):
    record, _ = offline("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    pub = record["published"]
    assert record["facts"]["num_layers"]["evidence"] == "derived:len(llm_config.layers_block_type)"
    assert pub["layer_type_counts"] == {"mamba2": 40, "moe": 40, "full_attention": 8}
    assert pub["context_length"] == 1_048_576 and pub["moe_latent_dim"] == 1024
    assert (
        record["facts"]["released_weights_format"]["evidence"] == "config:quantization_config"
    )  # root, not llm_config
    # modeling_radio.py is the vision tower: its SiLU MLP does not contradict the decoder's relu2
    assert pub["activation"] == "relu2"
    counted = record["params"]["counted_from_weights"]["by_component"]
    assert counted["vision"] > 0 and counted["vision_projector"] > 0  # mlp1 is the projector, not text weights
    assert [r["field"] for r in record["review_queue"]] == ["params_total"]  # 121.5B stored vs the card's "120B"


def test_mistral_large3_params_json_records_its_source(offline):
    record, _ = offline("mistralai/Mistral-Large-3-675B-Instruct-2512")
    pub, facts = record["published"], record["facts"]
    assert record["config_file"] == "params.json" and "params.json" in record["source_files"]
    assert facts["num_layers"] == {"value": 61, "evidence": "params.json:n_layers"}
    assert facts["hidden_size"]["evidence"] == "params.json:dim"
    assert facts["moe_num_experts"] == {"value": 128, "evidence": "params.json:moe.num_experts"}
    assert facts["moe_expert_intermediate"]["evidence"] == "params.json:moe.expert_hidden_dim"
    assert facts["dense_layers"] == {"value": 3, "evidence": "params.json:moe.first_k_dense_replace"}
    # this params.json has no max_seq_len, so the context falls back to max_position_embeddings, and says so
    assert facts["context_length"] == {"value": 294912, "evidence": "params.json:max_position_embeddings"}
    assert all("config:" not in str(f["evidence"]) for f in facts.values())
    assert pub["attention_type"] == "MLA" and pub["mla_cached_width_per_token"] == 576
    assert pub["layer_type_counts"] == {"full_attention_mla": 61}
    assert check(record, "layer_pattern_per_layer")["status"] == "verified"
    assert pub["moe_shared_experts"] == 1 and pub["ffn_gated"] is True
    assert record["review_queue"] == []


def test_mistral_params_prefer_the_served_sequence_length():
    facts, coverage = extract.extract_mistral_params(
        {"dim": 8, "n_layers": 2, "n_heads": 2, "max_seq_len": 262144, "max_position_embeddings": 294912}
    )
    assert facts["context_length"] == (262144, "params.json:max_seq_len")
    assert coverage["source"] == "params.json" and "max_position_embeddings" in coverage["unconsumed"]


def test_btl3_adapter_is_skipped_as_out_of_scope(offline):
    with pytest.raises(pipeline.OutOfScope, match="out of scope like fine-tunes"):
        offline("badtheorylabs/BTL-3")


def test_fixtures_redistribute_no_third_party_model_files():
    for folder in FIXTURES.iterdir():
        stored = {p.name.removesuffix(".gz") for p in folder.iterdir()}
        assert not [n for n in stored if n.endswith(".py")], folder.name  # modelling code is never copied here
        licence = (json.loads((folder / "repo_info.json").read_text()).get("license") or "").lower()
        if licence not in ("mit", "apache-2.0"):
            assert stored <= {"repo_info.json", "tensors.json", "sources.json"}, folder.name
        if (folder / "sources.json").exists():
            listed = json.loads((folder / "sources.json").read_text())
            assert not set(listed) & stored and all(
                e["url"].split("/resolve/")[1].startswith(folder.name.split("@")[1]) for e in listed.values()
            )
