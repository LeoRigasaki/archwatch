# Code reader evaluation

Prompt `code_reader_v1` (sha256 `45bed7ae3932`). Reference: our published records for models whose fields the deterministic rules already read. Nothing here is published.

Columns: **value correct** = claimed value equals our published value (claims whose field has a reference); **citation valid** = cited file is fetched modelling code, the line range exists and the quote equals those lines; **config agrees** = cited config keys exist and agree; **weight check** = the cited line builds a module whose tensors, mapped from the code's own module tree, have consistent shapes (fields with a weight only); **passes all checks** = every check above with no conflict (the version cap is reported separately); **wrong but passes** = passes every check yet disagrees with our record (the verifier must keep this at 0).

| Model | Reader | Code | Claims | Value correct | Citation valid | Config agrees | Weight check | Passes all checks | Wrong but passes | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | gemini-3.5-flash-lite | transformers v4.43.0 | 10 | 9/9 (100%) | 10/10 (100%) | 10/10 (100%) | 1/7 (14%) | 4/10 (40%) | 0 | 0 |
| Qwen3-0.6B-Base | gemini-3.5-flash-lite | transformers v4.51.0 | 8 | 0/8 (0%) | 8/8 (100%) | 0/8 (0%) | 5/6 (83%) | 0/8 (0%) | 0 | 0 |
| DeepSeek-V2-Lite | gemini-3.5-flash-lite | repo | 5 | 2/5 (40%) | 5/5 (100%) | 2/5 (40%) | 3/3 (100%) | 2/5 (40%) | 0 | 2 |
| LFM3-1.7B-Base | gemini-3.5-flash-lite | repo | 12 | 1/11 (9%) | 11/12 (92%) | 2/12 (17%) | 5/7 (71%) | 1/12 (8%) | 0 | 1 |
| Nemotron-H-8B-Base-8K | gemini-3.5-flash-lite | repo | 11 | 0/9 (0%) | 11/11 (100%) | 1/11 (9%) | 3/7 (43%) | 1/11 (9%) | 0 | 1 |
| Llama-3.1-8B | gemini-3.8-flash | transformers v4.43.0 | error: Configured free daily request cap reached | | | | | | | |
| Qwen3-0.6B-Base | gemini-3.8-flash | transformers v4.51.0 | error: Configured free daily request cap reached | | | | | | | |
| DeepSeek-V2-Lite | gemini-3.8-flash | repo | error: Configured free daily request cap reached | | | | | | | |
| LFM3-1.7B-Base | gemini-3.8-flash | repo | 12 | 11/11 (100%) | 12/12 (100%) | 12/12 (100%) | 7/7 (100%) | 12/12 (100%) | 0 | 11 |
| Nemotron-H-8B-Base-8K | gemini-3.8-flash | repo | error: Configured free daily request cap reached | | | | | | | |

**gemini-3.5-flash-lite overall:** 46 claims; value correct 12/42 (29%); citation valid 45/46 (98%); weight check 17/30 (57%); passes all checks 8/46 (17%); wrong but passes 0.

**gemini-3.8-flash overall:** 12 claims; value correct 11/11 (100%); citation valid 12/12 (100%); weight check 7/7 (100%); passes all checks 12/12 (100%); wrong but passes 0.

## Per field

| Field | gemini-3.5-flash-lite: correct / passes | gemini-3.8-flash: correct / passes |
|---|---|---|
| activation | 1/3 / 1/3 | 1/1 / 1/1 |
| context_length | 1/4 / 1/4 | 1/1 / 1/1 |
| ffn_intermediate | 1/3 / 0/3 | 1/1 / 1/1 |
| head_dim | 1/4 / 0/4 | 1/1 / 1/1 |
| hidden_size | 1/5 / 0/5 | 1/1 / 1/1 |
| layer_types | 0/0 / 1/1 | 0/0 / 1/1 |
| normalization_type | 2/2 / 3/4 | 1/1 / 1/1 |
| num_attention_heads | 1/4 / 0/4 | 1/1 / 1/1 |
| num_kv_heads | 1/4 / 0/4 | 1/1 / 1/1 |
| num_layers | 1/5 / 1/5 | 1/1 / 1/1 |
| sliding_window | 0/1 / 0/2 | 1/1 / 1/1 |
| tied_embeddings | 1/2 / 1/2 | 0/0 / 0/0 |
| vocab_size | 1/5 / 0/5 | 1/1 / 1/1 |

## Claims that failed or disagree with the record

- Llama-3.1-8B / gemini-3.5-flash-lite / ffn_intermediate = `14336` (record `14336`), modeling_llama.py:229-229: the cited lines build no module that reaches the model's state dict; code was not read at the model's own revision (see flags); capped at review
- Llama-3.1-8B / gemini-3.5-flash-lite / hidden_size = `4096` (record `4096`), modeling_llama.py:285-285: the cited lines build no module that reaches the model's state dict; code was not read at the model's own revision (see flags); capped at review
- Llama-3.1-8B / gemini-3.5-flash-lite / num_attention_heads = `32` (record `32`), modeling_llama.py:286-286: code was not read at the model's own revision (see flags); capped at review; no module cited and no heads x head_dim check against the projection rows was possible
- Llama-3.1-8B / gemini-3.5-flash-lite / head_dim = `128` (record `128`), modeling_llama.py:287-287: code was not read at the model's own revision (see flags); capped at review; no module cited and no heads x head_dim check against the projection rows was possible
- Llama-3.1-8B / gemini-3.5-flash-lite / num_kv_heads = `8` (record `8`), modeling_llama.py:288-288: code was not read at the model's own revision (see flags); capped at review; no module cited and no heads x head_dim check against the projection rows was possible
- Llama-3.1-8B / gemini-3.5-flash-lite / vocab_size = `128256` (record `128256`), modeling_llama.py:841-841: the cited lines build no module that reaches the model's state dict; code was not read at the model's own revision (see flags); capped at review
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / vocab_size = `"vocab_size"` (record `151936`), modeling_qwen3.py:480-480: vocab_size needs a number, got 'vocab_size'; value 'vocab_size' is a config key name, not its value
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / hidden_size = `"hidden_size"` (record `1024`), modeling_qwen3.py:480-480: hidden_size needs a number, got 'hidden_size'; value 'hidden_size' is a config key name, not its value
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / num_layers = `"num_hidden_layers"` (record `28`), modeling_qwen3.py:481-483: num_layers needs a number, got 'num_hidden_layers'; value 'num_hidden_layers' is a config key name, not its value
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / head_dim = `"head_dim"` (record `128`), modeling_qwen3.py:177-177: head_dim needs a number, got 'head_dim'; value 'head_dim' is a config key name, not its value; no module cited and no heads x head_dim check against the projection rows was possible
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / num_attention_heads = `"num_attention_heads"` (record `16`), modeling_qwen3.py:183-185: num_attention_heads needs a number, got 'num_attention_heads'; value 'num_attention_heads' is a config key name, not its value
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / num_kv_heads = `"num_key_value_heads"` (record `8`), modeling_qwen3.py:186-188: num_kv_heads needs a number, got 'num_key_value_heads'; value 'num_key_value_heads' is a config key name, not its value
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / activation = `"hidden_act"` (record `"silu"`), modeling_qwen3.py:91-91: value 'hidden_act' is a config key name, not its value; config hidden_act = 'silu', claim says 'hidden_act'
- Qwen3-0.6B-Base / gemini-3.5-flash-lite / context_length = `"max_position_embeddings"` (record `32768`), modeling_qwen3.py:323-323: context_length needs a number, got 'max_position_embeddings'; value 'max_position_embeddings' is a config key name, not its value
- DeepSeek-V2-Lite / gemini-3.5-flash-lite / vocab_size = `"config.vocab_size"` (record `102400`), modeling_deepseek.py:1415-1417: vocab_size needs a number, got 'config.vocab_size'
- DeepSeek-V2-Lite / gemini-3.5-flash-lite / hidden_size = `"config.hidden_size"` (record `2048`), modeling_deepseek.py:1415-1417: hidden_size needs a number, got 'config.hidden_size'
- DeepSeek-V2-Lite / gemini-3.5-flash-lite / num_layers = `"config.num_hidden_layers"` (record `27`), modeling_deepseek.py:1418-1423: num_layers needs a number, got 'config.num_hidden_layers'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / hidden_size = `"config.hidden_size"` (record `2048`), modeling_lfm3.py:58-58: hidden_size needs a number, got 'config.hidden_size'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / vocab_size = `"config.vocab_size"` (record `128000`), modeling_lfm3.py:595-595: vocab_size needs a number, got 'config.vocab_size'; the cited lines build no module that reaches the model's state dict
- LFM3-1.7B-Base / gemini-3.5-flash-lite / num_layers = `"config.num_hidden_layers"` (record `24`), modeling_lfm3.py:599-599: num_layers needs a number, got 'config.num_hidden_layers'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / context_length = `"config.max_position_embeddings"` (record `262144`), modeling_lfm3.py:80-80: context_length needs a number, got 'config.max_position_embeddings'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / num_attention_heads = `"config.num_attention_heads"` (record `16`), modeling_lfm3.py:225-225: num_attention_heads needs a number, got 'config.num_attention_heads'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / num_kv_heads = `"config.num_key_value_heads"` (record `4`), modeling_lfm3.py:226-226: num_kv_heads needs a number, got 'config.num_key_value_heads'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / head_dim = `"config.head_dim"` (record `128`), modeling_lfm3.py:221-221: head_dim needs a number, got 'config.head_dim'; no module cited and no heads x head_dim check against the projection rows was possible
- LFM3-1.7B-Base / gemini-3.5-flash-lite / ffn_intermediate = `"config.intermediate_size"` (record `7680`), modeling_lfm3.py:152-152: ffn_intermediate needs a number, got 'config.intermediate_size'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / normalization_type = `"RMSNorm"` (record `"RMSNorm"`), modeling_lfm3.py:52-52: quote does not equal the cited lines
- LFM3-1.7B-Base / gemini-3.5-flash-lite / sliding_window = `"config.sliding_window"` (record `256`), modeling_lfm3.py:232-232: sliding_window needs a number, got 'config.sliding_window'
- LFM3-1.7B-Base / gemini-3.5-flash-lite / tied_embeddings = `"True"` (record `true`), modeling_lfm3.py:804-804: tied_embeddings needs true or false, got 'True'
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / num_layers = `"config.num_hidden_layers"` (record `52`), modeling_nemotron_h.py:1283-1283: num_layers needs a number, got 'config.num_hidden_layers'
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / hidden_size = `"config.hidden_size"` (record `4096`), modeling_nemotron_h.py:1282-1282: hidden_size needs a number, got 'config.hidden_size'
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / vocab_size = `"config.vocab_size"` (record `131072`), modeling_nemotron_h.py:1282-1282: vocab_size needs a number, got 'config.vocab_size'
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / context_length = `"config.max_position_embeddings"` (record `8192`), modeling_nemotron_h.py:851-851: context_length needs a number, got 'config.max_position_embeddings'
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / num_attention_heads = `"config.num_attention_heads"` (record `32`), modeling_nemotron_h.py:844-844: num_attention_heads needs a number, got 'config.num_attention_heads'; no module cited and no heads x head_dim check against the projection rows was possible
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / num_kv_heads = `"config.num_key_value_heads"` (record `8`), modeling_nemotron_h.py:849-849: num_kv_heads needs a number, got 'config.num_key_value_heads'; no module cited and no heads x head_dim check against the projection rows was possible
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / head_dim = `"config.attention_head_dim"` (record `128`), modeling_nemotron_h.py:845-846: head_dim needs a number, got 'config.attention_head_dim'; no module cited and no heads x head_dim check against the projection rows was possible
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / ffn_intermediate = `"config.intermediate_size"` (record `21504`), modeling_nemotron_h.py:806-806: ffn_intermediate needs a number, got 'config.intermediate_size'; the cited lines build no module that reaches the model's state dict
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / activation = `"config.mlp_hidden_act"` (record `"relu2"`), modeling_nemotron_h.py:809-809: config mlp_hidden_act = 'relu2', claim says 'config.mlp_hidden_act'
- Nemotron-H-8B-Base-8K / gemini-3.5-flash-lite / sliding_window = `"config.sliding_window"` (record `null`), modeling_nemotron_h.py:996-996: sliding_window needs a number, got 'config.sliding_window'
