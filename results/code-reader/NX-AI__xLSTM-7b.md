# NX-AI/xLSTM-7b

Commit `9dc507bd0939cf372a4a4f667335651d8e49dddb`. Code: transformers v4.54.0.
- Flag: transformers_version 4.48.0.dev0 is a development build, not a release
- Flag: modeling_xlstm.py is absent at v4.48.0; read at v4.54.0, the first later release

| Field | Value | Verdict | Model | Cited lines | Problems |
|---|---|---|---|---|---|
| num_layers | `32` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1382-1382 | num_layers is a weight shape, but the claim names no tensor to check it against; code was not read at the config's own transformers version |
| hidden_size | `4096` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1381-1381 | code was not read at the config's own transformers version |
| vocab_size | `50304` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1381-1381 | code was not read at the config's own transformers version |
| num_attention_heads | `8` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1061-1065 | code was not read at the config's own transformers version |
| mixer_type | `"mlstm"` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1178-1178 | code was not read at the config's own transformers version |
| ffn_intermediate | `10944` | review | gemini-3.5-flash-lite | modeling_xlstm.py:989-992 | code was not read at the config's own transformers version |
| activation | `"silu"` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1018-1018 | code was not read at the config's own transformers version |
| normalization_type | `"RMSNorm"` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1171-1177 | code was not read at the config's own transformers version |
| tied_embeddings | `false` | review | gemini-3.5-flash-lite | modeling_xlstm.py:1519-1519 | code was not read at the config's own transformers version |

Field decisions:

- activation: **review**  (failed a check or conflicts)
- ffn_intermediate: **review**  (failed a check or conflicts)
- hidden_size: **review**  (failed a check or conflicts)
- mixer_type: **review**  (failed a check or conflicts)
- normalization_type: **review**  (failed a check or conflicts)
- num_attention_heads: **review**  (failed a check or conflicts)
- num_layers: **review**  (failed a check or conflicts)
- tied_embeddings: **review**  (failed a check or conflicts)
- vocab_size: **review**  (failed a check or conflicts)
