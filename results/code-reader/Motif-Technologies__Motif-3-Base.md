# Motif-Technologies/Motif-3-Base

Commit `7f7531c7f2b0259b527477c72715ed1805dd95ce`. Code: repo.

| Field | Value | Verdict | Model | Cited lines | Problems |
|---|---|---|---|---|---|
| num_layers | `53` | review | gemini-3.5-flash-lite | configuration_motif.py:136-136 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; num_layers is a weight shape, but the claim names no tensor to check it against |
| hidden_size | `4096` | review | gemini-3.5-flash-lite | configuration_motif.py:134-134 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; hidden_size is a weight shape, but the claim names no tensor to check it against |
| vocab_size | `220160` | review | gemini-3.5-flash-lite | configuration_motif.py:133-133 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; vocab_size is a weight shape, but the claim names no tensor to check it against |
| context_length | `262144` | review | gemini-3.5-flash-lite | configuration_motif.py:140-140 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |
| num_attention_heads | `80` | review | gemini-3.5-flash-lite | configuration_motif.py:137-137 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; num_attention_heads is a weight shape, but the claim names no tensor to check it against |
| num_kv_heads | `16` | review | gemini-3.5-flash-lite | configuration_motif.py:138-138 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; num_kv_heads is a weight shape, but the claim names no tensor to check it against |
| head_dim | `192` | review | gemini-3.5-flash-lite | configuration_motif.py:154-154 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; head_dim is a weight shape, but the claim names no tensor to check it against |
| v_head_dim | `128` | review | gemini-3.5-flash-lite | configuration_motif.py:183-183 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; v_head_dim is a weight shape, but the claim names no tensor to check it against |
| attention_type | `"MLA"` | review | gemini-3.5-flash-lite | modeling_motif.py:535-540 | quote does not equal the cited lines; config attention_cls = 'gdla', claim says 'MLA' |
| moe_num_experts | `384` | review | gemini-3.5-flash-lite | configuration_motif.py:158-158 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; moe_num_experts is a weight shape, but the claim names no tensor to check it against |
| moe_active_experts | `8` | review | gemini-3.5-flash-lite | configuration_motif.py:159-159 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |
| moe_shared_experts | `1` | review | gemini-3.5-flash-lite | configuration_motif.py:160-160 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |
| moe_expert_intermediate | `1280` | review | gemini-3.5-flash-lite | configuration_motif.py:162-162 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; moe_expert_intermediate is a weight shape, but the claim names no tensor to check it against |
| ffn_intermediate | `12288` | review | gemini-3.5-flash-lite | configuration_motif.py:135-135 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; ffn_intermediate is a weight shape, but the claim names no tensor to check it against |
| activation | `"poly_norm"` | review | gemini-3.5-flash-lite | configuration_motif.py:139-139 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |
| normalization_type | `"RMSNorm"` | verified | gemini-3.5-flash-lite | modeling_motif.py:139-140 | none |
| mla_kv_lora_rank | `512` | review | gemini-3.5-flash-lite | configuration_motif.py:181-181 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; mla_kv_lora_rank is a weight shape, but the claim names no tensor to check it against |
| mla_q_lora_rank | `1024` | review | gemini-3.5-flash-lite | configuration_motif.py:180-180 | cites configuration_motif.py, the configuration class (defaults), not the modelling code; mla_q_lora_rank is a weight shape, but the claim names no tensor to check it against |
| mla_qk_rope_head_dim | `64` | review | gemini-3.5-flash-lite | configuration_motif.py:182-182 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |
| sliding_window | `128` | review | gemini-3.5-flash-lite | configuration_motif.py:148-148 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |
| tied_embeddings | `false` | review | gemini-3.5-flash-lite | configuration_motif.py:144-144 | cites configuration_motif.py, the configuration class (defaults), not the modelling code |

Field decisions:

- activation: **review**  (failed a check or conflicts)
- attention_type: **review**  (failed a check or conflicts)
- context_length: **review**  (failed a check or conflicts)
- ffn_intermediate: **review**  (failed a check or conflicts)
- head_dim: **review**  (failed a check or conflicts)
- hidden_size: **review**  (failed a check or conflicts)
- mla_kv_lora_rank: **review**  (failed a check or conflicts)
- mla_q_lora_rank: **review**  (failed a check or conflicts)
- mla_qk_rope_head_dim: **review**  (failed a check or conflicts)
- moe_active_experts: **review**  (failed a check or conflicts)
- moe_expert_intermediate: **review**  (failed a check or conflicts)
- moe_num_experts: **review**  (failed a check or conflicts)
- moe_shared_experts: **review**  (failed a check or conflicts)
- normalization_type: **verified** "RMSNorm" (verified)
- num_attention_heads: **review**  (failed a check or conflicts)
- num_kv_heads: **review**  (failed a check or conflicts)
- num_layers: **review**  (failed a check or conflicts)
- sliding_window: **review**  (failed a check or conflicts)
- tied_embeddings: **review**  (failed a check or conflicts)
- v_head_dim: **review**  (failed a check or conflicts)
- vocab_size: **review**  (failed a check or conflicts)
