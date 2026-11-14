# Primary-source gold exact match

Late-access addendum on the original unchanged first-run implementation. Gold entered from primary sources after the run, before viewing extracted values or changing code; reported separately from the seven strictly pre-run-frozen models.

Published fields: 11/16 exact matches across 1 labels.

| Field | Exact matches | Labeled | Not published |
|---|---:|---:|---:|
| activation | 1 | 1 | 0 |
| context_length | 1 | 1 | 0 |
| dense_ffn_intermediate | 0 | 1 | 0 |
| dense_layers | 0 | 1 | 1 |
| ffn_gated | 0 | 1 | 1 |
| head_dim | 1 | 1 | 0 |
| hidden_size | 1 | 1 | 0 |
| moe_active_experts | 1 | 1 | 0 |
| moe_expert_intermediate | 0 | 1 | 1 |
| moe_num_experts | 1 | 1 | 0 |
| moe_shared_experts | 0 | 1 | 1 |
| num_attention_heads | 1 | 1 | 0 |
| num_kv_heads | 1 | 1 | 0 |
| num_layers | 1 | 1 | 0 |
| params_total | 1 | 1 | 0 |
| vocab_size | 1 | 1 | 0 |

| Model | Exact matches | Labeled | Revision matches |
|---|---:|---:|---|
| meta-llama/Llama-4-Maverick-17B-128E | 11 | 16 | True |

## meta-llama/Llama-4-Maverick-17B-128E

- `moe_shared_experts`: expected `1`, got `None` (not published).
- `moe_expert_intermediate`: expected `8192`, got `None` (not published).
- `dense_layers`: expected `24`, got `None` (not published).
- `dense_ffn_intermediate`: expected `16384`, got `8192` (different value).
- `ffn_gated`: expected `True`, got `None` (not published).
