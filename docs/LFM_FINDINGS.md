# LFM investigation: effective dimensions and a separate context question

## Resolved: LFM2.5-Instruct FFN normalization

Model: `LiquidAI/LFM2.5-1.2B-Instruct`, source commit `b9d816ab510f9a28dca37cb182e9da8b0226e13c`.

The first extractor treated the config's nominal FFN width as the instantiated width. It reported a conflict between 12,288 in config and 8,192 rows in the released `feed_forward.w1` tensor. Investigation found an explicit config convention, not evidence of a Liquid AI config mistake.

The [pinned config](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/b9d816ab510f9a28dca37cb182e9da8b0226e13c/config.json) sets `block_auto_adjust_ff_dim: true`, `block_ffn_dim_multiplier: 1.0` and `block_multiple_of: 256`. Hugging Face's official [Lfm2MLP implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py) applies:

1. `int(2 * nominal_width / 3)` when auto-adjust is enabled.
2. If the multiplier is not `None`, multiply and truncate, then round up to the configured multiple.

For this checkpoint: `12,288 -> 8,192 -> 8,192`, already divisible by 256. That agrees with the tensor shape.

`extract.py` now retains `ffn_intermediate_raw` as diagnostic evidence and publishes the effective dimension only through the verifier. The live-refreshed record reports `ffn_intermediate = 8,192`, status `verified`, with config-derived and weight-shape candidates. Parameter totals are unchanged.

Regression coverage in `tests/test_lfm_dimensions.py` includes auto-adjust disabled, non-default multipliers, rounding boundaries, a `None` multiplier and agreement with weights. In particular, rounding belongs inside the non-`None` multiplier branch, matching the source implementation.

The interview story is: the verifier exposed an assumption in the extractor, primary-source investigation explained it, and normalization plus regression tests resolved it.

## Retained historical finding: context statements for other variants

The earlier private primary-source audit captured model cards advertising **32,768 tokens** while pinned configs specified **128,000** for `LiquidAI/LFM2.5-1.2B-Thinking` and `LiquidAI/LFM2.5-350M`. A config position limit alone does not establish the lab's supported or evaluated deployment context.

This is a separate historical source disagreement involving different model variants and a different field. It was not re-audited in the four-model run described above; current cards may have changed. Do not describe it as a confirmed model defect or as the resolved Instruct FFN issue.

Private snapshots and revision links are retained in `research-reference/gallery-prototype/research/huggingface-verification-report.md` and its Hugging Face source manifest. They are investigation references, not gold labels derived from another person's gallery.
