# LFM findings: two distinct source disagreements

## Imported prototype: FFN-width conflict

The saved `data/models/LiquidAI__LFM2.5-1.2B-Instruct.json` record contains:

- Source commit: `b9d816ab510f9a28dca37cb182e9da8b0226e13c`.
- Config evidence: `intermediate_size = 12,288`.
- Tensor evidence: `feed_forward.w1` rows = `8,192`.
- Verifier status: `conflict`; confidence 0.3; `published = false`.

The field is absent from the record's `published` mapping. The successor's diff explicitly uses the weight-derived value and reports the predecessor conflict. This is a useful example of checking a config claim against the released tensors.

## Earlier primary-source audit: context inconsistency

The separate local research audit checked `LiquidAI/LFM2.5-1.2B-Thinking` and `LiquidAI/LFM2.5-350M`. Their captured model cards advertise **32,768 tokens**, while the pinned configs specify **128,000**. Both source statements are retained; a config limit alone does not establish supported deployment context.

Private evidence is available under `research-reference/private-notes/research/huggingface-verification-report.md` and its Hugging Face source manifest. It is a primary-source disagreement retained for investigation, not a gold label derived from another person's gallery.

These context findings concern different variants from the imported Instruct record's FFN-width disagreement. Keep their model IDs, fields and source revisions separate in the README story.
