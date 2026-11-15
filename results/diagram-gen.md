# Diagram generation v3: balanced columns, callouts in the left panel, lines in the gaps

Specs come from published record fields (`archwatch.diagram_spec`). Zoom panels and the "Each layer"
box are small graphs. The spec's `layout` section fixes the page geometry: one zoom column, or two
when there are more than three panels, split to balance estimated heights. Gemini (`models.diagram`)
writes the SVG and is shown the Nemotron-H diagram as a style-only reference
(`config/diagram-gen/house-style/`). `archwatch.diagram_checks` runs 24 code checks, with up to two
repairs.

| Model | Result | Output tokens |
|---|---|---|
| Kimi-K3.1-Pro | v1 pass (two zoom columns) | 12,175 |
| LFM3-1.7B | v1 pass | 9,099 |
| Nemotron-H-8B-Base-8K | v1 pass | 7,872 |
| Llama-3.1-8B | v1 pass | 5,911 |

New in this round:
- no line crosses, or runs within 7.5 px of, a box it does not connect to
- the left column and the tallest zoom column are within 25% in height
- each callout sits inside the left panel, at most 340 px right of its target and within 60 px of level

On the earlier Kimi v2 SVG, the line check flags the two DSA lines that hugged the "output gate"
and "Linear 512" boxes; on the Nemotron reference it flags nothing.

Requests: 4 (one per model, no repairs), all on the 2026-11-15 ledger day with the new key.
Previews: `results/diagram-gen/<model>/v1.png`. Earlier runs are in `results/diagram-gen-v1/` and
`results/diagram-gen-v2/`.
