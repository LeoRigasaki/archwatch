# Diagram generation v2: graph zoom panels and fixed layout

Specs come from published record fields (`archwatch.diagram_spec`). Each zoom panel and the
"Each layer" box is a small graph: nodes, edges and captions. Gemini (`models.diagram` in
`config/providers.yaml`) writes the SVG. `archwatch.diagram_checks` runs 22 code checks, and failures
go back for at most two repairs.

| Model | Result | Notes |
|---|---|---|
| Kimi-K3.1-Pro | v1 pass | all 22 checks |
| Nemotron-H-8B-Base-8K | v1 pass | all 22 checks |
| LFM3-1.7B | not generated | HTTP 503 (provider overloaded) |
| Llama-3.1-8B | not generated | HTTP 429 (provider daily quota) |

The new checks cover: every spec edge is drawn and no other edge is, each arrow connects its two boxes
in the right direction, the fixed sections (title, stack, Each layer, zooms) are in their fixed
places, each repeat count appears once, and callouts sit beside their part.

Budget: every call is recorded in `data/gemini-requests.json` and counts against the model's daily
limit. The provider's 429 arrived at 13 recorded calls on 2026-11-14 (Pacific), below the configured
20. The closed route-test PR's calls predate the shared ledger and most likely fell on the same quota
day. The remaining two models resume with `tools/diagram_gen.py --models ...` after the daily reset.

Previews: `results/diagram-gen/<model>/vN.png`. The first run (flat panels) is archived in
`results/diagram-gen-v1/`.
