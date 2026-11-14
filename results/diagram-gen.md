# Spec-driven diagram generation (diagram-gen-v1)

Each spec is built by code from published record fields (`archwatch.diagram_spec`); Gemini
(`models.diagram` in `config/providers.yaml`) writes the full SVG; code checks it
(`archwatch.diagram_checks`); failed checks go back to Gemini for at most two repairs.

| Model | Result | Versions | Failures fixed by repair |
|---|---|---|---|
| Kimi-K3.1-Pro | pass | v1 fail, v2 pass | "Decoder" (not a spec word); "96x192" glued token |
| LFM3-1.7B | pass | v1 fail, v2 pass | "decoder_layer" key shown as text |
| Nemotron-H-8B-Base-8K | pass | v1 fail, v2 fail, v3 pass | wrong Mamba-2/MLP bar order in the 11-layer group; stray words |
| Llama-3.1-8B | pass | v1 pass | none |

15 checks per version: SVG safety, canvas size, numbers from spec, words from spec, do_not_include,
layer sum, layer bars match spec bottom to top, zoom panels, callouts, text overlap, covered text,
lines through text, leader lines, 16 px canvas margin, font size >= 13.5 px.

Requests: 9 of the 12 budgeted (8 SVGs, 1 HTTP 503 on the first LFM3 call, resumed later).
The bar-order check was added after the first pass: it caught a Nemotron v2 that the original
layer-count check had passed. Kimi, Llama and LFM3 versions were rechecked with it; Kimi v1-v2 and
Nemotron v1-v2 were generated before the prompt named the exact `data-layer` label and order.

Spec numbers match the hand-written kit specs (`docs/diagram-kit/specs`) except one kit
inconsistency: the kit's "5.7 MB" plain-MHA cache uses 10^6 while its "33 KB" uses 1024; the
builder uses 1024 for both (5.4 MB).

Previews: `results/diagram-gen/<model>/vN.png`. Not human-reviewed yet.
