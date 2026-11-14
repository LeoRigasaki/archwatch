# Frozen diagram route experiment

This protocol is committed before live generation. It is an eight-case, one-shot experiment, not the generation/repair loop or explainer. No cron or recurring queue is enabled.

## Inputs and routes

The four subjects in `config/providers.yaml` are the requested Kimi-K3.1-Pro, historical LFM3-1.7B, Nemotron-H-8B and Llama-3.1-8B records. Their published values must match their verifier decisions and pass the existing pattern/input gate. The LFM record is an explicitly requested private experiment subject; its watcher derivative status is not overridden for public admission.

Both routes receive the same frozen labels, semantic nodes, label membership and edge IDs, derived entirely from published fields. Layer-kind overviews are used rather than asking Gemini to invent detailed internals. Each label retains its source field/path and value in `spec.yaml`; all required labels must appear exactly once. The experiment includes the physical layer-kind compositions, dimensions and relevant FFN/mixer settings. It does not require displaying every field in the full record.

- **A:** Gemini writes the complete standalone SVG, including text, coordinates and styling, within the fixed content contract.
- **B:** Gemini writes JSON canvas/node geometry, label references/order and edge references; Python inserts the published label text and draws the SVG. The schema includes `nodes`, `groups`, `labels` and `edges`; groups are empty in this bounded overview experiment. Gemini supplies no architecture values.

The generator is Gemini 4 Hydrogen, resolved from configuration. The [official model catalog](https://aistudio.google.com/docs/models) and [free pricing](https://aistudio.google.com/docs/pricing#gemini-4-hydrogen) were checked; authenticated model metadata confirms `generateContent` support. The adapter uses standard [generateContent requests](https://aistudio.google.com/api-reference/generate-content), never the separately priced Batch API.

## Checks and selection, declared before outcomes

Each candidate is checked automatically for:

1. All visible labels/numbers match their allowed, source-linked published text, ignoring whitespace around colons and numeric thousands separators; geometric coordinates are not architecture facts.
2. Every required label, node and edge occurs once, with labels on the correct semantic node.
3. The visible per-kind layer counts sum to the published total.
4. Browser-measured text boxes do not overlap, including translated/rotated groups.
5. All rendered text/shapes/edges remain inside the bounded canvas.
6. Labels are visible, readable at 13px or larger, fit their nodes, and nodes do not overlap.

Active/external SVG content is rejected before browser loading. Browser network requests are blocked. Geometry is measured with installed Chrome through Playwright, at one CSS pixel per SVG unit. Node/edge identity checks do not establish exhaustive wiring semantics, contrast quality or general aesthetic quality; inspect the saved previews as well.

Select by most complete passes, then most individual check passes, then fewer output tokens. A tie remains a tie. If neither route produces a passing case, no route is selected. No selection occurs until all eight cases are generated and scored. Every failed case remains in the report; none is silently dropped.

## Budget and artifacts

Eight planned generation calls, one per model/route, zero architecture repairs, zero fallback calls. Limits and model IDs live in configuration. The adapter enforces the screenshot-derived daily cap, a 15-request weekly cap, four starts per minute and a conservative token bound. Reservations are written before transport; an uncertain/failed request still counts. Quota/provider failures stop the experiment rather than upgrading service or switching models. Reviews are deterministic; the diagram model never grades itself.

Each case saves `spec.yaml`, `prompt.txt`, `v1.response.txt`, `v1.metadata.json`, `v1.svg`, `v1.review.json` and a preview. B also saves `v1.layout.json`. Unsafe/malformed candidates retain raw text and a failed review instead of being loaded or published. Cached responses cannot be reused with changed input/prompt hashes. Nothing from this experiment is published in the gallery automatically.

```bash
uv run --frozen python tools/diagram_route_test.py --generate --score
# Recompute checks from cached outputs, with no generation calls:
uv run --frozen python tools/diagram_route_test.py --score
```

The result is `results/diagram-route-test.md`. A winning route is recorded in `config/diagram-route-choice.yaml` with both generation loop and explainer disabled. The user must see the results before those later stages begin.
