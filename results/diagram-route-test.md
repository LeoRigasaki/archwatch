# Diagram route comparison

Eight first-response candidates; same fixed facts and graph, one model; no repairs or fallback.

Generator: `gemini-4-hydrogen` from `config/providers.yaml`. Reviews are deterministic; no model grades its own output.

| Model | Route | Facts | Layer sum | No text overlap | Inside canvas | Coverage | Full pass | Output tokens |
|---|---|---|---|---|---|---|---|---:|
| moonshotai/Kimi-K3.1-Pro | A | pass | pass | pass | pass | pass | pass | 2556 |
| moonshotai/Kimi-K3.1-Pro | B | pass | pass | pass | pass | pass | pass | 661 |
| LiquidAI/LFM3-1.7B | A | pass | pass | pass | pass | pass | pass | 2163 |
| LiquidAI/LFM3-1.7B | B | pass | pass | pass | pass | pass | pass | 519 |
| nvidia/Nemotron-H-8B-Base-8K | A | pass | pass | pass | pass | pass | pass | 2440 |
| nvidia/Nemotron-H-8B-Base-8K | B | pass | pass | pass | pass | pass | pass | 578 |
| meta-llama/Llama-3.1-8B | A | pass | pass | pass | pass | pass | pass | 1484 |
| meta-llama/Llama-3.1-8B | B | pass | pass | pass | pass | pass | pass | 348 |

Selection rule: Most complete passes, then most individual check passes, then fewer output tokens; declared before generation.

Result: **B**.

Each case saves its frozen spec, prompt, raw response, SVG, review JSON and provider usage. Route B also saves its JSON layout. Geometric dimensions are not architecture facts; visible labels must match source-linked published values, allowing colon whitespace and numeric thousands separators. Input facts, node membership and edges are fixed for both routes.

The LFM3-1.7B record is the requested historical record, used only for this private experiment; its current watcher derivative status is not overridden or newly admitted. The other subjects are the requested established records, not the newest unqualified layouts.

These are four models and one sample per route. The checks test factual control, required-label coverage, layer totals and browser geometry, not exhaustive semantic architecture correctness or general aesthetic quality. A route failing validation is not publishable. No output is added to the public gallery by this experiment. The generation/repair loop and explainer remain disabled.

Request accounting: **10 reserved attempts**, 8 successful candidates, 2 HTTP 503 failures. Failed requests count against the 15-request weekly cap. Manual resumption used the same prompts and reused completed outputs; zero architecture repairs, fallback or model-review calls.

A passed 4/4 cases (48/48 individual checks); B passed 4/4 (48/48). Output tokens: A 8,643, B 2,106. The selected route wins the predeclared token tie-break; this is not evidence of better visual quality.

## Saved previews

| Model | Full SVG (A) | JSON + Python (B) |
|---|---|---|
| moonshotai/Kimi-K3.1-Pro | [preview](diagram-route-test/moonshotai__Kimi-K3.1-Pro/A/v1.png) | [preview](diagram-route-test/moonshotai__Kimi-K3.1-Pro/B/v1.png) |
| LiquidAI/LFM3-1.7B | [preview](diagram-route-test/LiquidAI__LFM3-1.7B/A/v1.png) | [preview](diagram-route-test/LiquidAI__LFM3-1.7B/B/v1.png) |
| nvidia/Nemotron-H-8B-Base-8K | [preview](diagram-route-test/nvidia__Nemotron-H-8B-Base-8K/A/v1.png) | [preview](diagram-route-test/nvidia__Nemotron-H-8B-Base-8K/B/v1.png) |
| meta-llama/Llama-3.1-8B | [preview](diagram-route-test/meta-llama__Llama-3.1-8B/A/v1.png) | [preview](diagram-route-test/meta-llama__Llama-3.1-8B/B/v1.png) |
