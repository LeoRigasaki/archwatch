# Diagram route comparison

Eight first attempts; same fixed facts and graph, one model; no repairs or fallback.

Generator: `gemini-4-hydrogen` from `config/providers.yaml`. Reviews are deterministic; no model grades its own output.

| Model | Route | Facts | Layer sum | No text overlap | Inside canvas | Coverage | Full pass | Output tokens |
|---|---|---|---|---|---|---|---|---:|
| moonshotai/Kimi-K3.1-Pro | A | fail | fail | fail | fail | fail | fail | 0 |
| moonshotai/Kimi-K3.1-Pro | B | fail | fail | fail | fail | fail | fail | 0 |
| LiquidAI/LFM3-1.7B | A | fail | fail | fail | fail | fail | fail | 0 |
| LiquidAI/LFM3-1.7B | B | fail | fail | fail | fail | fail | fail | 0 |
| nvidia/Nemotron-H-8B-Base-8K | A | fail | fail | fail | fail | fail | fail | 0 |
| nvidia/Nemotron-H-8B-Base-8K | B | fail | fail | fail | fail | fail | fail | 0 |
| meta-llama/Llama-3.1-8B | A | fail | fail | fail | fail | fail | fail | 0 |
| meta-llama/Llama-3.1-8B | B | fail | fail | fail | fail | fail | fail | 0 |

Selection rule: Most complete passes, then most individual check passes, then fewer output tokens; declared before generation.

Result: **incomplete; no route selected**.

Each case saves its frozen spec, prompt, raw response, SVG, review JSON and provider usage. Route B also saves its JSON layout. Geometric dimensions are not architecture facts; visible labels must match source-linked published values exactly. Input facts, node membership and edges are fixed for both routes.

The LFM3-1.7B record is the requested historical record, used only for this private experiment; its current watcher derivative status is not overridden or newly admitted. The other subjects are the requested established records, not the newest unqualified layouts.

These are four models and one sample per route. The checks test factual control, required-label coverage, layer totals and browser geometry, not exhaustive semantic architecture correctness or general aesthetic quality. A route failing validation is not publishable. No output is added to the public gallery by this experiment. The generation/repair loop and explainer remain disabled.

## moonshotai/Kimi-K3.1-Pro — Route A

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "97a266f8a1d14e91c04223675d6a0496406fc3556f2b3c16cb05aff54087558c"
}
```

## moonshotai/Kimi-K3.1-Pro — Route B

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "97a266f8a1d14e91c04223675d6a0496406fc3556f2b3c16cb05aff54087558c"
}
```

## LiquidAI/LFM3-1.7B — Route A

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "0bf2bf8b4186840ab41fb49345d76916fbd48f92f8a244484a6b8f5c9b32f3b5"
}
```

## LiquidAI/LFM3-1.7B — Route B

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "0bf2bf8b4186840ab41fb49345d76916fbd48f92f8a244484a6b8f5c9b32f3b5"
}
```

## nvidia/Nemotron-H-8B-Base-8K — Route A

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "afca43cbb3875715a5f2c03a72779b4adc018e2b091f836fd77d20c946b4c993"
}
```

## nvidia/Nemotron-H-8B-Base-8K — Route B

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "afca43cbb3875715a5f2c03a72779b4adc018e2b091f836fd77d20c946b4c993"
}
```

## meta-llama/Llama-3.1-8B — Route A

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "153845e275c85f3287d52d28e8a2e793430a19a1730f8d88b5564889d519cf2c"
}
```

## meta-llama/Llama-3.1-8B — Route B

```json
{
  "generated": false,
  "reviewer": "deterministic validators",
  "spec_hash": "153845e275c85f3287d52d28e8a2e793430a19a1730f8d88b5564889d519cf2c"
}
```
