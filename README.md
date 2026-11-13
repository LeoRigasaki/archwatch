# ArchWatch

ArchWatch extracts architecture facts from a model's own Hugging Face config, tensor headers, Python code and model card, checks their agreement, and generates an architecture page with evidence. Conflicting fields are withheld from the record's `published` dictionary and from architecture views.

The user-supplied prototype is integrated under `src/archwatch/`. The deterministic core has been run live on four models. Discovery, LLM readers, independent evaluation and the review queue remain open.

## Local demo

[site/index.html](site/index.html) is a self-contained two-model demo with generated SVGs, source evidence, predecessor/tensor diffs and an interactive cache calculator.

```bash
python -m http.server 4173 --directory site
```

The earlier earlier private research is private research material, separate from this demo. Owner-only access and anonymous HTTP 401 checks are recorded in [the access review](docs/deployment-access-review.json). Its hosted address is omitted here.

## Implemented components

| Stage | Current capability |
|---|---|
| Hub access | Metadata refreshed once per analysis; file/header reads and caches pinned to that source commit; bounded safetensors-header range requests |
| Config | Nested text configs, dense/hybrid/MoE, latent MoE, MLA/DSA/KDA, explicit layer lists and one-based indexing; LFM effective FFN dimensions |
| Weights | Tensor-shape parameter totals, supported packed-weight handling, component counts and active-parameter conventions |
| Code | Static AST classes/patterns with file and line citations; remote code is not executed |
| Model card | Structured table claims with rounding tolerances |
| Calculator | GQA, sliding-window, short-convolution, MLA, DSA-index and KDA cache estimates with stated assumptions |
| Verifier | Verified/single-source/conflict decisions; conflicts withheld; rendering inputs carry explicit publication decisions |
| Diff | Published-field and tensor-group diffs against an explicitly supplied predecessor |
| Diagram/site | Published inputs only; verified weight-kind layer pattern required; site cache estimates and field comparisons recomputed |
| Tooling | uv lock, ruff, pre-commit and offline PR CI configured; source-only ZIP exporter |

The input list is fixed in `watchlist.txt`. The watcher and LLM code/report readers are not implemented.

## Validation

[Results](results/README.md), [offline validation](results/prototype-validation.json) and [live-run comparison](results/live-validation.json) cover:

- `moonshotai/Kimi-K3` and `moonshotai/Kimi-K3.1-Pro`;
- `LiquidAI/LFM2.5-1.2B-Instruct` and `LiquidAI/LFM3-1.7B`.

On 13 November 2026, a fresh primary-source run reproduced all four source commits and exact parameter totals. The original 81 core verdicts were unchanged except for the resolved LFM FFN field. With added decisions for rendering inputs, the records now contain **61 verified checks, 67 single-source checks and zero conflicts**. These are agreement counts, not independent accuracy scores. The offline suite has **41 passing tests**.

The Kimi diff still identifies exactly **178,526,208 added DSA-indexer parameters**. The LFM2.5 config's nominal 12,288-wide FFN becomes **8,192** under its explicit auto-adjust convention, agreeing with the tensors. The extractor now handles that rule; the initial flag was an extractor gap, not evidence of a mistake by Liquid AI. See [the LFM investigation](docs/LFM_FINDINGS.md).

## Run and develop

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --python 3.12
uv run --frozen pre-commit install
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q --junitxml=results/test-results.xml
uv run --frozen python build_site.py
uv run --frozen python tools/summarize_prototype.py
```

The checks/build use saved records. To refresh the four models from primary sources:

```bash
uv run --frozen python run.py moonshotai/Kimi-K3.1-Pro:moonshotai/Kimi-K3 LiquidAI/LFM3-1.7B:LiquidAI/LFM2.5-1.2B-Instruct
```

The live command contacts Hugging Face and reads source files and weight headers. It does not request weight payloads; ignored ranges and unsupported header lengths are rejected. Each analysis resolves current metadata, then pins all reads to that commit.

To share source without environments, caches, egg-info or private research:

```bash
uv run --frozen python tools/package_source.py
```

The output is `dist/archwatch-source.zip`.

## Automation and next work

The GitHub remote is configured. `.github/workflows/ci.yml` runs offline lint, formatting, tests and the site build on pull requests and pushes to main. [Foundations PR #1](https://github.com/LeoRigasaki/archwatch/pull/1) merged after [remote CI passed](https://github.com/LeoRigasaki/archwatch/actions/runs/43489941541); main CI also passed.

The existing `daily.yml` supports manual dispatch over the fixed watchlist. Its schedule remains commented out. Its PR label does not enable auto-merge. Discovery and controlled publication must be exercised manually before unattended runs.

The next patch is the watcher: the ten preferred labs plus a discovery route for promising outside models. The proposed cadence is daily metadata discovery and a bounded weekly Gemini generation batch. [Model selection](docs/MODEL_SELECTION.md), [provider constraints](docs/PROVIDER_LIMITS.md) and [next steps](docs/NEXT_STEPS.md) record the choices and remaining work. No Gemini integration or recurring task is enabled yet.

## Limits of the evidence

- Four live-checked records and two displayed successors are current coverage; the independent 30-model gold set and 95% target have not been measured.
- Confidence scores are prototype rules, not calibrated probabilities. Single-source values remain publishable and are labeled as such.
- Weight-kind pattern agreement cannot independently distinguish sliding from global attention when their tensor layouts are identical. Effective mechanisms need config/code/report evidence and broader family tests.
- Public renderers reject unverified patterns, missing required inputs and unsupported cache rules. Unknown families can remain withheld until supported.
- Cache formulas estimate cache state under stated dtype/kernel assumptions; they do not measure full runtime VRAM.
- Watcher/seen state, typed schema, LLM readers, report retrieval/page citations, issue review, tracing, MCP, own RSS, full collection UI and production deployment remain open.

[PROGRESS.md](PROGRESS.md) tracks acceptance. [The import manifest](docs/prototype-import.json) records the original archive. Private spot checks are neither production inputs nor the independent gold set.
