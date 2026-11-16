# ArchWatch

ArchWatch extracts architecture facts from a model's own Hugging Face config, tensor headers, Python code and model card, checks their agreement, and generates an architecture page with evidence. Conflicting fields are withheld from the record's `published` dictionary and from architecture views.

The user-supplied prototype is integrated under `src/archwatch/`. The deterministic core and bounded watcher now cover a primary-source backfill from all ten core publishers. Ten manually transcribed development labels and per-field evaluation are included; human review, held-out evaluation, LLM readers and the issue review queue remain open.

## Local demo

The site is built by `build_site.py` from the committed model records and watcher state (offline, no LLM calls) and deployed on Vercel (`vercel.json`); `site/` is a build output and not committed. It currently shows 12 eligible architectures across ten publishers, with generated SVGs, pinned source links, predecessor/tensor diffs and interactive cache calculators. Two historical derived records are held.

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
| Model card | HTML and Markdown pipe tables plus key/value bullets, with rounding tolerances |
| Calculator | GQA, sliding-window, short-convolution, MLA, DSA-index, KDA and Mamba-2 state estimates with stated assumptions |
| Verifier | Verified/single-source/conflict decisions; conflicts withheld; rendering inputs carry explicit publication decisions |
| Diff | Published-field and tensor-group diffs; watcher selects predecessors from inspected same-org/family history |
| Diagram/site | Published inputs only; verified weight-kind layer pattern required; site cache estimates and field comparisons recomputed |
| Watcher | Ten core publishers plus outside trending/recent discovery, pinned admission/analysis, duplicate architecture withholding, predecessor selection and restartable seen state |
| Tooling | uv lock, ruff, pre-commit and offline PR CI configured; source-only ZIP exporter |

The fixed list in `watchlist.txt` remains the reproducible smoke test. `watch.py` adds discovery and optional bounded processing; LLM code/report readers are not implemented. [Watcher operation](docs/WATCHER.md) documents admission, state, retries and limitations.

## Validation

[Backfill processing](results/backfill-processing.json) covers one primary release each from DeepSeek, Qwen, Mistral, Liquid AI, NVIDIA, Google, Meta, Microsoft, Moonshot and Z.ai. [The exact selections](config/backfill.yaml) and [gold labels](evals/README.md) retain pinned primary links. The gallery holds 26 saved diagnostic records, of which 12 meet its current display policy.

The ten-model development comparison improved from **117/144 to 144/144 published-field exact matches** after family fixes. [Per-field results](results/gold-exact-match.md) include missing fields in the denominator. Labels were manually transcribed by the assistant from configs, cards and raw weight headers before the fixes; **they are not human-reviewed or a held-out accuracy estimate**. The offline suite has **116 passing tests**.

The original four-model live run remains in [live-validation.json](results/live-validation.json). [Family rules](docs/FAMILY_RULES.md) document the new Phi parallel graph, Mistral all-layer sliding attention, GLM fused FFN and Nemotron-H operator-only graph. Cache estimates are not measured runtime VRAM.

The Kimi diff still identifies exactly **178,526,208 added DSA-indexer parameters**. The LFM2.5 config's nominal 12,288-wide FFN becomes **8,192** under its explicit auto-adjust convention. Its card/config context disagreement remains held and is not described as a lab defect. See [the LFM investigation](docs/LFM_FINDINGS.md).

The [newest-release backfill](docs/NEWEST_BACKFILL.md) is separate from that development set. Its strictly pre-run-frozen seven-model score is **97/139 (69.8%)**; late-access Meta scored **11/16** separately. After source-based fixes, adaptation is 138/139 plus 16/16 on Meta. Seven new topologies remain held for diagram/cache qualification, with primary facts retained. The original first-run results and gold hashes remain immutable.

## Run and develop

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --python 3.12
uv run --frozen pre-commit install
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q --junitxml=results/test-results.xml
uv run --frozen python build_site.py
uv run --frozen python tools/evaluate_gold.py
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

For a controlled backfill, pass explicit repositories from `config/backfill.yaml` to `watch.py --process`, at most five per run. `--reprocess` deliberately reanalyzes unchanged commits after rule/schema changes; ordinary runs leave unchanged artifacts untouched. Approved gated sources use the local Hugging Face login. Cross-host redirects strip its credential. Remote model code is never executed.

## Automation and next work

The GitHub remote is configured. `.github/workflows/ci.yml` runs offline lint, formatting, tests and the site build on pull requests and pushes to main. [Foundations PR #1](https://github.com/LeoRigasaki/archwatch/pull/1) merged after [remote CI passed](https://github.com/LeoRigasaki/archwatch/actions/runs/43489941541); main CI also passed.

The existing `daily.yml` supports manual `fixed` (default), `discover` and `process` modes. Its schedule remains commented out. Its PR label does not enable auto-merge. An [optional Discord summary](docs/DISCORD_NOTIFICATIONS.md) sends one text message after a changed watcher run; missing secrets and no-op runs are silent, and delivery failures cannot fail the run. Only the notification step receives its webhook secret. Discovery and controlled publication must be exercised manually before unattended runs.

The watcher uses the ten preferred labs plus outside discovery. A live temporary-root run processed the four known models with both expected predecessors; a second pass changed no artifact bytes or mtimes. A separate 80-candidate discovery scan did not publish new models. The proposed cadence remains daily metadata discovery and a bounded weekly Gemini batch, with cron disabled pending approval. [Model selection](docs/MODEL_SELECTION.md), [provider constraints](docs/PROVIDER_LIMITS.md) and [next steps](docs/NEXT_STEPS.md) record the choices and remaining work. No Gemini integration or recurring task is enabled yet.

## Outside comparison

[results/reference-agreement.md](results/reference-agreement.md) compares published fields with the fact sheets in
an external author's [reference gallery](https://github.com/owner/reference) (Apache-2.0),
fetched at a pinned commit by `tools/reference_agreement.py` and not vendored. At that commit 45 of 52 compared
values agree across 9 shared models; each disagreement is listed with ArchWatch's own evidence. The same file lists
gallery models ArchWatch has not processed. It is a comparison only: never a source for published fields or gold
labels, and its images are not downloaded or used. See [NOTICE](NOTICE) and [hard rules](docs/HARD_RULES.md).

## Limits of the evidence

- Ten primary releases form the current development gold set. Human review, the independent held-out 30-model evaluation and the 95% target remain open.
- Confidence scores are prototype rules, not calibrated probabilities. Single-source values remain publishable and are labeled as such.
- Weight-kind pattern agreement cannot independently distinguish sliding from global attention when their tensor layouts are identical. Effective mechanisms need config/code/report evidence and broader family tests.
- Public renderers reject unverified patterns, missing required inputs and unsupported cache rules. Unknown families can remain withheld until supported.
- Cache formulas estimate cache state under stated dtype/kernel assumptions; they do not measure full runtime VRAM.
- Typed schema, LLM readers, report retrieval/page citations, issue review, tracing, MCP, own RSS, full collection UI and production deployment remain open.

[PROGRESS.md](PROGRESS.md) tracks acceptance. [The import manifest](docs/prototype-import.json) records the original archive. Private spot checks are neither production inputs nor the independent gold set.
