# ArchWatch

## Problem

ArchWatch extracts architecture facts from a model's own Hugging Face config, tensor headers, Python code and model card, cross-checks those claims, and generates an architecture page with evidence. Model cards are claims to check; conflicting fields are held out of the record's `published` dictionary.

The implementation baseline is the user-supplied `archwatch.zip`, now integrated under `src/archwatch/`. The deterministic core works on four saved model records. The watcher, LLM readers and broader evaluation/automation remain open; this is approximately the Day 9 prototype stage with earlier gaps and several later diff/site features already implemented.

## Live demo

The self-contained local demo is [site/index.html](site/index.html), built from the two successor records. It includes generated SVGs, source evidence, predecessor diffs, tensor-group changes and an interactive cache calculator.

```bash
python -m http.server 4173 --directory site
```

The earlier third-party-gallery reconstruction is private research material and is not the ArchWatch demo. Its hosted address is omitted from this README and was not found in the portfolio projects checked. The owner-only access and anonymous HTTP 401 checks are recorded in [the access review](docs/deployment-access-review.json).

## Architecture

| Stage | Implementation | Current capability |
|---|---|---|
| Hub access | `src/archwatch/hub.py` | Repository metadata, commit-specific file/cache reads, bounded safetensors-header range requests |
| Config | `extract.py` | Nested text configs, dense/hybrid/MoE, latent MoE, MLA/DSA/KDA, explicit layer lists and one-based indexing |
| Weight counting | `weights.py` | Tensor-shape parameter totals, supported packed-weight handling, component breakdowns, active-parameter conventions |
| Code analysis | `codescan.py` | Static AST classes/patterns with file and line citations; remote code is not executed |
| Model card | `card.py` | Structured table claims with rounding tolerances |
| Calculator | `calc.py` | GQA, sliding-window, short-convolution, MLA, DSA-index and KDA cache estimates with assumptions |
| Verifier | `verify.py` | `verified`, `single source`, `conflict`; confidence values and record-level publication withholding |
| Diff | `pipeline.py` | Field and tensor-group diffs against an explicitly supplied predecessor |
| Diagram/site | `diagram.py`, `build_site.py` | Record-generated SVG layer/block diagrams and a self-contained two-model page |

The input list is fixed in `watchlist.txt`; a new-model watcher is not implemented. The LLM code reader and report agent are also not implemented.

## Results

[Generated results](results/README.md) and [machine-readable validation](results/prototype-validation.json) summarize the four imported records:

- `moonshotai/Kimi-K3` and `moonshotai/Kimi-K3.1-Pro`;
- `LiquidAI/LFM2.5-1.2B-Instruct` and `LiquidAI/LFM3-1.7B`.

The saved records have **60 verified field checks, 20 single-source checks and one conflict**. The offline suite currently has **22 passing tests**. These are prototype checks, not accuracy against the planned independent 30-model gold set.

The recorded Kimi predecessor diff identifies **178,526,208 added DSA-indexer parameters**. The LFM2.5-Instruct record holds an FFN-width conflict for review: config `intermediate_size=12,288` versus weight-shape evidence `8,192`. That field is excluded from its published dictionary.

The separate LFM2.5 model-card/config context inconsistency is retained in [the evidence note](docs/LFM_FINDINGS.md). It concerns different explicitly named variants and should not be conflated with the FFN-width conflict.

## How to run

Python 3.12 or newer. Minimal installable src-layout packaging is included; dependency locking, ruff, pre-commit and dedicated PR CI remain to be configured.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q --junitxml=results/test-results.xml
.venv/bin/python build_site.py
.venv/bin/python tools/summarize_prototype.py
```

The test/build commands above use saved local records. To run fresh primary-source analysis explicitly:

```bash
.venv/bin/python run.py moonshotai/Kimi-K3.1-Pro:moonshotai/Kimi-K3 LiquidAI/LFM3-1.7B:LiquidAI/LFM2.5-1.2B-Instruct
```

That command contacts Hugging Face and reads source files and weight headers. It does not request weight payloads; unsupported or ignored range responses are rejected. Hub metadata is cached locally, and cache refresh/discovery policy belongs to the future watcher.

`.github/workflows/daily.yml` is written but has not run. It analyzes the fixed watchlist, builds the site, runs tests and opens a labeled PR once connected to a GitHub repository. An `auto-merge` label alone does not enable automatic merging.

The initial GitHub baseline enables manual dispatch only. The daily cron remains commented out until the remaining extraction, rendering and tooling fixes land.

## Limitations

- Four saved records and two displayed successor models are the current coverage; the formal gold set and 95% config-field target have not been measured.
- Today's validation was offline. The imported tensor counts/source runs were not fetched again from the Hub. Revision-specific fetching was corrected for future runs; that change does not retroactively re-establish the original archive's source snapshots.
- Confidence scores are prototype rules, not calibrated probabilities. Single-source values are allowed by the current threshold.
- The record verifier withholds conflicts, but the general site/SVG renderers still merge raw facts for some display/calculation inputs. A broader release needs strict consumer-side gating and handling for incomplete/conflicted target records; the two displayed imported targets currently have no conflicts.
- Cache formulas are estimates under stated dtype/state/kernel assumptions; they are not measurements of full runtime VRAM. Unknown layer types and additional model families need further validation.
- Predecessors are supplied manually. Watcher/seen state, LLM readers, report retrieval/page citations, GitHub review issues, Langfuse, MCP tools, own RSS and production deployment remain open.

See [PROGRESS.md](PROGRESS.md) for the corrected component and milestone checklist. The [import manifest](docs/prototype-import.json) records the archive hash and imported paths. Private spot-check research is excluded from version control and is not the gold set or production input.
