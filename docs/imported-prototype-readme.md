# ArchWatch (prototype)

Auto-generated, verified architecture pages for open-weight LLMs on Hugging Face.
Every number comes from the model's own files and is cross-checked across sources;
the model card is treated as a claim to check.

Tested on `moonshotai/Kimi-K3.1-Pro` (2.78T) and `LiquidAI/LFM3-1.7B` (1.73B), each
diffed against its predecessor. No weights are downloaded.

## Run

```bash
uv venv && uv pip install -r requirements.txt   # or: pip install -r requirements.txt
python run.py moonshotai/Kimi-K3.1-Pro:moonshotai/Kimi-K3 LiquidAI/LFM3-1.7B:LiquidAI/LFM2.5-1.2B-Instruct
python build_site.py      # -> site/index.html
python -m pytest -q
```

`repo:predecessor` pairs; the predecessor part is optional. Everything is cached in `cache/`.

## Pipeline

| Stage | Module | What it does | LLM? |
| --- | --- | --- | --- |
| Fetch | `hub.py` | Repo info pinned to a commit; config, code and card files; safetensors **headers** via HTTP range requests (8 bytes + JSON per shard) | no |
| Config | `extract.py` | `config.json` to normalized facts, each with the key it came from. Handles `text_config`, `layer_types`, 1-based `linear_attn_config`, MLA, DSA, MoE, latent MoE | no |
| Weights | `weights.py` | Exact parameter counts from tensor shapes (4-bit packed = 2 params/byte, scales skipped), component breakdown, per-layer kind from tensor names, shape facts | no |
| Code | `codescan.py` | `ast` scan of the repo's own `modeling_*.py` / `configuration_*.py` (never imported or run), facts with `file#line` citations | no |
| Card | `card.py` | Card summary table to claims, with the tolerance its rounding implies ("2.8T" = 2.75 to 2.85T) | no |
| Calculator | `calc.py` | Cache bytes per token and fixed cache per layer type (GQA, sliding, short conv, MLA, DSA indexer, KDA state); every assumption listed | no |
| Verify | `verify.py` | Each field from every source: verified / single source / conflict. Conflicts go to review, not the page | no |
| Diff | `pipeline.py` | Field diff and tensor-group diff against the predecessor | no |
| Diagram | `diagram.py` | Layer map + decoder block SVG from verified fields | no |
| Publish | `build_site.py` | One self-contained static page | no |

The planned LLM stages (reading modeling code that `ast` can't interpret, reading tech
reports) slot in as extra sources for the verifier; they never publish on their own.

## Automation

`.github/workflows/daily.yml` runs the watcher daily, rebuilds records and opens a PR.
A PR with any conflict is labelled `needs-review` and is not auto-merged.
