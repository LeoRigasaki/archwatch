# Primary-source development labels

`gold/` contains one manually transcribed YAML label for each of the ten core publishers in `config/backfill.yaml`. Each label pins its config, model card, weight files and representative header shapes to the source commit. Labels were committed before the family fixes, in commit `0fa673f`; they were not generated from `data/models`, extractor results or calculator output.

The assistant inspected the primary configs/cards and raw safetensors header shapes, then entered expected values explicitly. Parameter totals use an independent shape sum, excluding scales and unpacking the supported four-bit tensors. This is assistant manual curation, **not owner/human review**: every label has `human_reviewed: false`. A reviewer should confirm each expectation and the scope notes against those links before accepting it as a trusted benchmark.

The shared GLM family card discusses larger models; its 9B dimensions come from that checkpoint's config and released shapes. Some cards omit exact dimensions, so missing claims were not invented. Gated sources were read only where the local account already had access; the code does not accept licenses.

```bash
uv run --frozen python tools/evaluate_gold.py
```

The evaluator requires the exact labeled revision and scores `published` fields only. Missing/withheld fields count as failures. Results are per field and per model in `results/gold-exact-match.{json,md}`. `results/backfill-before-fixes.*` preserves the initial 117/144 comparison; missing held models count as missing, not excluded from the denominator. The current report is 144/144 after rule development.

This is a **development set used to fix the rules**. Its score does not establish generalization, calibrated verifier confidence, diagram semantic accuracy, cache/runtime correctness or the project's 95% held-out target. A separate unseen model set and human review are still needed. Cache estimates and active-parameter conventions are outside these labels.
