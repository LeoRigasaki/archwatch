# Newest-release backfill and first-run protocol

The cohort is pinned in `config/newest-backfill.yaml` as of 14 November 2026. Official Hub chronology is grouped by release/architecture, with primary cards checked to exclude conversions and post-training adaptations. The selection is deliberately newer than several illustrative targets: Qwen3.8-Flash-Next, NVIDIA Dory and DiffusionGemma are included. Google's launch posts confirm [Gemma 4 on 2 April](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) preceded [DiffusionGemma on 10 June](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/). FrogNano's card identifies it as Qwen post-training; Mage-VL introduces a new primary visual architecture instead. Llama Guard is a safety adaptation, not a newer general foundation generation.

## Frozen evaluation

- Baseline code: merged PR #8, commit `5977d090797cad34f68645ae6eea47d3b9186716`.
- Seven unseen models, 139 source-transcribed labels, frozen in commit `a0a3f88` **before the pipeline run**. First-run records/results were committed in `46c1585`, before fixes.
- **First-run held-out exact match: 97/139 (69.8%).** The immutable report, per-field failures, source commits and gold hashes are in `results/newest-first-run/`.
- Liquid LFM3 Base and Kimi-K3.1-Pro were already development cases. They are processed but excluded from held-out accuracy.
- Meta access arrived during the run. Its 16-field primary-source label was entered afterward, before inspecting extracted values or changing code, and committed in `af7d814`. The separate late-access baseline is **11/16**. It is not retroactively added to the strictly pre-run-frozen cohort.
- After fixes, adaptation is **138/139** on the seven-model set and **16/16** on late-access Meta. These are adaptation/regression scores, not fresh held-out accuracy. The original labels and results are preserved; all labels remain `human_reviewed: false`.

## What changed

Counting now handles explicit native DeepSeek FP4 expert storage (two values per I8), Mistral FP8 `qscale_weight` metadata and non-parameter I64 lookup buffers. Learned BF16 router/layer scales remain parameters. Both fused expert tensor orientations are recognized. Vision, auxiliary classifier and MTP tensor namespaces cannot override main text dimensions or layer counts. One fused shared module does not contradict a config's explicit logical shared-expert count.

Extraction records QSA/Gated DeltaNet, indexed MLA, compressed attention, global/local KV dimensions, hyper-connections, n-gram layers, MTP and recurrent/diffusion settings. Llama4's alternating dense/MoE schedule and separate dense width are handled. Field diffs include those mechanisms rather than hiding a generation-type change behind matching dimensions. Family rules separate Dory, Mage, DiffusionGemma and the Mistral Large product stream.

## Processing and remaining review

`tools/backfill_latest.py --batch 0` and `--batch 1` process five primary selections each, with their source-pinned predecessors. Main facts are retained as diagnostic JSON even when a diagram is withheld. Only eligible records acquire public YAML/SVG artifacts. The batches are manual; no cron or LLM generation loop is added.

All ten newest selections now have analyzed records. Three selected records are diagram-eligible (Liquid Base, Kimi-Pro and Mage-VL). Seven newer graph/cache topologies remain explicitly held: compressed/shared attention, QSA/hyper-connections, indexed/shared MLA, normalized-GPT recurrence, block diffusion and Llama4 alternating/chunked layouts require qualified rendering/cache rules. Correct scalar extraction alone does not qualify those diagrams. The original Dory first run incorrectly passed the generic diagram gate; the explicit recurrent-layout hold prevents that misleading graph.

Two predecessor limitations remain visible. Llama-3.2-90B-Vision still returns HTTP 403. Mistral Large 3 Base exposes native safetensors but lacks root `config.json`; its field extraction needs a native-format adapter and is held under the existing admission rule. Its readable tensor headers still support a tensor-group comparison. These are source/access limitations, not successful field comparisons.

DiffusionGemma's exact released tensor total is 25,823,781,228, while its card states 25.2B. The latter appears consistent with a different counting scope, but that is not assumed as fact: `params_total` remains withheld by the verifier. The gold value is unchanged, and this remains the one failed adaptation field. No upstream defect is claimed.

The current gallery has 26 diagnostic model records, 12 eligible views and 28 tracked source repositories. Review files and source links remain inspectable; held architectures do not acquire fabricated diagrams. Repeat-batch byte/mtime evidence is in `results/newest-backfill-no-op.json`.
