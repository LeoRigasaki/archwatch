# Concept guides

The site has 32 short guides to architecture mechanisms: attention variants (MHA, GQA, MLA, sliding window, sparse and linear attention), mixture of experts, Mamba-2, short convolution, normalisation, positional encoding, multi-token prediction, the KV cache and related topics. Each guide explains what the mechanism does, works one small example, shows the equation, says what it means for memory, and lists the published models that use it.

Guides teach. They never supply a model fact: the extractor, verifier, gold labels and memory calculator do not read them.

## How models are matched

Each guide in `config/concepts.yaml` has a `rule` over published record fields, for example `attention_type equals MHA` or "the layer pattern contains an MLA layer". `src/archwatch/concepts.py` applies it:

1. The record must first pass the normal publication gate. Held models and fine-tunes never match.
2. The rule runs on the published values only.
3. Every field the rule uses must have a current, passing verifier decision (`verified` or `single source`) whose value equals the published value. Otherwise the model is not listed.
4. The badge shows the weakest supporting field. A computed value keeps its inputs' evidence; it does not count as another source.

A missing field never means "absent": a guide with no examples says it has no qualified example yet, not that the models lack the mechanism. Each model page lists its matches with the supporting fields and pinned source links, and offers them as a JSON download.

Some rules need facts that the original extractor did not read: normalisation type (from the modelling code), KV sharing, index sharing and per-layer embeddings (from the config). `src/archwatch/concept_extract.py` reads them from the files pinned at each record's commit. `tools/enrich_concepts.py` adds them to existing records offline, through the normal verifier, without changing any existing fact. Five records carry such facts today.

## How the text is sourced

- `config/concepts.yaml` holds each guide's summary, mechanism, memory and "how to verify" text, its rule, and one primary source.
- `config/concept-lessons.yaml` holds the worked example, equation, notation, pitfalls and the small code illustration.
- Every primary source is a paper (mostly arXiv) or the model's own files: its Hugging Face repository, card or technical report. There are no blogs, galleries or third-party summaries, and a test enforces this.
- The text was written for ArchWatch. Worked-example numbers are toy values, labelled as such, and are not any checkpoint's facts.
- Illustrations are drawn by code (`src/archwatch/concept_diagrams.py`) from the lesson data, not by an image model.

## Review status

Each guide has `reviewed: false` until the maintainer has read it. The site marks unreviewed guides with a small "draft" label on the index and on the guide page. To approve a guide, set `reviewed: true` in `config/concepts.yaml`.
