"""LLM code reader, only for architectures the deterministic rules cannot read.

The model proposes claims; code decides. Every claim must cite numbered lines of the modelling code that we
fetched at a pinned revision, quote those lines exactly, name config.json keys that exist and agree, and name
tensor patterns that match our own safetensors headers with consistent shapes. A claim that fails any check
goes to review. Nothing here writes data/models or the site: results are written for a person to read first.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel, Field

from . import hub
from .gemini import Gemini, ProviderError

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "code-reader"
TRANSFORMERS = "https://raw.githubusercontent.com/huggingface/transformers/{tag}/src/transformers/models/{mt}/{name}"
TAGS = "https://api.github.com/repos/huggingface/transformers/tags?per_page=100&page={page}"
MAX_QUOTE_LINES = 40


# ---------------------------------------------------------------------------------------------- schema


class Claim(BaseModel):
    field: str = Field(description="ArchWatch field name from the list in the instructions, snake_case.")
    value: str = Field(description='The value as a JSON literal: 64, true, "silu", or a JSON list for per-layer types.')
    file: str = Field(description="Code file name exactly as given, e.g. modeling_motif.py.")
    line_start: int = Field(description="First line number of the cited code (the numbers printed before '|').")
    line_end: int = Field(description="Last line number of the cited code, inclusive.")
    quote: str = Field(description="The cited lines copied exactly, without the line numbers.")
    config_keys: list[str] = Field(description="config.json keys the code reads for this value (dotted for nested).")
    tensor_patterns: list[str] = Field(
        description="Tensor-name patterns from the tensor list that hold this part, with N for any index."
    )


class Claims(BaseModel):
    claims: list[Claim]


FIELDS = {
    "num_layers": "decoder layers (blocks) in the main stack",
    "hidden_size": "model width",
    "vocab_size": "vocabulary size (embedding rows)",
    "context_length": "maximum positions the model is configured for",
    "num_attention_heads": "query heads per attention layer",
    "num_kv_heads": "key-value heads per attention layer",
    "head_dim": "width of one query/key head",
    "v_head_dim": "width of one value head, when it differs",
    "attention_type": "MHA, GQA, MQA, MLA or another named mechanism",
    "layer_types": "JSON list with one mixer kind per decoder layer, in order",
    "mixer_type": "the sequence mixer when it is not attention (e.g. mLSTM)",
    "moe_num_experts": "routed experts per MoE layer",
    "moe_active_experts": "routed experts selected per token",
    "moe_shared_experts": "always-on shared experts per MoE layer",
    "moe_expert_intermediate": "inner width of one routed expert",
    "ffn_intermediate": "inner width of the dense feed-forward network",
    "dense_layers": "layers with a dense FFN instead of experts",
    "activation": "feed-forward activation function",
    "normalization_type": "decoder normalization (RMSNorm, LayerNorm, ...)",
    "mla_kv_lora_rank": "MLA compressed key-value width",
    "mla_q_lora_rank": "MLA compressed query width",
    "mla_qk_rope_head_dim": "MLA rotary key width",
    "sliding_window": "sliding-window size in tokens",
    "tied_embeddings": "input embedding and output head share weights",
    "num_mtp_layers": "multi-token prediction modules",
}

SYSTEM = """You read model source code and report architecture facts as claims.
Rules:
- Report only what the code and config establish. Every claim cites one file and a line range from the numbered
  code, and quotes exactly those lines (copy them without the line numbers).
- config_keys: the config.json keys the cited code reads to get the value. Use [] only for constants in the code.
- tensor_patterns: patterns copied from the tensor list (with N for indices) for the weights that implement it.
  Use [] only when no weight corresponds (e.g. an activation).
- value is the concrete value, looked up in config.json when the code reads a config attribute. Example: the code
  says `self.hidden_size = config.hidden_size` and config.json has "hidden_size": 4096, so value is 4096 and
  config_keys is ["hidden_size"]. Strings only for names, e.g. "silu".
- Report every listed field that the code and config establish; a decoder model usually establishes most of them.
  Do not guess; omit a field you cannot cite.
- Do not use external knowledge about the model; the files given are the only source."""


def prompt_text(files: dict[str, str], config: dict, tensors_summary: str, fields: dict[str, str]) -> str:
    field_list = "\n".join(f"- {name}: {meaning}" for name, meaning in fields.items())
    code = "\n\n".join(f"=== {name} ===\n{number(text)}" for name, text in sorted(files.items()))
    return (
        f"Fields to report when the code establishes them:\n{field_list}\n\n"
        f"config.json:\n{json.dumps(compact(config), indent=1, sort_keys=True)}\n\n"
        f"Tensors from the safetensors headers (pattern, count, dtype, shape):\n{tensors_summary}\n\n"
        f"Modelling code (line numbers added by us):\n{code}\n"
    )


def number(text: str) -> str:
    return "\n".join(f"{i:5d}| {line}" for i, line in enumerate(text.splitlines(), 1))


def compact(value, limit=400):
    """Config with very long lists/maps shortened, so per-tensor quantization maps cannot flood the prompt."""
    if isinstance(value, dict):
        if len(json.dumps(value)) > 20 * limit and len(value) > 50:
            return {"__omitted__": f"{len(value)} entries"}
        return {k: compact(v, limit) for k, v in value.items()}
    if isinstance(value, list) and len(json.dumps(value)) > 4 * limit and len(value) > 64:
        return {"__omitted__": f"list of {len(value)} items"}
    return value


INDEX = re.compile(r"\.(\d+)\.")


def pattern_of(name: str) -> str:
    return INDEX.sub(".N.", name)


def summarize_tensors(tensors: dict[str, dict], limit=600) -> str:
    groups: dict[str, list] = {}
    for name, t in sorted(tensors.items()):
        g = groups.setdefault(pattern_of(name), [0, t["dtype"], t["shape"]])
        g[0] += 1
    lines = [f"{p} | {n} | {dtype} | {shape}" for p, (n, dtype, shape) in sorted(groups.items())]
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more patterns omitted"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- code sources


def _fetch(url: str, attempts: int = 3) -> bytes | None:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "archwatch"})
            with urllib.request.urlopen(request, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))  # transient TLS/connection errors
    return None


def _transformers_file(tag: str, model_type: str, name: str) -> str | None:
    cached = CACHE / "transformers" / tag / model_type / name
    if cached.exists():
        return cached.read_text()
    data = _fetch(TRANSFORMERS.format(tag=tag, mt=model_type, name=name))
    if data is None:
        return None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data.decode("utf-8", errors="replace")


def _release_tags() -> list[tuple[int, ...]]:
    cached = CACHE / "transformers" / "_tags.json"
    if cached.exists():
        names = json.loads(cached.read_text())
    else:
        names = []
        for page in range(1, 6):
            batch = json.loads(_fetch(TAGS.format(page=page)) or b"[]")
            names += [t["name"] for t in batch]
            if len(batch) < 100:
                break
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(names))
    return sorted({tuple(int(x) for x in m.groups()) for n in names if (m := re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", n))})


def fetch_code(repo: str, info: dict, config: dict) -> dict:
    """Modelling code at a pinned revision: the repo's own files, else transformers at config.transformers_version."""
    own = sorted(f for f in info["files"] if re.fullmatch(r"(?:modeling|configuration)_[\w.]+\.py", f))
    if any(f.startswith("modeling_") for f in own):
        files = {f: hub.get_text(repo, f, revision=info["sha"]) for f in own}
        return dict(
            files={k: v for k, v in files.items() if v},
            origin={f: f"https://huggingface.co/{repo}/blob/{info['sha']}/{f}" for f in own},
            source="repo",
            flags=[],
            version_matches=True,
        )
    version = str(config.get("transformers_version") or "")
    model_type = str(config.get("model_type") or "")
    base = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not base or not model_type:
        return dict(
            files={}, origin={}, source=None, flags=["no modelling code and no transformers_version/model_type"]
        )
    flags = []
    dev = ".dev" in version
    if dev:
        flags.append(f"transformers_version {version} is a development build, not a release")
    name = f"modeling_{model_type}.py"
    wanted = tuple(int(x) for x in base.groups())
    tags = [wanted]
    if dev:  # a dev build can predate the first release that ships this model: try later releases, flagged
        tags += [t for t in _release_tags() if t > wanted]
    for tag_tuple in tags[:40]:
        tag = "v" + ".".join(map(str, tag_tuple))
        text = _transformers_file(tag, model_type, name)
        if text is None:
            continue
        matches = tag_tuple == wanted
        if not matches:
            flags.append(f"{name} is absent at v{'.'.join(map(str, wanted))}; read at {tag}, the first later release")
        return dict(
            files={name: text},
            origin={
                name: TRANSFORMERS.format(tag=tag, mt=model_type, name=name).replace(
                    "raw.githubusercontent.com/huggingface/transformers/", "github.com/huggingface/transformers/blob/"
                )
            },
            source=f"transformers {tag}",
            flags=flags,
            version_matches=matches and not dev,
        )
    return dict(files={}, origin={}, source=None, flags=flags + [f"{name} not found in transformers releases"])


# ---------------------------------------------------------------------------------------------- verification


def _norm(text: str) -> str:
    return " ".join(text.split())


def _config_value(config: dict, key: str):
    node = config
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return KeyError
    return node


def _resolve_key(config: dict, key: str):
    """A key as the model wrote it, or the same key inside text_config / llm_config."""
    value = _config_value(config, key)
    if value is KeyError:
        for nested in ("text_config", "llm_config"):
            value = _config_value(config, f"{nested}.{key}")
            if value is not KeyError:
                break
    return value


def _regex(pattern: str) -> re.Pattern:
    return re.compile("^" + re.escape(pattern).replace(r"\.N\.", r"\.\d+\.").replace(r"\.N$", r"\.\d+") + "$")


def _indices(names: list[str], pattern: str, marker: str) -> set[int]:
    """Distinct indices at the N that follows `marker` (e.g. 'layers', 'experts', 'blocks')."""
    out = set()
    for name in names:
        m = re.search(rf"(?:^|\.){marker}\.(\d+)\.", name)
        if m:
            out.add(int(m.group(1)))
    return out


LAYER_MARKERS = ("layers", "blocks", "h")
# fields the safetensors headers can confirm: a claim must name tensors for them, and the shapes must agree
SHAPE_FIELDS = {
    "num_layers",
    "hidden_size",
    "vocab_size",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "v_head_dim",
    "moe_num_experts",
    "moe_expert_intermediate",
    "ffn_intermediate",
    "mla_kv_lora_rank",
    "mla_q_lora_rank",
    "layer_types",
}
NUMERIC_FIELDS = {
    "num_layers",
    "hidden_size",
    "vocab_size",
    "context_length",
    "num_attention_heads",
    "num_kv_heads",
    "head_dim",
    "v_head_dim",
    "moe_num_experts",
    "moe_active_experts",
    "moe_shared_experts",
    "moe_expert_intermediate",
    "ffn_intermediate",
    "dense_layers",
    "mla_kv_lora_rank",
    "mla_q_lora_rank",
    "mla_qk_rope_head_dim",
    "sliding_window",
    "num_mtp_layers",
}
Q_PROJ = re.compile(r"(?:q_proj|wq|query|q_b_proj)\.weight$")
K_PROJ = re.compile(r"(?:k_proj|wk|key)\.weight$")


def _parse(value: str):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def check_claim(claim: Claim, files: dict[str, str], config: dict, tensors: dict[str, dict]) -> list[str]:
    """Problems with one claim, empty when every check passes."""
    problems = []
    text = files.get(claim.file)
    if text is None:
        return [f"cites {claim.file}, which is not among the fetched files"]
    if not claim.file.startswith("modeling_"):
        problems.append(f"cites {claim.file}, the configuration class (defaults), not the modelling code")
    lines = text.splitlines()
    if not (1 <= claim.line_start <= claim.line_end <= len(lines)):
        problems.append(f"line range {claim.line_start}-{claim.line_end} is outside {claim.file} (1-{len(lines)})")
    elif claim.line_end - claim.line_start + 1 > MAX_QUOTE_LINES:
        problems.append(f"cites {claim.line_end - claim.line_start + 1} lines; at most {MAX_QUOTE_LINES}")
    elif _norm(claim.quote) != _norm("\n".join(lines[claim.line_start - 1 : claim.line_end])):
        problems.append("quote does not equal the cited lines")
    value = _parse(claim.value)
    if claim.field in NUMERIC_FIELDS and (isinstance(value, bool) or not isinstance(value, int)):
        problems.append(f"{claim.field} needs a number, got {value!r}")
    if isinstance(value, str) and (value in claim.config_keys or _resolve_key(config, value) is not KeyError):
        problems.append(f"value {value!r} is a config key name, not its value")
    scalars = []
    for key in claim.config_keys:
        found = _resolve_key(config, key)
        if found is KeyError:
            problems.append(f"config key {key} does not exist")
        elif not isinstance(found, (dict, list)):
            scalars.append((key, found))
    # only same-kind values can contradict each other: "RMSNorm" is not refuted by rms_norm_eps = 1e-05
    scalars = [(k, v) for k, v in scalars if isinstance(v, str) == isinstance(value, str)]
    if not isinstance(value, (dict, list)) and scalars and value not in [v for _, v in scalars]:
        if len(scalars) == 1:
            problems.append(f"config {scalars[0][0]} = {scalars[0][1]!r}, claim says {value!r}")
        elif not claim.tensor_patterns:
            problems.append("value is derived from several config keys and no tensor confirms it")
    if claim.field in SHAPE_FIELDS and not claim.tensor_patterns:
        problems.append(f"{claim.field} is a weight shape, but the claim names no tensor to check it against")
    names = list(tensors)
    matched: dict[str, list[str]] = {}
    for pattern in claim.tensor_patterns:
        regex = _regex(pattern)
        hits = [n for n in names if regex.match(n)]
        if not hits:
            problems.append(f"tensor pattern {pattern} matches no tensor in the headers")
        matched[pattern] = hits
    hits = sorted({n for group in matched.values() for n in group})
    if hits and isinstance(value, (int, float)) and not isinstance(value, bool):
        problems += _shape_problems(claim.field, value, hits, tensors)
    if claim.field == "layer_types" and isinstance(value, list):
        problems += _pattern_problems(value, tensors)
    return problems


def _shape_problems(field: str, value, hits: list[str], tensors: dict[str, dict]) -> list[str]:
    if field == "num_layers":
        counts = [len(_indices(hits, "", m)) for m in LAYER_MARKERS]
        return [] if value in counts else [f"matched tensors span {max(counts)} layers, claim says {value}"]
    if field == "moe_num_experts":
        counts = {len(_indices(hits, "", "experts"))} | {
            tensors[n]["shape"][0] for n in hits if len(tensors[n]["shape"]) == 3
        }
        return [] if value in counts else [f"matched tensors hold {sorted(counts)} experts, claim says {value}"]
    dims = {d for n in hits for d in tensors[n]["shape"]}
    if field in ("num_attention_heads", "num_kv_heads", "head_dim", "moe_active_experts", "moe_shared_experts"):
        return []  # counts that are not a tensor dimension; checked across claims (heads x head_dim)
    return [] if value in dims else [f"no matched tensor has a dimension equal to {value} (dims {sorted(dims)[:12]})"]


def _pattern_problems(kinds: list, tensors: dict[str, dict]) -> list[str]:
    per_layer: dict[int, set] = {}
    for name in tensors:
        m = re.search(r"(?:^|\.)(?:layers|blocks)\.(\d+)\.(.+)$", name)
        if m and "mtp" not in name:
            per_layer.setdefault(int(m.group(1)), set()).add(m.group(2))
    if len(per_layer) != len(kinds):
        return [f"{len(kinds)} layer kinds claimed, the headers hold {len(per_layer)} layers"]
    signature = {}
    problems = []
    for index, kind in enumerate(kinds):
        sig = frozenset(per_layer.get(index, ()))
        for other, other_sig in signature.items():
            if other != kind and other_sig == sig:
                problems.append(f"layers typed {other!r} and {kind!r} hold identical tensors")
        if kind in signature and signature[kind] != sig:
            problems.append(f"layers typed {kind!r} hold different tensors (layer {index})")
        signature.setdefault(kind, sig)
    return sorted(set(problems))


def cross_checks(values: dict, tensors: dict[str, dict]) -> dict[str, list[str]]:
    """heads x head_dim must equal the query/key projection widths in the headers."""
    out: dict[str, list[str]] = {}
    hd = values.get("head_dim")
    for field, regex in (("num_attention_heads", Q_PROJ), ("num_kv_heads", K_PROJ)):
        heads = values.get(field)
        if not (isinstance(heads, int) and isinstance(hd, int)):
            continue
        widths = {tensors[n]["shape"][0] for n in tensors if regex.search(n) and len(tensors[n]["shape"]) == 2}
        if widths and heads * hd not in widths:
            out.setdefault(field, []).append(
                f"{field} {heads} x head_dim {hd} = {heads * hd}; projection rows {sorted(widths)}"
            )
            out.setdefault("head_dim", []).append(
                f"head_dim {hd} x {field} {heads} does not match projection rows {sorted(widths)}"
            )
    return out


def verdicts(claims: list[Claim], code: dict, config: dict, tensors: dict, rule_facts: dict) -> list[dict]:
    rows = []
    for claim in claims:
        problems = check_claim(claim, code["files"], config, tensors)
        if not code.get("version_matches", True):
            problems.append("code was not read at the config's own transformers version")
        rows.append(dict(claim.model_dump(), value=_parse(claim.value), problems=problems))
    values = {r["field"]: r["value"] for r in rows if not r["problems"]}
    for field, issues in cross_checks(values, tensors).items():
        for r in rows:
            if r["field"] == field:
                r["problems"] = r["problems"] + issues
    by_field: dict[str, set] = {}
    for r in rows:
        by_field.setdefault(r["field"], set()).add(json.dumps(r["value"], sort_keys=True))
    for r in rows:
        conflicts = []
        if len(by_field[r["field"]]) > 1:
            conflicts.append("another claim gives this field a different value")
        if r["field"] in rule_facts and json.dumps(rule_facts[r["field"]], sort_keys=True) != json.dumps(
            r["value"], sort_keys=True
        ):
            conflicts.append(f"rule-extracted value is {rule_facts[r['field']]!r}")
        r["conflicts"] = conflicts
        r["verdict"] = "verified" if not r["problems"] and not conflicts else "review"
    return rows


# ---------------------------------------------------------------------------------------------- calls


class CodeReader(Gemini):
    """Structured-output calls through the Interactions API; each call reserved in the one shared ledger."""

    def __init__(self, config, *, role, run_id, clock=time.time, wait=time.sleep, client=None):
        section = config["code_reader"]
        super().__init__(config, ROOT / section["ledger"], role=role, section="code_reader", clock=clock, wait=wait)
        self.run_id = run_id
        self.client = client
        self.calls = 0  # calls reserved by this reader, including ones that later failed

    def read(self, repo: str, commit: str, prompt: str) -> tuple[Claims, dict]:
        model = self.config["models"][self.role]["id"]
        digest = hashlib.sha256((SYSTEM + "\n" + prompt).encode()).hexdigest()
        cached = CACHE / "responses" / f"{commit}-{model}-{digest[:16]}.json"
        if cached.exists():  # same commit, same prompt, same model: no new call
            saved = json.loads(cached.read_text())
            return Claims.model_validate_json(saved["output_text"]), dict(saved["meta"], cached=True)
        self._reserve(len(SYSTEM.encode()) + len(prompt.encode()))
        self.calls += 1
        self._annotate(repo=repo, prompt_sha256=digest)
        client = self.client
        if client is None:
            from google import genai

            client = genai.Client(api_key=self.key)
        started = self.clock()
        try:
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                system_instruction=SYSTEM,
                response_format={"type": "text", "mime_type": "application/json", "schema": Claims.model_json_schema()},
                generation_config={
                    "temperature": self.config["temperature"],
                    "max_output_tokens": self.section["max_output_tokens"],
                },
                store=False,
            )
        except Exception as error:  # noqa: BLE001 - provider details (and keys) are never echoed
            raise ProviderError(f"Gemini interaction failed ({type(error).__name__}); no fallback attempted") from None
        text = interaction.output_text or ""
        claims = Claims.model_validate_json(text)
        meta = dict(model=model, prompt_sha256=digest, elapsed_seconds=round(self.clock() - started, 3), cached=False)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(output_text=text, meta=meta), indent=1))
        return claims, meta

    def _annotate(self, **fields):
        """Add who/what to the entry _reserve just wrote, so every call in the ledger names its purpose."""
        entries = json.loads(self.ledger.read_text())
        entries[-1].update(fields, run_id=self.run_id)
        self.ledger.write_text(json.dumps(entries, indent=2) + "\n")
