"""LLM code reader, only for architectures the deterministic rules cannot read.

The model proposes claims; code decides. A claim cites numbered lines of the modelling code fetched at a pinned
revision and quotes them exactly; its config.json keys must exist and agree. The model never names tensors: for a
size, it cites the line that builds the module (`self.q_proj = nn.Linear(...)`), and code maps that attribute to
its state-dict path (module tree from the code's own syntax tree) and checks the shapes in our safetensors
headers. A claim that fails any check goes to review. Nothing here writes data/models or the site.
"""

from __future__ import annotations

import ast
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
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
GITHUB_COMMITS = "https://api.github.com/repos/{repo}/commits?until={until}&per_page=1"
MAX_QUOTE_LINES = 40
# Frozen prompts: editing a prompt file means a new version. The hash makes an accidental edit fail loudly.
PROMPTS = {"code_reader_v1": "45bed7ae3932a5680d900c88b7b91a0434af27b0719d9621d4a8712bb5cab493"}
CONFIG_CLASS = re.compile(r"(?:^|/)configuration_[\w.]+\.py$")


# ---------------------------------------------------------------------------------------------- schema


class Claim(BaseModel):
    field: str = Field(description="Field name from the list in the instructions, snake_case.")
    value: str = Field(description='The value as a JSON literal: 64, true, "silu", or a JSON list for per-layer types.')
    file: str = Field(description="Code file name exactly as given in the === headers.")
    line_start: int = Field(description="First line number of the cited code (the numbers printed before '|').")
    line_end: int = Field(description="Last line number of the cited code, inclusive.")
    quote: str = Field(description="The cited lines copied exactly, without the line numbers.")
    config_keys: list[str] = Field(description="config.json keys the cited code reads (dotted for nested keys).")


class Claims(BaseModel):
    claims: list[Claim]


# ---------------------------------------------------------------------------------------------- prompt


def load_prompt(path: Path) -> dict:
    """A frozen prompt file: its version (the file name), system text, field list and task template."""
    text = path.read_text()
    version = path.stem
    digest = hashlib.sha256(text.encode()).hexdigest()
    if PROMPTS.get(version) != digest:
        raise ValueError(f"{path.name} does not match its frozen sha256; a changed prompt needs a new version")
    sections = dict(re.findall(r"^## (\w+)\n(.*?)(?=^## |\Z)", text, re.S | re.M))
    fields = dict(re.findall(r"^- (\w+): (.+)$", sections["Fields"], re.M))
    return dict(version=version, sha256=digest, system=sections["System"].strip(), fields=fields, task=sections["Task"])


def prompt_text(prompt: dict, files: dict[str, str], config: dict) -> str:
    code = "\n\n".join(f"=== {name} ===\n{number(text)}" for name, text in sorted(files.items()))
    return (
        prompt["task"]
        .replace("{fields}", "\n".join(f"- {k}: {v}" for k, v in prompt["fields"].items()))
        .replace("{config}", json.dumps(compact(config), indent=1, sort_keys=True))
        .replace("{code}", code)
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


def _cached_fetch(key: Path, url: str) -> str | None:
    cached = CACHE / key
    if cached.exists():
        return cached.read_text()
    data = _fetch(url)
    if data is None:
        return None
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data.decode("utf-8", errors="replace")


def _transformers_file(tag: str, model_type: str, name: str) -> str | None:
    return _cached_fetch(
        Path("transformers") / tag / model_type / name, TRANSFORMERS.format(tag=tag, mt=model_type, name=name)
    )


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


def _commit_before(repo: str, until: str) -> tuple[str, str] | None:
    """The last commit of a GitHub repo at or before a timestamp (cached)."""
    cached = CACHE / "github" / repo.replace("/", "__") / f"commit-until-{until[:19].replace(':', '')}.json"
    if not cached.exists():
        batch = json.loads(_fetch(GITHUB_COMMITS.format(repo=repo, until=until)) or b"[]")
        if not batch:
            return None
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps([batch[0]["sha"], batch[0]["commit"]["committer"]["date"]]))
    sha, date = json.loads(cached.read_text())
    return sha, date


def _github_code(repo: str, info: dict, source: dict) -> dict:
    """Code from the authors' GitHub repo at the last commit before the model's Hub release (configured per model)."""
    released = info.get("created_at")
    found = _commit_before(source["github"], released) if released else None
    if not found:
        return dict(files={}, origin={}, source=None, flags=[f"no {source['github']} commit before {released}"])
    sha, date = found
    files, origin = {}, {}
    for path in source["paths"]:
        text = _cached_fetch(
            Path("github") / source["github"].replace("/", "__") / sha / path,
            GITHUB_RAW.format(repo=source["github"], sha=sha, path=path),
        )
        if text is not None:
            files[Path(path).name] = text
            origin[Path(path).name] = f"https://github.com/{source['github']}/blob/{sha}/{path}"
    flags = [
        f"code from {source['github']} at {sha[:10]} ({date}), the last commit before the Hub release ({released}); "
        "not the model repository's own files"
    ]
    return dict(
        files=files, origin=origin, source=f"github {source['github']}@{sha[:10]}", flags=flags, version_matches=False
    )


def fetch_code(repo: str, info: dict, config: dict, sources: dict | None = None) -> dict:
    """Modelling code at a pinned revision.

    Order: a configured authors' repository (pinned to the release date; always capped at review), the repo's own
    modeling_*.py at the record commit, else transformers at config.transformers_version. Configuration classes are
    never fetched: their defaults are not the checkpoint's values.
    """
    if sources and repo in sources:
        return _github_code(repo, info, sources[repo])
    own = sorted(f for f in info["files"] if re.fullmatch(r"modeling_[\w.]+\.py", f))
    if own:
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
                name: f"https://github.com/huggingface/transformers/blob/{tag}/src/transformers/models/{model_type}/{name}"
            },
            source=f"transformers {tag}",
            flags=flags,
            version_matches=matches and not dev,
        )
    return dict(files={}, origin={}, source=None, flags=flags + [f"{name} not found in transformers releases"])


# ---------------------------------------------------------------------------------------------- module tree


def _callee(node) -> str | None:
    if isinstance(node, ast.Call):
        return _callee(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _children(node, dicts) -> tuple[set[str], bool]:
    """Classes a `self.x = ...` value builds, and whether they sit in an indexed list (ModuleList, Sequential)."""
    if isinstance(node, ast.IfExp):
        a, la = _children(node.body, dicts)
        b, lb = _children(node.orelse, dicts)
        return a | b, la or lb
    if not isinstance(node, ast.Call):
        return set(), False
    name = _callee(node)
    if name in ("ModuleList", "Sequential"):
        kids: set[str] = set()
        for arg in node.args:
            elements = (
                arg.elts
                if isinstance(arg, (ast.List, ast.Tuple))
                else [arg.elt]
                if isinstance(arg, (ast.ListComp, ast.GeneratorExp))
                else [arg]
            )
            for element in elements:
                kids |= _children(element, dicts)[0]
        return kids, True
    if isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name) and node.func.value.id in dicts:
        return set(dicts[node.func.value.id]), False  # e.g. MIXER_TYPES[block_type](config)
    return ({name} if name else set()), False


def _builds(node) -> bool:
    """A constructor call (nn.Linear, nn.Parameter, ModuleList, MIXERS[kind](...)), not getattr()/min()/a value."""
    if isinstance(node, ast.IfExp):
        return _builds(node.body) or _builds(node.orelse)
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Subscript):
        return True
    name = _callee(node)
    return bool(name) and name[0].isupper()


def module_tree(files: dict[str, str]) -> dict:
    """{class: {"bases": [...], "attrs": {attr: {"children", "list", "lines": [(file, line)]}}}} from the code itself."""
    trees = {}
    for fname, src in files.items():
        try:
            trees[fname] = ast.parse(src)
        except SyntaxError:
            continue
    dicts = {}
    for tree in trees.values():
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Dict):
                    dicts[node.targets[0].id] = [n for v in node.value.values if (n := _callee(v))]
    classes: dict[str, dict] = {}
    for fname, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            entry = classes.setdefault(node.name, {"bases": [_callee(b) for b in node.bases], "attrs": {}})
            for sub in ast.walk(node):
                if not isinstance(sub, (ast.Assign, ast.AnnAssign)) or sub.value is None:
                    continue
                for target in sub.targets if isinstance(sub, ast.Assign) else [sub.target]:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        and _builds(sub.value)  # builds a module/parameter, not a plain value
                    ):
                        kids, is_list = _children(sub.value, dicts)
                        attr = entry["attrs"].setdefault(target.attr, {"children": set(), "list": False, "lines": []})
                        attr["children"] |= kids
                        attr["list"] |= is_list
                        attr["lines"] += [(fname, n) for n in range(sub.lineno, (sub.end_lineno or sub.lineno) + 1)]
    return classes


def _attrs(classes, name, seen=()):
    """A class's attributes including those it inherits from classes defined in the same code."""
    if name not in classes or name in seen:
        return {}
    merged = {}
    for base in classes[name]["bases"]:
        merged.update(_attrs(classes, base, (*seen, name)))
    merged.update(classes[name]["attrs"])
    return merged


def state_paths(classes: dict, root: str | None) -> dict[tuple[str, int], set[str]]:
    """{(file, line): {state-dict module paths}} for every attribute assignment reachable from the root class."""
    children_of = {k for c in classes.values() for a in c["attrs"].values() for k in a["children"]}
    roots = [root] if root in classes else [c for c in classes if c not in children_of and classes[c]["attrs"]]
    out: dict[tuple[str, int], set[str]] = {}

    def walk(cls, prefix, seen):
        for attr, a in _attrs(classes, cls).items():
            path = prefix + [attr] + (["N"] if a["list"] else [])
            for line in a["lines"]:
                out.setdefault(line, set()).add(".".join(path))
            for kid in a["children"]:
                if kid in classes and kid not in seen and len(path) < 14:
                    walk(kid, path, seen | {kid})

    for r in roots:
        walk(r, [], {r})
    return out


def tensors_under(paths: set[str], tensors: dict[str, dict]) -> list[str]:
    """Header tensors that live under any of these module paths (a top-level prefix such as language_model. is allowed)."""
    regexes = [re.compile(r"(?:^|\.)" + re.escape(p).replace(r"\.N", r"\.\d+") + r"(?:\.|$)") for p in paths]
    return sorted(n for n in tensors if any(r.search(n) for r in regexes))


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


def _indices(names: list[str], marker: str) -> set[int]:
    """Distinct indices at the N that follows `marker` (e.g. 'layers', 'experts', 'blocks')."""
    return {int(m.group(1)) for name in names if (m := re.search(rf"(?:^|\.){marker}\.(\d+)\.", name))}


LAYER_MARKERS = ("layers", "blocks", "h")
# fields the safetensors headers can confirm: the cited line must build a module whose weights are checked
SHAPE_FIELDS = {
    "num_layers", "hidden_size", "vocab_size", "num_attention_heads", "num_kv_heads", "head_dim", "v_head_dim",
    "moe_num_experts", "moe_expert_intermediate", "ffn_intermediate", "mla_kv_lora_rank", "mla_q_lora_rank",
}  # fmt: skip
NUMERIC_FIELDS = SHAPE_FIELDS | {
    "context_length", "moe_active_experts", "moe_shared_experts", "dense_layers", "mla_qk_rope_head_dim",
    "sliding_window", "num_mtp_layers",
}  # fmt: skip
BOOL_FIELDS = {"tied_embeddings"}
# sizes that are usually plain values (self.head_dim = ...): checked by heads x head_dim = projection rows
CROSS_CHECKED = {"num_attention_heads", "num_kv_heads", "head_dim"}
Q_PROJ = re.compile(r"(?:q_proj|wq|query|q_b_proj)\.weight$")
K_PROJ = re.compile(r"(?:k_proj|wk|key)\.weight$")


def _parse(value: str):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def assess(claim: Claim, code: dict, config: dict, tensors: dict[str, dict], paths: dict) -> dict:
    """Three separate verdicts for one claim: citation, config agreement, weight check (or n/a)."""
    citation, agreement, weights = [], [], []
    files = code["files"]
    text = files.get(claim.file)
    value = _parse(claim.value)
    if CONFIG_CLASS.search(claim.file):
        citation.append(f"cites {claim.file}, a configuration class: its defaults are not this checkpoint's values")
    if text is None:
        citation.append(f"cites {claim.file}, which is not among the fetched modelling files")
    else:
        lines = text.splitlines()
        if not (1 <= claim.line_start <= claim.line_end <= len(lines)):
            citation.append(f"line range {claim.line_start}-{claim.line_end} is outside {claim.file} (1-{len(lines)})")
        elif claim.line_end - claim.line_start + 1 > MAX_QUOTE_LINES:
            citation.append(f"cites {claim.line_end - claim.line_start + 1} lines; at most {MAX_QUOTE_LINES}")
        elif _norm(claim.quote) != _norm("\n".join(lines[claim.line_start - 1 : claim.line_end])):
            citation.append("quote does not equal the cited lines")
    if claim.field in BOOL_FIELDS and not isinstance(value, bool):
        agreement.append(f"{claim.field} needs true or false, got {value!r}")
    if claim.field in NUMERIC_FIELDS and (isinstance(value, bool) or not isinstance(value, int)):
        agreement.append(f"{claim.field} needs a number, got {value!r}")
    if isinstance(value, str) and (value in claim.config_keys or _resolve_key(config, value) is not KeyError):
        agreement.append(f"value {value!r} is a config key name, not its value")
    scalars = []
    for key in claim.config_keys:
        found = _resolve_key(config, key)
        if found is KeyError:
            agreement.append(f"config key {key} does not exist")
        elif not isinstance(found, (dict, list)):
            scalars.append((key, found))
    # only same-kind values can contradict each other: "RMSNorm" is not refuted by rms_norm_eps = 1e-05
    scalars = [(k, v) for k, v in scalars if isinstance(v, str) == isinstance(value, str)]
    if not isinstance(value, (dict, list)) and len(scalars) == 1 and value != scalars[0][1]:
        agreement.append(f"config {scalars[0][0]} = {scalars[0][1]!r}, claim says {value!r}")
    mapped: set[str] = set()
    for line in range(claim.line_start, claim.line_end + 1):
        mapped |= paths.get((claim.file, line), set())
    hits = tensors_under(mapped, tensors) if mapped else []
    status = "n/a"
    if claim.field in SHAPE_FIELDS:
        if not mapped and claim.field in CROSS_CHECKED:
            status = "cross-check"  # decided in verdicts(): heads x head_dim against the projection rows
        elif not mapped:
            weights.append("the cited lines build no module that reaches the model's state dict")
        elif not hits:
            weights.append(f"cited module {sorted(mapped)[0]} maps to no tensor in the headers")
        elif isinstance(value, int) and not isinstance(value, bool):
            weights += _shape_problems(claim.field, value, hits, tensors)
        if status != "cross-check":
            status = "failed" if weights else "passed"
    return dict(
        citation=citation,
        agreement=agreement,
        weights=weights,
        weight_status=status,
        module_paths=sorted(mapped),
        tensors=len(hits),
    )


def _shape_problems(field: str, value, hits: list[str], tensors: dict[str, dict]) -> list[str]:
    if field == "num_layers":
        counts = [len(_indices(hits, m)) for m in LAYER_MARKERS]
        return [] if value in counts else [f"the cited module holds {max(counts)} layers, claim says {value}"]
    if field == "moe_num_experts":
        counts = {len(_indices(hits, "experts"))} | {
            tensors[n]["shape"][0] for n in hits if len(tensors[n]["shape"]) == 3
        }
        return [] if value in counts else [f"the cited module holds {sorted(counts)} experts, claim says {value}"]
    if field in ("num_attention_heads", "num_kv_heads", "head_dim"):
        return []  # not a single dimension: checked across claims (heads x head_dim = projection rows)
    dims = {d for n in hits for d in tensors[n]["shape"]}
    return (
        [] if value in dims else [f"no tensor of the cited module has a dimension {value} (dims {sorted(dims)[:12]})"]
    )


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
    classes = module_tree(code["files"])
    root = ((config.get("architectures") or [None])[0]) if isinstance(config.get("architectures"), list) else None
    paths = state_paths(classes, root)
    rows = []
    for claim in claims:
        a = assess(claim, code, config, tensors, paths)
        problems = a["citation"] + a["agreement"] + a["weights"]
        value = _parse(claim.value)
        if claim.field == "layer_types" and isinstance(value, list):
            problems += _pattern_problems(value, tensors)
        if not code.get("version_matches", True):
            problems.append("code was not read at the model's own revision (see flags); capped at review")
        rows.append(dict(claim.model_dump(), value=value, problems=problems, **a))
    values = {r["field"]: r["value"] for r in rows if not r["problems"]}
    for field, issues in cross_checks(values, tensors).items():
        for r in rows:
            if r["field"] == field:
                r["problems"] = r["problems"] + issues
                r["weights"] = r["weights"] + issues
                r["weight_status"] = "failed"
    checkable = isinstance(values.get("head_dim"), int) and any(
        isinstance(values.get(f), int) for f in ("num_attention_heads", "num_kv_heads")
    )
    for r in rows:
        if r["weight_status"] == "cross-check":
            if checkable and not r["weights"]:
                r["weight_status"] = "passed"
            else:
                issue = "no module cited and no heads x head_dim check against the projection rows was possible"
                r["weights"] = r["weights"] + ([] if r["weights"] else [issue])
                r["problems"] = r["problems"] + ([] if r["weights"] != [issue] else [issue])
                r["weight_status"] = "failed"
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


def server_error(error) -> bool:
    """A 5xx from the provider (the only failure worth one retry); 4xx such as 429 never is."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(code, int):
        return code >= 500
    return type(error).__name__ in ("ServerError", "InternalServerError", "ServiceUnavailableError")


class CodeReader(Gemini):
    """Structured-output calls through the Interactions API; each call reserved in the one shared ledger."""

    def __init__(self, config, *, role, run_id, clock=time.time, wait=time.sleep, client=None):
        section = config["code_reader"]
        super().__init__(config, ROOT / section["ledger"], role=role, section="code_reader", clock=clock, wait=wait)
        self.run_id = run_id
        self.client = client
        self.calls = 0  # calls reserved by this reader, including ones that later failed

    def read(self, repo: str, commit: str, prompt: dict, text: str, purpose: str = "read") -> tuple[Claims, dict]:
        model = self.config["models"][self.role]["id"]
        digest = hashlib.sha256((prompt["system"] + "\n" + text).encode()).hexdigest()
        cached = CACHE / "responses" / f"{commit}-{model}-{prompt['version']}-{digest[:16]}.json"
        if cached.exists():  # same commit, model, prompt version and prompt text: no new call
            saved = json.loads(cached.read_text())
            return Claims.model_validate_json(saved["output_text"]), dict(saved["meta"], cached=True)
        client = self.client or self._client()
        started = self.clock()
        for attempt in (1, 2):  # one retry, only after a server error, and it is reserved as its own call
            self._reserve(len(prompt["system"].encode()) + len(text.encode()))
            self.calls += 1
            self._annotate(
                repo=repo,
                prompt_version=prompt["version"],
                prompt_sha256=digest,
                purpose=purpose if attempt == 1 else f"{purpose}: retry after a server error",
            )
            try:
                interaction = client.interactions.create(
                    model=model,
                    input=text,
                    system_instruction=prompt["system"],
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": Claims.model_json_schema(),
                    },
                    generation_config={
                        "temperature": self.config["temperature"],
                        "max_output_tokens": self.section["max_output_tokens"],
                    },
                    store=False,
                )
                break
            except Exception as error:  # noqa: BLE001 - the key is stripped from anything echoed
                detail = str(error).replace(self.key, "<key>")[:160]
                if attempt == 1 and server_error(error):
                    continue
                raise ProviderError(f"Gemini interaction failed ({type(error).__name__}: {detail})") from None
        output = interaction.output_text or ""
        claims = Claims.model_validate_json(output)
        meta = dict(
            model=model,
            prompt_version=prompt["version"],
            prompt_sha256=digest,
            elapsed_seconds=round(self.clock() - started, 3),
            cached=False,
        )
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(output_text=output, meta=meta), indent=1))
        return claims, meta

    def _client(self):
        """One request per reserved call: the SDK's automatic retries are off, because every retry is a real
        request against the free daily limit that the ledger would never see. read() makes its own single retry
        after a server error, reserved like any call. A stalled call times out."""
        from google import genai
        from google.genai import types

        return genai.Client(
            api_key=self.key,
            http_options=types.HttpOptions(
                timeout=int(self.config["request_timeout_seconds"] * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    def _annotate(self, **fields):
        """Add who/what to the entry _reserve just wrote, so every call in the ledger names its purpose."""
        entries = json.loads(self.ledger.read_text())
        entries[-1].update(fields, run_id=self.run_id)
        self.ledger.write_text(json.dumps(entries, indent=2) + "\n")
