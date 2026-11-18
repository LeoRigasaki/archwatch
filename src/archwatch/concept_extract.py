"""Optional concept facts. Explicit configuration intent stays separate from execution proof.

These facts do not qualify new diagram topologies or memory formulas. Unnamed
residual multipliers, norm epsilon values and generic top-k settings are not
canonical mechanism evidence.
"""

from __future__ import annotations

import ast
import re

FIELDS = (
    "normalization_type",
    "normalization_modules",
    "per_layer_embeddings",
    "index_sharing",
    "kv_sharing",
    "compressed_attention_types",
    "residual_mechanism",
    "attention_budget",
    "attention_mechanism",
)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def config_facts(config):
    text, prefix = (config["text_config"], "text_config.") if "text_config" in config else (config, "")
    out = {}
    n = text.get("num_hidden_layers")
    norm = text.get("norm_type") or text.get("normalization_type")
    if isinstance(norm, str) and norm.lower().replace("_", "") in {"rmsnorm", "layernorm"}:
        out["normalization_type"] = (
            "RMSNorm" if norm.lower().replace("_", "") == "rmsnorm" else "LayerNorm",
            f"config:{prefix}norm_type/normalization_type (explicit name)",
        )
    # Dedicated PLE settings, not generic embedding or n-gram fields. IDs remain in source convention.
    width, ids = text.get("ple_embed_dim"), text.get("ple_layer_ids")
    if (
        type(n) is int
        and n > 0
        and _positive(width)
        and isinstance(ids, list)
        and ids
        and all(type(i) is int and 0 <= i < n for i in ids)
    ):
        out["per_layer_embeddings"] = (
            {"width": width, "source_layer_ids": ids},
            f"config:{prefix}ple_embed_dim/ple_layer_ids",
        )
    modes = text.get("indexer_types")
    if isinstance(modes, list) and len(modes) == n and set(modes) <= {"full", "shared"} and "shared" in modes:
        out["index_sharing"] = (
            {"modes": modes, "source_mapping": "not stated by this field"},
            f"config:{prefix}indexer_types (explicit shared modes)",
        )
    # A complete mapping is actionable; an anchor list of unknown scope is not expanded.
    for key, field in [("index_source_layer_ids", "index_sharing"), ("kv_source_layers", "kv_sharing")]:
        mapping = text.get(key)
        if (
            isinstance(mapping, list)
            and len(mapping) == n
            and all(type(i) is int and 0 <= i <= layer for layer, i in enumerate(mapping))
            and any(i != layer for layer, i in enumerate(mapping))
        ):
            out[field] = ({"source_layers": mapping}, f"config:{prefix}{key}")
    kinds = text.get("compressed_attention_types")
    if (
        isinstance(kinds, list)
        and len(kinds) == n
        and set(kinds) <= {"csa", "hca", "full"}
        and set(kinds) & {"csa", "hca"}
    ):
        out["compressed_attention_types"] = (kinds, f"config:{prefix}compressed_attention_types")
    residual = text.get("residual_type")
    if residual in {"mhc", "gated_residual"}:
        out["residual_mechanism"] = (residual, f"config:{prefix}residual_type (explicit mechanism)")
    # A generic index_topk is deliberately not an attention-budgeting fact.
    budget = text.get("num_attention_heads_per_layer") or text.get("attention_head_budget")
    if (
        isinstance(budget, list)
        and len(budget) == n
        and all(type(h) is int and h > 0 for h in budget)
        and len(set(budget)) > 1
    ):
        out["attention_budget"] = (
            {"query_heads_per_layer": budget},
            f"config:{prefix}num_attention_heads_per_layer/attention_head_budget (explicit allocation)",
        )
    mechanism = text.get("attention_mechanism")
    if mechanism == "cca" or text.get("cca") is True:
        out["attention_mechanism"] = ("cca", f"config:{prefix}attention_mechanism/cca (explicit mechanism)")
    return out


def _square_base(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "square" and not node.args:
            return ast.unparse(node.func.value)
        if (
            node.func.attr == "pow"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 2
        ):
            return ast.unparse(node.func.value)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and isinstance(node.right, ast.Constant)
        and node.right.value == 2
    ):
        return ast.unparse(node.left)
    return None


def _rms_formula(method):
    """Recognize a last-dimension squared mean followed by multiplication with its inverse RMS.

    Names are not proof. Reject centered norms and statistics that do not feed the
    inverse-RMS multiplication. This intentionally recognizes a narrow AST subset.
    """
    if any(isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)) for node in ast.walk(method)):
        return None
    statistics = {}
    for statement in ast.walk(method):
        if isinstance(statement, ast.BinOp) and isinstance(statement.op, ast.Sub):
            return None  # centered variance / LayerNorm is outside this RMS rule
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            continue
        value = statement.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "mean":
            axis = value.args[0] if value.args else next((k.value for k in value.keywords if k.arg == "dim"), None)
            if axis is not None and ast.unparse(axis) == "-1":
                base = _square_base(value.func.value)
                if base:
                    statistics[statement.targets[0].id] = (base, statement.lineno)
    for node in ast.walk(method):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        for base, inverse in [(node.left, node.right), (node.right, node.left)]:
            if not (
                isinstance(inverse, ast.Call)
                and ast.unparse(inverse.func).split(".")[-1] == "rsqrt"
                and len(inverse.args) == 1
            ):
                continue
            used = {n.id for n in ast.walk(inverse.args[0]) if isinstance(n, ast.Name)}
            valid_statistics = []
            for stat, (tensor, line) in statistics.items():
                overwritten = any(
                    isinstance(write, ast.Assign)
                    and line < write.lineno < node.lineno
                    and any(ast.unparse(target) in {stat, tensor} for target in write.targets)
                    for write in ast.walk(method)
                )
                if stat in used and ast.unparse(base) == tensor and not overwritten:
                    valid_statistics.append(stat)
            if valid_statistics:
                tainted = set()
                for statement in method.body:
                    if not isinstance(statement, (ast.Assign, ast.Return)):
                        continue
                    expression = statement.value
                    depends = node in list(ast.walk(expression)) or bool(
                        tainted & {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}
                    )
                    if isinstance(statement, ast.Return):
                        if depends:
                            return node.lineno
                        break
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            if depends:
                                tainted.add(target.id)
                            else:
                                tainted.discard(target.id)
    return None


def code_facts(files):
    """Qualify only locally defined RMS operators wired into used decoder normalization modules."""
    out = {}
    for filename, source in sorted(files.items()):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        implementations = {}
        for cls in (c for c in tree.body if isinstance(c, ast.ClassDef)):
            for method in (
                m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == "forward"
            ):
                line = _rms_formula(method)
                if line:
                    implementations[cls.name] = line
        modules = []
        for cls in (
            c for c in tree.body if isinstance(c, ast.ClassDef) and re.search(r"DecoderLayer|TransformerBlock", c.name)
        ):
            init = next((m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
            forward = next((m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "forward"), None)
            if not init or not forward:
                continue
            called = {ast.unparse(n.func) for n in ast.walk(forward) if isinstance(n, ast.Call)}
            # Only unconditional direct assignments; unresolved branches are not promoted.
            for stmt in init.body:
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1 or not isinstance(stmt.value, ast.Call):
                    continue
                target, implementation = ast.unparse(stmt.targets[0]), ast.unparse(stmt.value.func)
                writes = sum(
                    any(ast.unparse(t) == target for t in write.targets)
                    for write in ast.walk(init)
                    if isinstance(write, ast.Assign)
                )
                if (
                    writes == 1
                    and target in called
                    and implementation in implementations
                    and target.startswith("self.")
                    and "norm" in target.lower()
                ):
                    modules.append(
                        {
                            "kind": "RMSNorm",
                            "scope": "decoder",
                            "module": cls.name + "." + target[5:],
                            "implementation": implementation,
                            "formula_line": implementations[implementation],
                            "call_line": stmt.lineno,
                        }
                    )
        if modules:
            evidence = f"code:{filename}#L{min(m['formula_line'] for m in modules)}"
            out["normalization_type"] = ("RMSNorm", evidence)
            out["normalization_modules"] = (sorted(modules, key=lambda m: m["module"]), evidence)
    return out
