"""Static scan of a repo's own Python files (modeling_*.py, configuration_*.py) with `ast`.

It never imports or runs the code (remote code is untrusted). It records classes, what each
attention or mixer class builds in __init__, and a few patterns that settle questions the config
leaves open, each with a file:line citation.
"""

from __future__ import annotations

import ast
import re


def _src_line(src: str, lineno: int) -> str:
    return src.splitlines()[lineno - 1].strip()


def scan(files: dict[str, str]) -> dict:
    out: dict = {"files": sorted(files), "classes": [], "facts": {}}
    facts = out["facts"]
    for fname, src in files.items():
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            out.setdefault("errors", []).append(f"{fname}: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc = (ast.get_docstring(node) or "").split("\n")[0][:160]
                out["classes"].append({"name": node.name, "file": fname, "line": node.lineno, "doc": doc})
                is_mixer = re.search(r"Attention|Conv|Mixer|DeltaNet|KDA", node.name)
                if not is_mixer:
                    continue
                built = []
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Assign)
                        and len(sub.targets) == 1
                        and isinstance(sub.targets[0], ast.Attribute)
                        and isinstance(sub.value, ast.Call)
                    ):
                        fn = ast.unparse(sub.value.func)
                        if re.search(r"Linear|Conv1d|Norm", fn):
                            built.append(f"{sub.targets[0].attr}={fn.split('.')[-1]}")
                            if sub.targets[0].attr in ("q_norm", "k_norm"):
                                facts["qk_norm"] = (True, f"code:{fname}#L{sub.lineno}")
                            if sub.targets[0].attr in ("gate_proj", "g_proj") and "Attention" in node.name:
                                facts["attention_output_gate"] = (True, f"code:{fname}#L{sub.lineno}")
                out["classes"][-1]["builds"] = built
            # (layer_idx + 1) in <list>  -> per-layer lists in the config are 1-based
            if isinstance(node, ast.Compare) and any(isinstance(o, ast.In) for o in node.ops):
                left = ast.unparse(node.left)
                if re.fullmatch(r"\(?layer_idx \+ 1\)?", left):
                    facts["layer_index_base_in_config"] = (
                        1,
                        f"code:{fname}#L{node.lineno}: {_src_line(src, node.lineno)}",
                    )
        if re.search(r"sliding_window", src) and "sliding_attention" in src:
            m = re.search(r"sliding_window\s*=.*sliding_attention.*", src)
            if m:
                ln = src[: m.start()].count("\n") + 1
                facts["sliding_window_applies_to"] = ("sliding_attention layers only", f"code:{fname}#L{ln}")
    has_modeling = any(f.startswith("modeling_") for f in files)
    facts["modeling_code_in_repo"] = (has_modeling, "repo file list")
    return out
