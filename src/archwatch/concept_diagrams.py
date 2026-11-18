"""Original teaching figures. Toy geometry is never a checkpoint architecture claim."""

from __future__ import annotations

import html
import textwrap


def _text(x, y, text, size=14, anchor="middle", color="#1b302b"):
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{html.escape(str(text))}</text>'


def _box(x, y, w, h, label, fill="#fffef8", stroke="#cfd8c9"):
    lines = textwrap.wrap(label, 25)
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}"/>' + "".join(
        _text(x + w / 2, y + h / 2 + (i - (len(lines) - 1) / 2) * 20 + 5, line) for i, line in enumerate(lines)
    )


def render(key, lesson):
    out = []
    height = 220
    if key == "mha":
        height = 300
        out.append(_box(330, 12, 260, 44, "Token representations"))
        for i in range(4):
            x = 25 + 225 * i
            out.append(f'<path d="M460 56V78H{x + 100}V102" class="arrow"/>')
            out.append(_box(x, 102, 200, 72, f"Head {i + 1}: separate Q, K, V", "#e7f0e7" if i % 2 == 0 else "#f7e9d9"))
            out.append(f'<path d="M{x + 100} 174V200H460V224" class="arrow"/>')
        out.append(_box(300, 224, 320, 54, "Concatenate and project outputs"))
    elif key == "gqa":
        height = 360
        out.append(_text(120, 24, "Eight query heads", 16))
        out.append(_text(535, 24, "Two shared key/value groups", 16))
        for i in range(8):
            y = 42 + i * 37
            group = i // 4
            cy = 116 + group * 148
            out.append(_box(55, y, 130, 28, f"Query {i + 1}", "#e7f0e7" if group == 0 else "#f7e9d9"))
            out.append(
                f'<path d="M185 {y + 14}C280 {y + 14} 330 {cy} 420 {cy}" fill="none" stroke="{["#1f6552", "#c87540"][group]}" stroke-width="1.5"/>'
            )
        out.append(_box(420, 84, 230, 64, "KV group 1: queries 1–4", "#e7f0e7"))
        out.append(_box(420, 232, 230, 64, "KV group 2: queries 5–8", "#f7e9d9"))
        out.append(_text(785, 143, "Same 8 query heads", 14))
        out.append(_text(785, 175, "Fewer retained K/V heads", 14))
        out.append(_text(785, 207, "Group size = 4", 14))
    elif key == "sliding-window":
        height = 350
        for panel, label in enumerate(["Full causal visibility", "Window of 3 positions"]):
            ox = 75 + panel * 455
            out.append(_text(ox + 136, 26, label, 16))
            for row in range(8):
                for col in range(8):
                    active = col <= row and (panel == 0 or col >= row - 2)
                    out.append(
                        f'<rect x="{ox + col * 30}" y="{45 + row * 30}" width="27" height="27" rx="3" fill="{"#3b8063" if active else "#e6e6df"}"/>'
                    )
            out.append(_text(ox + 120, 309, "Keys →   (rows are queries)", 12))
        out.append(_text(460, 340, "Colored cells are visible token pairs; future positions remain masked.", 12))
    elif key == "rmsnorm":
        height = 250
        out.append(_text(460, 24, "Same direction, rescaled magnitude; no mean subtraction", 16))
        for i, (label, value, ox) in enumerate(
            [("x₁ = 3", 3, 50), ("x₂ = 4", 4, 50), ("y₁ ≈ 0.849", 0.849, 490), ("y₂ ≈ 1.131", 1.131, 490)]
        ):
            y = 66 + (i % 2) * 70
            out.append(_text(ox, y + 17, label, 13, "start"))
            out.append(
                f'<rect x="{ox + 115}" y="{y}" width="{value * 60}" height="25" rx="4" fill="{"#c87540" if i < 2 else "#1f6552"}"/>'
            )
        out.append(_text(460, 222, "RMS before ≈ 3.536   →   RMS after = 1 (with γ=1, ignoring ε)", 13))
    elif key == "rope":
        out.append('<circle cx="460" cy="110" r="72" fill="none" stroke="#ccd8cd"/>')
        out.append('<path d="M340 110H580M460 200V20" fill="none" stroke="#ccd8cd"/>')
        out.append('<path d="M460 110H532" stroke="#c87540" stroke-width="4"/>')
        out.append('<path d="M460 110V38" stroke="#1f6552" stroke-width="4"/>')
        out.append(_text(640, 115, "Original (1,0)", 15))
        out.append(_text(460, 16, "Rotated (0,1)", 15))
        out.append(_text(235, 105, "90° rotation", 16))
        out.append(_text(235, 135, "Vector length stays 1", 13))
    elif key == "moe":
        height = 300
        out.append(_box(20, 112, 140, 64, "Token → router"))
        for i in range(8):
            x = 230 + (i % 4) * 145
            y = 46 + (i // 4) * 148
            active = i in [2, 5]
            out.append(
                _box(
                    x,
                    y,
                    120,
                    54,
                    f"Expert {i + 1}",
                    "#e7f0e7" if active else "#eeeee8",
                    "#1f6552" if active else "#d8dbd2",
                )
            )
            if active:
                out.append(f'<path d="M160 144L{x} {y + 27}" class="arrow"/>')
        out.append(_text(460, 145, "Only selected experts process this token", 13))
        out.append(
            _text(460, 281, "All eight expert parameter sets still need storage or an explicit placement strategy.", 13)
        )
    else:
        labels = lesson["diagram"]
        n = len(labels)
        width = 840 / n
        out.append(_text(460, 28, "Illustrative mechanism — separate from model evidence", 14))
        for i, label in enumerate(labels):
            x = 40 + i * width
            out.append(_box(x, 77, width - 30, 90, label, "#e7f0e7" if i % 2 == 0 else "#fffef8"))
            if i < n - 1:
                out.append(f'<path d="M{x + width - 30} 122H{x + width - 7}" class="arrow"/>')
        out.append(_text(460, 199, "Follow the worked example below for the numerical assumptions.", 12))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 {height}" role="img" aria-label="Original {html.escape(key)} teaching illustration"><title>{html.escape(key)}: illustrative mechanism</title><defs><marker id="arrow-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#637b68"/></marker></defs><rect width="920" height="{height}" fill="#f9faf4"/><style>text{{font-family:system-ui,sans-serif}}.arrow{{fill:none;stroke:#637b68;stroke-width:1.5;marker-end:url(#arrow-{key})}}</style>'
        + "".join(out)
        + "</svg>"
    )
