"""Render Figure 1: serving-boundary attribution mechanism, 7-engine contract audit (I1-I5), and multi-scale impact."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

here = Path(__file__).resolve().parent

fig, ax = plt.subplots(figsize=(10.2, 3.65), dpi=300)
ax.set_xlim(0, 120)
ax.set_ylim(0, 52)
ax.axis("off")

FONT = "DejaVu Sans"


def box(x, y, w, h, title, lines, fc="#f8f9fa", ec="#444444", title_color="#111111", lw=1.0, body_fs=6.2, title_fs=7.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.22,rounding_size=0.9",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    )
    ax.add_patch(p)
    cy = y + h - 1.35
    ax.text(x + w / 2, cy, title, ha="center", va="top", fontsize=title_fs, fontweight="bold", color=title_color, family=FONT)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + (h - 3.35) / 2 + 0.12, body, ha="center", va="center", fontsize=body_fs, color="#222222", linespacing=1.28, family=FONT)


def arrow(x1, y1, x2, y2, color="#444444", style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=9.0, linewidth=1.05,
        color=color, connectionstyle=f"arc3,rad={rad}",
    ))


# Column headers
for cx, label in (
    (12.2, "1. Sampling Event"),
    (39.7, "2. Serializer Boundary"),
    (71.5, "3. 7-Engine Audit (I1–I5)"),
    (106.0, "4. Multi-Scale Impact"),
):
    ax.text(cx, 50.2, label, ha="center", va="center", fontsize=7.9, fontweight="bold", color="#1a1a1a", family=FONT)

# Column 1: Sampling Event (x=1.0..23.4, w=22.4)
box(
    1.0, 27.5, 22.4, 18.0,
    "Decoder Draw (t = 5)",
    [r'$u_t = 24313$ (" monumental")', r"raw $s_t = -4.4375$ nats", "text emits sampled word"],
    fc="#eef6f4", ec="#12675d", title_color="#12675d", body_fs=6.4, title_fs=7.2,
)
box(
    1.0, 4.0, 22.4, 18.0,
    r"Candidates $A_{t,5}$",
    [r'$A[0]$: $6277$ (" architecture")', r"score $= -2.2500$ nats", r"$24313$ omitted from $A_{t,5}$"],
    fc="#fdf4ef", ec="#994715", title_color="#994715", body_fs=6.4, title_fs=7.2,
)

# Column 2: Serializer Split (x=28.0..51.4, w=23.4)
box(
    28.0, 28.5, 23.4, 17.0,
    "Omit List or 15-Check Patch",
    ["reads (tokens, token_logprobs)", "emits SSE & non-stream rows", "384/384 live pass (135M–7B)"],
    fc="#eef6f4", ec="#12675d", title_color="#12675d", body_fs=6.3, title_fs=7.1,
)
box(
    28.0, 4.0, 23.4, 17.5,
    "Stock MLX-LM (k > 0)",
    ["dict(i[0], top_logprobs=i)", "reports (6277, -2.2500)", "+2.1875 nats bias; SSE drops"],
    fc="#fdf0ed", ec="#994715", title_color="#994715", lw=1.15, body_fs=6.3, title_fs=7.1,
)

# Column 3: Unified Contract Audit across 7 Engines (x=56.2..86.8, w=30.6)
rows = [
    ("MLX-LM 0.30.6–0.32.2", "ID/score swap; SSE omit $\\to$ I1, I2, I5", "#fdf0ed", "#994715"),
    ("SGLang 0.5.23 vs vLLM 0.34.0", "Empty-$\\delta$ drop (I3, I5) vs frag (I4)", "#fdf4ef", "#994715"),
    ("Ollama 0.40.17 (live)", "Coalesce/EOS (I3); k=5 drift (I2)", "#f5f6f8", "#555555"),
    ("llama.cpp & mistral.rs", "UTF-8 omit (I3) & U+FFFD bytes (I4)", "#f5f6f8", "#555555"),
    ("CTranslate2 4.9.0", "2V/0V batch leak (I3) & logits (I2)", "#f5f6f8", "#555555"),
]
for idx, (eng, desc, fc, ec) in enumerate(rows):
    yy = 39.2 - idx * 9.05
    box(56.2, yy, 30.6, 7.4, eng, [desc], fc=fc, ec=ec, title_color=ec, lw=0.95, body_fs=6.05, title_fs=6.8)

# Column 4: Downstream Impact (x=92.0..119.0, w=27.0)
cons = [
    ("3B & 7B Routing + Inspect", "+49.2 pp / +35.7 pp false-acc\nECE drops while safety fails", "#fdf0ed", "#994715"),
    ("DeepEval G-Eval (3B vs 7B)", "3B: 6/94 binary pass flips\n7B: 2/95 shifts, 0/95 flips", "#fdf4ef", "#994715"),
    ("7B GRPO & 128-Seed RL", "86.4% neg-adv grad zeroed\n-1.90 pp SST-2 acc (128 seeds)", "#eef6f4", "#12675d"),
]
for idx, (c_title, c_line, fc, ec) in enumerate(cons):
    yy = 34.4 - idx * 14.5
    box(92.0, yy, 27.0, 11.4, c_title, c_line.split("\n"), fc=fc, ec=ec, title_color=ec, lw=0.95, body_fs=6.15, title_fs=6.85)

# Arrows
arrow(23.7, 36.5, 27.7, 36.5, color="#12675d")
arrow(23.7, 32.0, 27.7, 15.5, color="#994715", rad=0.07)
arrow(23.7, 12.8, 27.7, 12.8, color="#994715")

arrow(51.7, 37.5, 55.9, 43.0, color="#12675d")
arrow(51.7, 13.5, 55.9, 41.5, color="#994715", rad=-0.08)

for cy in (40.1, 25.6, 11.2):
    arrow(87.1, cy, 91.7, cy, color="#555555")

fig.savefig(
    here / "mechanism.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
    metadata={
        "Author": None,
        "Creator": "Matplotlib",
        "Title": "Serving-boundary attribution mechanism, seven-engine contract audit, and multi-scale downstream impact",
    },
)
plt.close(fig)
