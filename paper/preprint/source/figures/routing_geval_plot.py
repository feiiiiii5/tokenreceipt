"""Render multi-scale Qwen2.5-3B vs 7B ARC-Challenge routing false-accepts and DeepEval GEval scale contrast."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

here = Path(__file__).resolve().parent
d3b = here.parents[2] / "reproducibility" / "downstream_3b"
d24 = here.parents[2] / "reproducibility" / "downstream_2026_11_24"
cache_file = here / "routing-geval.json"


def build_routing_rows(routing: dict, policies: list[tuple[str, str]]) -> list[dict]:
    out = []
    for key, label in policies:
        r = routing[key]
        out.append({
            "key": key,
            "label": label,
            "native_far_pct": round(100.0 * r["native_false_accept_rate"], 4),
            "buggy_far_pct": round(100.0 * r["buggy_false_accept_rate"], 4),
            "native_fa": r["native_false_accepts"],
            "buggy_fa": r["buggy_false_accepts"],
            "wrong_total": r["wrong_total"],
            "mcnemar_p": r["false_accept_mcnemar_p"],
        })
    return out


def load_geval_points(directory: Path, prefix: str) -> list[dict]:
    pts = []
    for panel_name, fname in (("A", f"{prefix}panel_a.jsonl"), ("B", f"{prefix}panel_b.jsonl")):
        for line in (directory / fname).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["parsed_both"]:
                pts.append({
                    "panel": panel_name,
                    "score_orig": row["score_orig"],
                    "score_corr": row["score_corr"],
                    "passed_orig": row["passed_orig"],
                    "passed_corr": row["passed_corr"],
                })
    return pts


if (
    d3b.is_dir()
    and (d3b / "arc_routing_expected.json").exists()
    and d24.is_dir()
    and (d24 / "arc_routing_7b_expected.json").exists()
):
    arc3 = json.loads((d3b / "arc_routing_expected.json").read_text(encoding="utf-8"))
    arc7 = json.loads((d24 / "arc_routing_7b_expected.json").read_text(encoding="utf-8"))
    rows_3b = build_routing_rows(
        arc3["analysis_summary"]["confirmation_metrics"]["routing"],
        [
            ("pilot_tau_youden", "$\\tau_J$\n0.625"),
            ("pilot_tau_85", "$\\tau_{85}$\n0.666"),
            ("fixed_0.70", "Fixed\n0.70"),
            ("fixed_0.80", "Fixed\n0.80"),
            ("fixed_0.90", "Fixed\n0.90"),
        ],
    )
    rows_7b = build_routing_rows(
        arc7["analysis_summary"]["confirmation_metrics"]["routing"],
        [
            ("pilot_tau_85", "$\\tau_{85}$\n0.526"),
            ("pilot_tau_90", "$\\tau_{90}/\\tau_J$\n0.682"),
            ("fixed_0.70", "Fixed\n0.70"),
            ("fixed_0.80", "Fixed\n0.80"),
            ("fixed_0.90", "Fixed\n0.90"),
        ],
    )
    payload = {
        "routing_rows": rows_3b,
        "routing_rows_7b": rows_7b,
        "ece_3b": arc3["analysis_summary"]["confirmation_metrics"]["calibration_all"]["ece_10bin"],
        "ece_7b": arc7["analysis_summary"]["confirmation_metrics"]["calibration_all"]["ece_10bin"],
        "geval_points": load_geval_points(d3b, "geval_"),
        "geval_points_7b": load_geval_points(d24, "geval_7b_"),
    }
    cache_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
else:
    payload = json.loads(cache_file.read_text(encoding="utf-8"))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.5, 2.9), dpi=300, layout="constrained")


def plot_routing(ax, rows: list[dict], title: str, ece: dict, ylabel: bool) -> None:
    x = np.arange(len(rows))
    width = 0.36
    nat_vals = [r["native_far_pct"] for r in rows]
    bug_vals = [r["buggy_far_pct"] for r in rows]
    ax.bar(x - width / 2, nat_vals, width, label="Native scores", color="#12675d", edgecolor="#0d4a43", linewidth=0.55)
    ax.bar(x + width / 2, bug_vals, width, label="Buggy wire scores", color="#994715", edgecolor="#6e320e", linewidth=0.55)
    for idx, row in enumerate(rows):
        top = max(nat_vals[idx], bug_vals[idx])
        p = row["mcnemar_p"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else "ns")
        ax.text(idx, top + 2.0, sig, ha="center", va="bottom", fontsize=7.6, color="#222222", fontweight="bold" if sig != "ns" else "normal")
    ax.set_xticks(x, [r["label"] for r in rows], fontsize=7.2)
    ax.set_xlabel("Acceptance threshold $\\tau$", fontsize=8.3)
    if ylabel:
        ax.set_ylabel("Wrong-answer false-accept (%)", fontsize=8.3)
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=7.7)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=8.8, pad=5)
    ax.text(
        0.98, 0.96,
        f"ECE: {ece['native']:.3f} -> {ece['buggy']:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.9, color="#333333",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cccccc", "alpha": 0.94, "lw": 0.6},
    )


plot_routing(ax1, payload["routing_rows"], "ARC Routing (Qwen2.5-3B, N=59 wrong)", payload["ece_3b"], ylabel=True)
ax1.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#dddddd", fontsize=6.7)
plot_routing(ax2, payload["routing_rows_7b"], "ARC Routing (Qwen2.5-7B, N=42 wrong)", payload["ece_7b"], ylabel=False)

# Right panel: Multi-scale DeepEval GEval score comparison (3B N=94 vs 7B N=95)
ax3.add_patch(Rectangle((-0.04, 0.5), 0.54, 0.55, facecolor="#fdf0ed", edgecolor="none", zorder=0))
ax3.plot([-0.03, 1.05], [-0.03, 1.05], color="#888888", linestyle=":", linewidth=0.95, zorder=1)
ax3.axvline(0.5, color="#555555", linestyle="--", linewidth=0.8, zorder=1)
ax3.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8, zorder=1)

pts3 = payload["geval_points"]
pts7 = payload["geval_points_7b"]
unchanged = [p for p in pts3 if p["score_orig"] == p["score_corr"]]
shifted3 = [p for p in pts3 if p["score_orig"] != p["score_corr"] and p["passed_orig"] == p["passed_corr"]]
flipped3 = [p for p in pts3 if p["passed_orig"] != p["passed_corr"]]
shifted7 = [p for p in pts7 if p["score_orig"] != p["score_corr"]]

ax3.scatter([p["score_corr"] for p in unchanged], [p["score_orig"] for p in unchanged], s=16, color="#888888", alpha=0.45, edgecolors="none", zorder=2, label=f"3B/7B unchanged")
ax3.scatter([p["score_corr"] for p in shifted3], [p["score_orig"] for p in shifted3], s=32, color="#12675d", edgecolors="white", linewidth=0.5, zorder=4, label=f"3B shift ({len(shifted3)})")
ax3.scatter([p["score_corr"] for p in flipped3], [p["score_orig"] for p in flipped3], s=42, marker="D", color="#c0392b", edgecolors="#222222", linewidth=0.55, zorder=5, label=f"3B flip ({len(flipped3)}/94)")
ax3.scatter([p["score_corr"] for p in shifted7], [p["score_orig"] for p in shifted7], s=46, marker="^", color="#d97706", edgecolors="#222222", linewidth=0.55, zorder=6, label=f"7B shift ({len(shifted7)}/95, 0 flip)")

ax3.text(
    0.03, 0.96, "3B: 6/94 True->False\n7B: 0/95 pass flips",
    fontsize=6.9, color="#992211", fontweight="bold", va="top", ha="left",
    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#c0392b", "alpha": 0.92, "lw": 0.65},
)
ax3.set_xlim(-0.04, 1.04)
ax3.set_ylim(-0.04, 1.04)
ax3.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax3.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax3.tick_params(labelsize=7.7)
ax3.set_xlabel("Repaired weighted G-Eval score", fontsize=8.3)
ax3.set_ylabel("Buggy wire G-Eval score", fontsize=8.3)
ax3.grid(color="#ececec", linewidth=0.5, zorder=0)
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#dddddd", fontsize=6.5)
ax3.set_title("DeepEval 4.3.4 Judge (3B vs 7B)", fontsize=8.8, pad=5)

fig.savefig(
    here / "routing-geval.pdf",
    metadata={
        "Author": None,
        "Creator": "Matplotlib",
        "Title": "Multi-scale confidence routing and G-Eval judge score shifts across Qwen2.5-3B and Qwen2.5-7B",
    },
)
plt.close(fig)
