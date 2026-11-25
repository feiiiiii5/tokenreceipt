"""Plot Figure 5: (Left) Qwen2.5-7B GRPO Step-0 clipping censoring/attenuation & (Right) 128-seed SST-2 forest plot."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np

here = Path(__file__).resolve().parent
d24_grpo = here.parents[2] / "reproducibility" / "downstream_2026_11_24" / "grpo_7b_step0.json"
grpo_cache = here / "grpo-step0.json"

if d24_grpo.exists():
    summary = json.loads(d24_grpo.read_text(encoding="utf-8"))["expected_summary"]
    grpo_cache.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
else:
    summary = json.loads(grpo_cache.read_text(encoding="utf-8"))

with (here / "learning.csv").open(encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 2.75), dpi=300, layout="constrained")

# Left panel: Qwen2.5-7B GRPO Step-0 asymmetric distortion across epsilons
eps_keys = ["0.1", "0.2", "0.28"]
x = np.arange(len(eps_keys))
w = 0.25
neg_zero = [100.0 * summary["epsilons"][k]["neg_adv"]["hard_zeroed"] / summary["epsilons"][k]["neg_adv"]["score_corrupt"] for k in eps_keys]
pos_atten = [100.0 * summary["epsilons"][k]["pos_adv"]["attenuated"] / summary["epsilons"][k]["pos_adv"]["score_corrupt"] for k in eps_keys]
neg_mult = [100.0 * summary["epsilons"][k]["neg_adv"]["mean_grad_mult_on_corrupt"] for k in eps_keys]

ax_l.bar(x - w, neg_zero, w, label=r"$\hat{A}<0$ hard-zeroed (% of 110 corrupt)", color="#994715", edgecolor="#6e320e", linewidth=0.55)
ax_l.bar(x, pos_atten, w, label=r"$\hat{A}>0$ attenuated (% of 92 corrupt)", color="#12675d", edgecolor="#0d4a43", linewidth=0.55)
ax_l.bar(x + w, neg_mult, w, label=r"$\hat{A}<0$ mean grad mult ($\times 100$)", color="#7f8c8d", edgecolor="#4d5656", linewidth=0.55)

for xi, v1, v2, v3 in zip(x, neg_zero, pos_atten, neg_mult):
    ax_l.text(xi - w, v1 + 2.0, f"{v1:.1f}", ha="center", va="bottom", fontsize=6.6, color="#6e320e", fontweight="bold")
    ax_l.text(xi, v2 + 2.0, f"{v2:.0f}", ha="center", va="bottom", fontsize=6.6, color="#0d4a43", fontweight="bold")
    ax_l.text(xi + w, v3 + 2.0, f"{v3:.1f}", ha="center", va="bottom", fontsize=6.6, color="#333333")

ax_l.set_xticks(x, [rf"$\epsilon = {k}$" + ("\n(PPO std)" if k == "0.2" else "") for k in eps_keys], fontsize=7.6)
ax_l.set_ylabel("Share / retained gradient (%)", fontsize=8.1)
ax_l.set_ylim(0, 166)
ax_l.set_yticks([0, 25, 50, 75, 100])
ax_l.tick_params(axis="y", labelsize=7.5)
ax_l.grid(axis="y", color="#dddddd", linewidth=0.5)
ax_l.spines[["top", "right"]].set_visible(False)
ax_l.legend(loc="upper left", frameon=True, framealpha=0.94, edgecolor="#dddddd", fontsize=6.0, labelspacing=0.25, borderpad=0.3)
ax_l.set_title("Qwen2.5-7B GRPO Step-0 (200 rollouts, 8,328 pos)", fontsize=8.3, pad=5)
sk = summary["spurious_kl_step0"]
ax_l.text(
    0.98, 0.97,
    f"Spurious step-0 KL > 0:\n{sk['rollouts_nonzero_kl']}/200 rollouts\n(corrupt mean {sk['per_corrupted_position_kl_mean']:.3f})",
    transform=ax_l.transAxes, ha="right", va="top", fontsize=6.0, color="#222222",
    bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#cccccc", "alpha": 0.94, "lw": 0.6},
)

# Right panel: Paired held-out reward contrasts (128-seed SST-2 confirmation)
ax_r.axvline(0, color="#555", linewidth=0.8, ls="--")
labels = []
transform = blended_transform_factory(ax_r.transAxes, ax_r.transData)
ys = [4.7, 3.7, 2.7, 1.7, 0]
for index, r in enumerate(rows):
    y = ys[index]
    lo, point, hi = (float(r[k]) for k in ("lo", "mean", "hi"))
    color = "#12675d" if index == 4 else "#555555"
    ax_r.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="s" if index == 4 else "o", color=color, capsize=3, ms=4.8, lw=1.4)
    ax_r.text(0.98, y, f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]", fontsize=6.9, va="center", ha="right", transform=transform, color=color, fontweight="bold" if index == 4 else "normal")
    labels.append(f'{r["task"]}, T={r["temperature"]} ({r["seeds"]}s)')
ax_r.set_yticks(ys, labels, fontsize=7.3)
ax_r.set_ylim(-0.65, 5.3)
ax_r.set_xlim(-9.8, 10.2)
ax_r.set_xticks([-8, -4, 0, 4])
ax_r.tick_params(axis="x", labelsize=7.5)
ax_r.tick_params(axis="y", length=0, pad=5)
ax_r.set_xlabel("Expected correctness diff (percentage points)", fontsize=8.1)
ax_r.spines[["top", "right"]].set_visible(False)
ax_r.set_title("Online Policy-Gradient Confirmation (SST-2)", fontsize=8.3, pad=5)

fig.savefig(
    here / "learning.pdf",
    metadata={"Title": "GRPO step-0 gradient censoring and held-out SST-2 policy-gradient learning contrast", "Author": None, "Creator": "Matplotlib"},
)
plt.close(fig)
