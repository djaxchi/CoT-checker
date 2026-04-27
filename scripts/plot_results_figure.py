import matplotlib.pyplot as plt
import numpy as np

methods = [
    "Qwen2.5-0.5B\nself-judge",
    "Qwen2.5-Math-7B\nself-judge",
    "Linear probe\n(SSAE latents)",
    "MLP probe\n(SSAE latents)",
]

params_label = ["494M", "7.6B", "897", "1.4M"]

f1     = [0.377, 0.662, 0.742, 0.746]
f1_err = [0,     0,     0.001, 0.003]

x = np.arange(len(methods))
colors = ["#7bafd4", "#4a90c4", "#e07b39", "#c0392b"]

fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(
    x, f1, 0.55,
    yerr=[f1_err, f1_err],
    color=colors, edgecolor="white", linewidth=0.8,
    capsize=5, error_kw={"elinewidth": 1.8, "ecolor": "#444"},
    zorder=3,
)

# Combined tick labels: method name + params on separate lines
tick_labels = [f"{m}\n({p})" for m, p in zip(methods, params_label)]
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=11)
ax.set_ylabel("Macro F1", fontsize=12)
ax.set_ylim(0, 0.92)
ax.set_title("Step-Level CoT Verification: Macro F1 Comparison", fontsize=13, fontweight="bold", pad=12)

ax.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels on top of bars
for bar, val, err in zip(bars, f1, f1_err):
    label = f"{val:.3f}" + (f" ±{err:.3f}" if err else "")
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (err or 0) + 0.012,
            label, ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
out = "results/method_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
