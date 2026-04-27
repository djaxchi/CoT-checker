import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

methods = [
    "MLP probe (SSAE latents)",
    "Linear probe (SSAE latents)",
    "Qwen2.5-0.5B self-judge",
    "Qwen2.5-Math-7B self-judge",
]

# steps/sec: probes from benchmark logs, judges estimated from tqdm (batch=32)
throughput = [
    np.mean([5242825, 5171185, 5039260, 5217241]),   # MLP probe
    np.mean([29555080, 29551648, 29777825, 27348267]), # linear probe
    17.0 * 32,   # 0.5B judge
    14.0 * 32,   # 7B judge
]

colors = ["#c0392b", "#e07b39", "#7bafd4", "#4a90c4"]

fig, ax = plt.subplots(figsize=(11, 1.7))
fig.subplots_adjust(left=0.30, right=0.94, top=0.82, bottom=0.28)

y = np.arange(len(methods))
bars = ax.barh(y, throughput, height=0.6, color=colors, edgecolor="none")

ax.set_xscale("log")
ax.set_xlim(100, 5e7)
ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=9, va="center", ha="right")
ax.set_xlabel("Steps / second  (log scale)", fontsize=8.5, labelpad=4)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else (f"{v/1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")
))
ax.tick_params(axis="x", labelsize=8)
ax.tick_params(axis="y", length=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, which="both", linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Value labels inside/beside bars
for bar, val in zip(bars, throughput):
    if val >= 1e6:
        label = f"{val/1e6:.1f}M"
    elif val >= 1e3:
        label = f"{val/1e3:.0f}k"
    else:
        label = f"{val:.0f}"
    ax.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8.5, fontweight="bold", color="#333")

out = "results/throughput_footer.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
