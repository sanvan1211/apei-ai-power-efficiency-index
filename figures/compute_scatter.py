"""Throughput vs. node power scatter, with iso-efficiency (constant J/token) curves."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# name -> (throughput tok/s, node power kW)
MODELS = {
    "GPT-5.5":          (56,  7.84),
    "Claude Opus 4.8":  (62,  11.2),
    "Gemini 3.1 Pro":   (123, 11.2),
    "Llama 4 Maverick": (93,  7.84),
    "Grok 4.3":         (152, 7.84),
}

ACCENT = "#4878A8"
ISO_GRAY = "#9aa0a6"

X_MIN, X_MAX = 0, 175      # pad past Grok (152)
Y_MIN, Y_MAX = 6.0, 12.5   # pad around 7.84..11.2

ISO_LEVELS = [50, 90, 120, 150, 200]   # J/token

# Marker shape encodes node-power tier so the figure reads in grayscale.
TIER_MARKER = {7.84: "o", 11.2: "s"}

LABEL_POS = {
    "GPT-5.5":          (-10, -18, "right",  "top"),
    "Claude Opus 4.8":  (-10,  12, "right",  "bottom"),
    "Gemini 3.1 Pro":   (  0,  14, "center", "bottom"),
    "Llama 4 Maverick": (  0, -20, "center", "top"),
    "Grok 4.3":         (-12,  12, "right",  "bottom"),
}


def energy_per_token(power_kW, throughput):
    return (power_kW * 1000.0) / throughput  # J/token


def main():
    print("Computed energy per token (J/token):")
    jpt = {}
    for name_, (thru, pwr) in MODELS.items():
        jpt[name_] = energy_per_token(pwr, thru)
        print(f"  {name_:18s} {jpt[name_]:7.2f} J/token")
    print(f"  range: {min(jpt.values()):.1f} -> {max(jpt.values()):.1f} J/token")

    plt.rcParams.update({
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
        "font.family":      "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Iso-efficiency curves (constant J/token) drawn first, behind the points.
    x_line = np.array([X_MIN, X_MAX], dtype=float)

    for j in ISO_LEVELS:
        y_line = (j * x_line) / 1000.0      # kW
        ax.plot(x_line, y_line, ls="--", lw=1.0, color=ISO_GRAY, alpha=0.7, zorder=1)

        # Label each curve where it exits the plot (top edge or right edge).
        x_at_top = (Y_MAX * 1000.0) / j
        if x_at_top <= X_MAX:
            lx, ly, va, ha = x_at_top, Y_MAX, "bottom", "center"
        else:
            lx, ly, va, ha = X_MAX, (j * X_MAX) / 1000.0, "center", "left"
        ax.annotate(f"{j} J/token", xy=(lx, ly), xytext=(0, 2),
                    textcoords="offset points", color=ISO_GRAY,
                    fontsize=8.5, va=va, ha=ha, clip_on=False, zorder=1)

    for name_, (thru, pwr) in MODELS.items():
        ax.scatter(thru, pwr, marker=TIER_MARKER[pwr], s=90,
                   color=ACCENT, edgecolor="white", linewidth=0.8, zorder=3)
        dx, dy, ha, va = LABEL_POS[name_]
        ax.annotate(f"{name_}\n{jpt[name_]:.1f} J/token",
                    xy=(thru, pwr), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9.5, color="#222222", zorder=4,
                    linespacing=1.2)

    # Gemini and Claude share node power (11.2 kW) but Gemini is ~2x faster.
    ax.plot([62, 123], [11.2, 11.2], color=ACCENT, lw=1.0, ls=":", alpha=0.6, zorder=2)
    ax.annotate(
        "Same node power (11.2 kW),\nbut Gemini is ~2x faster\n"
        "-> ~half the energy per token",
        xy=(92, 11.2), xytext=(96, 9.4), textcoords="data",
        fontsize=9, color="#333333", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f6f8", ec=ISO_GRAY, lw=0.8),
        arrowprops=dict(arrowstyle="->", color="#666666", lw=1.0,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=5)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Throughput (tokens/sec)")
    ax.set_ylabel("Node power (kW)")
    ax.set_title("Energy per Token as the Fundamental Unit of LLM Inference Efficiency",
                 pad=12)

    ax.grid(True, which="major", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=9, label="7.84 kW node"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ACCENT,
               markeredgecolor="white", markersize=9, label="11.2 kW node"),
    ]
    ax.legend(handles=legend_handles, title="Node power tier",
              loc="lower right", frameon=True, fontsize=9, title_fontsize=9)

    fig.tight_layout()
    fig.savefig("compute_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig("compute_scatter.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
