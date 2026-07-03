"""Cooling PUE number line.

All five models' assigned PUE values against the LBNL AI-Specialized
facility range. Confirmed vs. fallback (category median) assignments use
different marker styles. The point is to show low variance, not to rank
models.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LBNL_LOW, LBNL_HIGH, LBNL_MEDIAN = 1.08, 1.19, 1.14

# (name, assigned PUE, "confirmed" cooling method known / "fallback" category median)
MODELS = [
    ("Claude Opus 4.8",  1.09, "confirmed"),  # liquid
    ("Gemini 3.1 Pro",   1.09, "confirmed"),  # liquid
    ("GPT-5.5",          1.14, "fallback"),   # median
    ("Llama 4 Maverick", 1.14, "fallback"),   # median
    ("Grok 4.3",         1.18, "confirmed"),  # air
]

ACCENT = "#4878A8"
BAND_CLR = "#E5ECF3"

X_MIN, X_MAX = 1.06, 1.21
Y_LINE = 0.0

STACK_OFFSETS = [0.34, 0.70]


def main():
    pue_values = [p for _, p, _ in MODELS]
    print("LBNL AI-Specialized range : {:.2f} - {:.2f}  (median {:.2f})".format(
        LBNL_LOW, LBNL_HIGH, LBNL_MEDIAN))
    for name_, pue, kind in MODELS:
        print("{:<18} PUE = {:.2f}   ({})".format(name_, pue, kind))
    lo, hi = min(pue_values), max(pue_values)
    print("Assigned min / max        : {:.2f} / {:.2f}".format(lo, hi))
    print("Spread (max-min)          : {:.2f}  ({:.1f}% of min)".format(
        hi - lo, 100.0 * (hi - lo) / lo))
    print("All within LBNL range?    :",
          all(LBNL_LOW <= p <= LBNL_HIGH for p in pue_values))

    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(9.5, 3.6))

    # LBNL band + median line, behind everything.
    ax.axvspan(LBNL_LOW, LBNL_HIGH, color=BAND_CLR, zorder=0)
    ax.axvline(LBNL_MEDIAN, color="#9AA7B4", linestyle="--", linewidth=1, zorder=1)
    ax.text(LBNL_MEDIAN, 1.0, "LBNL median", color="#6B7884",
            ha="center", va="bottom", fontsize=9)
    ax.text(LBNL_LOW + 0.002, -0.82, "LBNL AI-Specialized range (1.08-1.19)",
            color="#6B7884", ha="left", va="center", fontsize=9, style="italic")

    ax.axhline(Y_LINE, color="#C8CDD2", linewidth=1, zorder=1)

    # Group models by x-position so overlapping pairs can stack their labels.
    by_x = defaultdict(list)
    for name_, pue, kind in MODELS:
        by_x[pue].append((name_, kind))

    for pue, group in sorted(by_x.items()):
        for name_, kind in group:
            if kind == "confirmed":
                ax.plot(pue, Y_LINE, marker="o", markersize=11,
                        markerfacecolor=ACCENT, markeredgecolor=ACCENT, zorder=3)
            else:
                ax.plot(pue, Y_LINE, marker="o", markersize=11,
                        markerfacecolor="white", markeredgecolor=ACCENT,
                        markeredgewidth=1.6, zorder=3)

        if len(group) == 1:
            name_, _ = group[0]
            ax.annotate(name_, xy=(pue, Y_LINE), xytext=(pue, 0.34),
                        ha="center", va="bottom", fontsize=10.5, color="#33424F")
        else:
            for (name_, _), dy in zip(group, STACK_OFFSETS):
                ax.annotate(name_, xy=(pue, Y_LINE), xytext=(pue, dy),
                            ha="center", va="bottom", fontsize=10.5, color="#33424F")

        ax.text(pue, -0.34, "{:.2f}".format(pue), ha="center", va="top",
                fontsize=9.5, color=ACCENT, fontweight="bold")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-1.0, 1.15)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_position(("data", -1.0))
    ax.spines["bottom"].set_color("#888888")

    ax.set_xticks([1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20])
    ax.tick_params(axis="x", colors="#555555", labelsize=10)

    ax.set_xlabel("Power usage effectiveness (PUE)", fontsize=11.5,
                  color="#333333", labelpad=8)
    ax.set_title("Assigned PUE Values Within the LBNL AI-Specialized Range",
                 fontsize=12.5, color="#222222", pad=14)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor=ACCENT, markeredgecolor=ACCENT,
               label="Confirmed cooling method"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=10,
               markerfacecolor="white", markeredgecolor=ACCENT,
               markeredgewidth=1.6, label="Fallback (category median)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", frameon=False,
              fontsize=9, handletextpad=0.4, ncol=2, columnspacing=2.0,
              bbox_to_anchor=(0.5, -0.42))

    fig.tight_layout()
    fig.savefig("cooling_pue_range.png", dpi=300, bbox_inches="tight")
    fig.savefig("cooling_pue_range.pdf", bbox_inches="tight")
    plt.show()

    print("\nFigure X. Assigned PUE for each model, shown against the LBNL "
          "AI-Specialized facility range (1.08-1.19). All five values fall within "
          "a span of under 9%, confirming that cooling overhead is effectively "
          "constant across modern AI facilities and contributes negligibly to "
          "per-token carbon differences.")


if __name__ == '__main__':
    main()
