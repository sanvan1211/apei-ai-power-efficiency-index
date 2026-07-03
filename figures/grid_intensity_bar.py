"""Grid carbon intensity by model - horizontal bar chart, saved as png + pdf.

Tiers drive the fill style (not hue) so the figure survives grayscale.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# model -> (grid carbon intensity lb CO2e/MWh, evidence tier)
DATA = [
    ("Grok 4.3",         1050.0, "blended"),  # blended on-site gas + grid
    ("Claude Opus 4.8",   916.0, "direct"),   # eGRID RFCW lookup
    ("Gemini 3.1 Pro",    770.9, "proxy"),    # US-average proxy
    ("Llama 4 Maverick",  770.9, "proxy"),    # US-average proxy
    ("GPT-5.5",           736.6, "direct"),   # eGRID ERCT lookup
]

US_AVERAGE = 770.9  # U.S. grid-average reference (lb CO2e/MWh)

ACCENT = "#4878A8"

# direct -> solid; blended -> hatched; proxy -> light fill. Grayscale-safe.
TIER_STYLE = {
    "direct":  dict(facecolor=ACCENT, edgecolor=ACCENT, hatch=None, alpha=1.0),
    "blended": dict(facecolor="white", edgecolor=ACCENT, hatch="//", alpha=1.0),
    "proxy":   dict(facecolor=ACCENT, edgecolor=ACCENT, hatch=None, alpha=0.28),
}


def main():
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "pdf.fonttype": 42,     # embed real TrueType fonts in the PDF
        "ps.fonttype": 42,
    })

    # Sort ascending so barh puts the largest value on top.
    data_sorted = sorted(DATA, key=lambda r: r[1])
    labels = [r[0] for r in data_sorted]
    values = [r[1] for r in data_sorted]
    tiers = [r[2] for r in data_sorted]
    y_pos = range(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))

    for y, val, tier in zip(y_pos, values, tiers):
        style = TIER_STYLE[tier]
        ax.barh(
            y, val,
            height=0.62,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            alpha=style["alpha"],
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            val + 12, y,
            f"{val:.0f} lb CO2e/MWh",
            va="center", ha="left",
            fontsize=10.5, color="black",
            zorder=4,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(labels) - 1 + 0.95)

    ax.axvline(US_AVERAGE, linestyle="--", color="0.35", linewidth=1.3, zorder=2)
    ax.text(
        US_AVERAGE, len(labels) - 1 + 0.6,
        "U.S. average",
        rotation=0, va="bottom", ha="center",
        fontsize=9.5, color="0.35",
    )

    ax.set_title("Grid Carbon Intensity by Model Serving Location",
                 pad=14, fontweight="bold", loc="left")
    ax.set_xlabel("Grid carbon intensity (lb CO2e/MWh)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="0.85", linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.34)

    legend_handles = [
        Patch(facecolor=ACCENT, edgecolor=ACCENT, label="Direct lookup"),
        Patch(facecolor="white", edgecolor=ACCENT, hatch="//", label="Blended estimate"),
        Patch(facecolor=ACCENT, edgecolor=ACCENT, alpha=0.28, label="Proxy"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        framealpha=0.95,
        edgecolor="0.8",
        fontsize=10,
        title="Evidence tier",
        title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig("grid_intensity.png", dpi=300, bbox_inches="tight")
    fig.savefig("grid_intensity.pdf", bbox_inches="tight")
    plt.show()

    print(
        "Figure X. Grok's value is a documented on-site-gas blend, not an eGRID "
        "lookup; it exceeds all grid subregions because dedicated fossil "
        "generation is not reflected in grid averages."
    )


if __name__ == '__main__':
    main()
