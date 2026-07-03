"""Carbon pillar figure: grid carbon intensity by model, in isolation.

Note: the final APEI ranking inverts this once compute efficiency is folded
in, so the caption spells out that Grok (the longest bar) produces the
cleanest tokens overall. Grid intensity alone does not predict per-token
emissions.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (model, value ug CO2e/J, source, confidence, tier), listed top -> bottom
ROWS_TOP_TO_BOTTOM = [
    ("GPT-5.5",           92.81,  "ERCT (Texas grid)",        "Low (proxy region)", "documented"),
    ("Gemini 3.1 Pro",    97.13,  "U.S. average (proxy)",     "Low (proxy)",        "proxy"),
    ("Llama 4 Maverick",  97.13,  "U.S. average (proxy)",     "Low (proxy)",        "proxy"),
    ("Claude Opus 4.8",  115.41,  "RFCW (coal-heavy)",        "High",               "documented"),
    ("Grok 4.3",         132.30,  "SRTV + on-site gas blend", "Medium",             "blend"),
]

US_AVERAGE = 97.1  # U.S. grid-average reference (ug CO2e/J)

ACCENT = "#4878A8"

# documented -> solid; proxy -> light fill; blend -> hatched.
TIER_STYLE = {
    "documented": dict(facecolor=ACCENT,   edgecolor=ACCENT, hatch=None, alpha=1.00),
    "proxy":      dict(facecolor=ACCENT,   edgecolor=ACCENT, hatch=None, alpha=0.45),
    "blend":      dict(facecolor="white",  edgecolor=ACCENT, hatch="//", alpha=1.00),
}


def main():
    print("Grid carbon intensity (ug CO2e/J) - verification")
    for model_, val, source, conf, tier in ROWS_TOP_TO_BOTTOM:
        print(f"  {model_:<18s} {val:7.2f}   [{tier}]  {source}")

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # barh draws y=0 at the bottom, so reverse to get the requested top->bottom order.
    rows = list(reversed(ROWS_TOP_TO_BOTTOM))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    sources = [r[2] for r in rows]
    confs = [r[3] for r in rows]
    tiers = [r[4] for r in rows]
    y_pos = range(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for y, val, source, conf, tier in zip(y_pos, values, sources, confs, tiers):
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
            2.0, y,
            f"{source}  |  {conf}",
            va="center", ha="left",
            fontsize=9.0, color="black",
            zorder=5,
        )
        ax.text(
            val + 1.5, y,
            f"{val:.2f}",
            va="center", ha="left",
            fontsize=10.5, fontweight="bold", color="black",
            zorder=4,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.7, len(rows) - 1 + 0.85)

    ax.axvline(US_AVERAGE, linestyle="--", color="0.35", linewidth=1.3, zorder=2)
    ax.text(
        US_AVERAGE, len(rows) - 1 + 0.55,
        "U.S. average",
        va="bottom", ha="center",
        fontsize=9.5, color="0.35",
    )

    ax.set_title("Grid Carbon Intensity by Model (Grid Pillar Only)",
                 pad=14, fontweight="bold", loc="left")
    ax.set_xlabel("Grid carbon intensity (ug CO2e/J)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="0.85", linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.25)

    legend_handles = [
        Patch(facecolor=ACCENT,  edgecolor=ACCENT, label="Documented lookup"),
        Patch(facecolor=ACCENT,  edgecolor=ACCENT, alpha=0.45, label="Proxy (U.S. average)"),
        Patch(facecolor="white", edgecolor=ACCENT, hatch="//", label="Blended estimate"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True, framealpha=0.95, edgecolor="0.8",
        fontsize=10, title="Evidence tier", title_fontsize=10,
    )

    fig.tight_layout()
    fig.savefig("carbon_intensity_5_2.png", dpi=300, bbox_inches="tight")
    fig.savefig("carbon_intensity_5_2.pdf", bbox_inches="tight")
    plt.show()

    print(
        "Figure X. Grid carbon intensity by model (carbon pillar in isolation). "
        "Bars are shaded by evidence tier: documented subregion lookups (GPT, "
        "Claude), U.S.-average proxies for undisclosed facilities (Gemini, Llama), "
        "and a constructed on-site-gas blend (Grok). Grok draws the most "
        "carbon-intensive power yet produces the cleanest tokens overall once "
        "compute efficiency is included (Section 5.4); grid intensity alone does "
        "not predict per-token emissions."
    )


if __name__ == '__main__':
    main()
