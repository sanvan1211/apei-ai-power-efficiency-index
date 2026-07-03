"""Grid carbon intensity with grid source + confidence annotations."""

import pandas as pd
import matplotlib.pyplot as plt


def build_dataframe():
    return pd.DataFrame({
        "Model": [
            "GPT-5.5",
            "Gemini 3.1 Pro",
            "Llama 4 Maverick",
            "Claude Opus 4.8",
            "Grok 4.3"
        ],
        "Grid": [
            "ERCT (Texas grid)",
            "U.S. average (proxy)",
            "U.S. average (proxy)",
            "RFCW (coal-heavy)",
            "SRTV + on-site gas blend"
        ],
        "Intensity": [
            92.81,
            97.13,
            97.13,
            115.41,
            132.30
        ],
        "Confidence": [
            "Low (proxy region)",
            "Low (proxy)",
            "Low (proxy)",
            "High",
            "Medium"
        ]
    })


def main():
    df = build_dataframe()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(df["Model"], df["Intensity"])
    ax.invert_yaxis()

    for bar, (_, row) in zip(bars, df.iterrows()):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + 1.0, y, f"{row['Intensity']:.2f}", va="center",
                fontsize=10, fontweight="bold")
        ax.text(5, y, f"{row['Grid']} | {row['Confidence']}", va="center",
                fontsize=9, alpha=0.85)

    ax.set_xlabel("Grid Carbon Intensity (μg CO$_2$e/J)", fontsize=11)
    ax.set_ylabel("")
    ax.set_title("Model Grid Carbon Intensity with Grid Source", fontsize=12, pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("carbon_intensity_full.png", dpi=300, bbox_inches="tight")
    plt.savefig("carbon_intensity_full.pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    main()
