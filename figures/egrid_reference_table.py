"""EPA eGRID2023 subregion table -> LaTeX (booktabs), Markdown, and a matplotlib image."""

import matplotlib.pyplot as plt
import pandas as pd

DATA = [
    ["RFCW", "Midwest (coal-heavy)", 916.1, 115.4],
    ["SRTV", "Tennessee Valley", 903.3, 113.8],
    ["U.S. average", "National mean", 770.9, 97.1],
    ["ERCT", "Texas (ERCOT)", 736.6, 92.8],
    ["NWPP", "Pacific Northwest", 635.3, 80.0],
    ["SRVC", "Carolinas", 596.4, 75.1],
    ["CAMX", "California", 430.0, 54.2],
]

COLUMNS = ["Subregion", "Region", "lb CO2e/MWh", "ug CO2e/J"]


def build_dataframe():
    return pd.DataFrame(DATA, columns=COLUMNS)


def to_latex(df_fmt, baseline_idx):
    latex = r"""
\begin{table}[h]
\centering
\caption{EPA eGRID2023 (Rev. 2) subregion carbon intensities used or referenced in this study. Conversion: ug CO2e/J = lb/MWh x 453.592 / 3.6e9.}
\label{tab:egrid_reference}
\begin{tabular}{l l r r}
\toprule
Subregion & Region & lb CO$_2$e/MWh & ug CO$_2$e/J \\
\midrule
"""
    for i, row in df_fmt.iterrows():
        if i == baseline_idx:
            latex += (
                r"\textbf{" + f"{row['Subregion']}" + r"} & \textbf{" + f"{row['Region']}"
                + r"} & \textbf{" + f"{row['lb CO2e/MWh']}" + r"} & \textbf{"
                + f"{row['ug CO2e/J']}" + r"} \\" + "\n"
            )
        else:
            latex += f"{row['Subregion']} & {row['Region']} & {row['lb CO2e/MWh']} & {row['ug CO2e/J']} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def print_markdown(df_fmt, baseline_idx):
    header = "| Subregion | Region | lb CO2e/MWh | ug CO2e/J |"
    sep = "|---|---|---:|---:|"
    print(header)
    print(sep)
    for i, row in df_fmt.iterrows():
        if i == baseline_idx:
            print(f"| **{row['Subregion']}** | **{row['Region']}** | "
                  f"**{row['lb CO2e/MWh']}** | **{row['ug CO2e/J']}** |")
        else:
            print(f"| {row['Subregion']} | {row['Region']} | "
                  f"{row['lb CO2e/MWh']} | {row['ug CO2e/J']} |")


def save_table_image(df, df_fmt):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis('off')
    table_data = df_fmt.values.tolist()
    tbl = ax.table(cellText=table_data, colLabels=COLUMNS, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#E6E6E6')
        else:
            if r % 2 == 0:
                cell.set_facecolor('#F7F7F7')
            if df.iloc[r - 1]["Subregion"] == "U.S. average":
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#D9EDF7')

    plt.tight_layout()
    plt.savefig("table_egrid_reference.png", dpi=300, bbox_inches='tight')
    plt.savefig("table_egrid_reference.pdf", bbox_inches='tight')
    plt.show()


def main():
    df = build_dataframe()

    df_fmt = df.copy()
    df_fmt["lb CO2e/MWh"] = df_fmt["lb CO2e/MWh"].map(lambda x: f"{x:.1f}")
    df_fmt["ug CO2e/J"] = df_fmt["ug CO2e/J"].map(lambda x: f"{x:.1f}")

    baseline_idx = df.index[df["Subregion"] == "U.S. average"][0]

    print(to_latex(df_fmt, baseline_idx))
    print_markdown(df_fmt, baseline_idx)
    save_table_image(df, df_fmt)


if __name__ == '__main__':
    main()
