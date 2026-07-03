"""APEI Monte Carlo robustness simulation.

    gCO2e_per_token = (P_node / T) * PUE * I_grid,
    I_grid [g/J] = lb_per_MWh * 453.592 / 3.6e9

All inputs are locked point estimates / bands; jitter only samples within
the bands.
"""

import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N = 10000
LB_TO_G_PER_J = 453.592 / 3.6e9  # lb CO2e/MWh -> g/J

# T and P_node are not jittered: no documented uncertainty for throughput or
# node power. Set these fractions > 0 if a tolerance is ever documented.
T_JITTER_FRAC = 0.0     # e.g. 0.05 for +/-5% uniform on throughput
P_JITTER_FRAC = 0.0     # e.g. 0.03 for +/-3% uniform on node power

NAMES = ["Grok 4.3", "Llama 4 Maverick", "Gemini 3.1 Pro", "GPT-5.5", "Claude Opus 4.8"]
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]  # Okabe-Ito

T = np.array([152.0,    93.0,     123.0,    56.0,     62.0])      # tok/s
P_NODE = np.array([7840.0,   7840.0,   11200.0,  7840.0,   11200.0])   # W
PUE_LO = np.array([1.17,     1.12,     1.08,     1.12,     1.08])      # PUE band low
PUE_HI = np.array([1.19,     1.16,     1.10,     1.16,     1.10])      # PUE band high
GRID_PT = np.array([1050.0,   770.884,  770.884,  736.6,    916.0])     # lb/MWh point
GRID_LO = np.array([903.0,    596.0,    596.0,    635.0,    635.0])     # lb/MWh band low
GRID_HI = np.array([1150.0,   927.0,    927.0,    900.0,    916.0])     # lb/MWh band high
# Grid jitter sub-band as fraction of point, by confidence:
#   HIGH (Grok, Claude) +/-5%; MEDIUM (GPT-5.5) +/-15%; LOW (Llama, Gemini) full band.
GRID_FRAC = np.array([0.05,    np.nan,   np.nan,   0.15,     0.05])
GRID_FULL = np.array([False,   True,     True,     False,    False])  # True => use entire band

# PUE bands for the fallback cooling models (Llama, GPT-5.5) are placeholders
# (+/-0.02), not locked. Flagged in the printout below.
PUE_PLACEHOLDER = np.array([False, True, False, True, False])

GROK, LLAMA, GEMINI, GPT, CLAUDE = 0, 1, 2, 3, 4


def run_simulation():
    """Run the Monte Carlo sweep. Returns the (N, M) array of gCO2e/token draws."""
    np.random.seed(SEED)
    M = len(NAMES)

    # Jitter draws: shape (N, M), one row per Monte Carlo run.
    pue = np.random.uniform(PUE_LO, PUE_HI, size=(N, M))

    # Grid: LOW models span the full band; HIGH/MEDIUM use a point +/- frac
    # sub-band clipped to the full documented band. Claude's point (916,
    # eGRID RFCW) sits at the top of its band, so its +/-5% sub-band clips to
    # downside-only draws - the sim never invents a dirtier-than-documented
    # Claude, which is the conservative setup for "Claude dirtiest".
    sub_lo = np.where(GRID_FULL, GRID_LO,
                      np.clip(GRID_PT - np.nan_to_num(GRID_FRAC) * GRID_PT, GRID_LO, GRID_HI))
    sub_hi = np.where(GRID_FULL, GRID_HI,
                      np.clip(GRID_PT + np.nan_to_num(GRID_FRAC) * GRID_PT, GRID_LO, GRID_HI))
    grid_lb = np.random.uniform(sub_lo, sub_hi, size=(N, M))

    T_draw = T * np.random.uniform(1 - T_JITTER_FRAC, 1 + T_JITTER_FRAC, size=(N, M)) \
        if T_JITTER_FRAC > 0 else np.broadcast_to(T, (N, M))
    P_draw = P_NODE * np.random.uniform(1 - P_JITTER_FRAC, 1 + P_JITTER_FRAC, size=(N, M)) \
        if P_JITTER_FRAC > 0 else np.broadcast_to(P_NODE, (N, M))

    # APEI per run: gCO2e per token for all 5 models, shape (N, M).
    g = (P_draw / T_draw) * pue * (grid_lb * LB_TO_G_PER_J)
    return g


def print_summary(g):
    cleanest = np.argmin(g, axis=1)
    dirtiest = np.argmax(g, axis=1)

    p_grok_clean = np.mean(cleanest == GROK)
    p_claude_dirty = np.mean(dirtiest == CLAUDE)
    p_joint = np.mean((cleanest == GROK) & (dirtiest == CLAUDE))

    # Grok in top-2 cleanest: rank (0 = cleanest) is 0 or 1.
    grok_rank = (g < g[:, [GROK]]).sum(axis=1)
    p_grok_top2 = np.mean(grok_rank <= 1)

    # Do Llama, Gemini, GPT-5.5 keep point-estimate order (Llama < Gemini < GPT-5.5)?
    mid = g[:, [LLAMA, GEMINI, GPT]]
    mid_in_order = (mid[:, 0] < mid[:, 1]) & (mid[:, 1] < mid[:, 2])
    frac_mid_reorder = np.mean(~mid_in_order)

    p05, p95 = np.percentile(g, [5, 95], axis=0)

    print(f"APEI Monte Carlo robustness  |  N = {N}  |  np.random.seed = {SEED}")
    print(">> PRELIMINARY: Llama/GPT-5.5 PUE bands are unconfirmed placeholders "
          "(see FLAG below); every number here carries an asterisk until locked. <<")
    print("ASSUMPTION: throughput (T) and node power (P_node) are NOT jittered "
          "(no documented uncertainty).")
    print("METHODOLOGY (Claude grid band): point estimate 916 lb/MWh (eGRID RFCW, "
          "'direct' tier) is the documented site ceiling; its band is bounded above "
          "by that value, so Claude takes downside-only grid draws. 'Claude dirtiest' "
          "is tested against Claude's cleanest plausible grid, not a stacked deck.\n")
    print(f"P(Grok cleanest)               : {p_grok_clean * 100:5.1f}%")
    print(f"P(Claude dirtiest)             : {p_claude_dirty * 100:5.1f}%")
    print(f"P(Grok cleanest & Claude dirty): {p_joint * 100:5.1f}%   <- strict joint claim")
    print(f"P(Grok in top-2 cleanest)      : {p_grok_top2 * 100:5.1f}%   (softer secondary stat)")
    print(f"Fraction of runs where middle-three (Llama<Gemini<GPT-5.5) reorder: "
          f"{frac_mid_reorder:.4f}\n")
    print("Empirical gCO2e/token band per model  [5th pct, point-ish, 95th pct]:")
    g_point = (P_NODE / T) * 0.5 * (PUE_LO + PUE_HI) * (GRID_PT * LB_TO_G_PER_J)
    for i, nm in enumerate(NAMES):
        print(f"  {nm:18s}: [{p05[i]:.4e}, {g_point[i]:.4e}, {p95[i]:.4e}]")
    print("\nFLAG: PUE bands for Llama 4 Maverick and GPT-5.5 are PLACEHOLDERS "
          "(+/-0.02 'fallback' cooling), NOT locked -- confirm against Section 3.3 "
          "before publishing.")


def plot_histograms(g):
    """Overlapping semi-transparent histograms, one color per model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(g.min(), g.max(), 60)
    for i, nm in enumerate(NAMES):
        ax.hist(g[:, i], bins=bins, color=COLORS[i], alpha=0.55, label=nm, edgecolor="none")
    ax.set_xlabel("Carbon emitted per token generated  (grams CO2e / token)")
    ax.set_ylabel("Number of Monte Carlo runs in this range")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("apei_montecarlo.pdf")
    fig.savefig("apei_montecarlo.png", dpi=200)
    print("\nSaved: apei_montecarlo.pdf, apei_montecarlo.png")


def main():
    g = run_simulation()
    print_summary(g)
    plot_histograms(g)


if __name__ == '__main__':
    main()
