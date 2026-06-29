# apei monte carlo robustness simulation
# simulates uncertainty in pue and grid carbon intensity using monte carlo sampling
# estimates the distribution of carbon emissions (gco2e/token) for each frontier ai model
# outputs summary statistics and a histogram comparing all five models

import numpy as np
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
N = 10000
LB_TO_G_PER_J = 453.592 / 3.6e9

# uncertainty settings
T_JITTER_FRAC = 0.0
P_JITTER_FRAC = 0.0

# model names and colors
NAMES = ["Grok 4.3", "Llama 4 Maverick", "Gemini 3.1 Pro", "GPT-5.5", "Claude Opus 4.8"]
COLORS = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]

# model inputs
T = np.array([152.0, 93.0, 123.0, 56.0, 62.0])
P_NODE = np.array([7840.0, 7840.0, 11200.0, 7840.0, 11200.0])

PUE_LO = np.array([1.17, 1.12, 1.08, 1.12, 1.08])
PUE_HI = np.array([1.19, 1.16, 1.10, 1.16, 1.10])

GRID_PT = np.array([1050.0, 770.884, 770.884, 736.6, 916.0])
GRID_LO = np.array([903.0, 596.0, 596.0, 635.0, 635.0])
GRID_HI = np.array([1150.0, 927.0, 927.0, 900.0, 916.0])

GRID_FRAC = np.array([0.05, np.nan, np.nan, 0.15, 0.05])
GRID_FULL = np.array([False, True, True, False, False])

M = len(NAMES)

# sample pue and grid values
pue = np.random.uniform(PUE_LO, PUE_HI, size=(N, M))

sub_lo = np.where(
    GRID_FULL,
    GRID_LO,
    np.clip(GRID_PT - np.nan_to_num(GRID_FRAC) * GRID_PT, GRID_LO, GRID_HI),
)

sub_hi = np.where(
    GRID_FULL,
    GRID_HI,
    np.clip(GRID_PT + np.nan_to_num(GRID_FRAC) * GRID_PT, GRID_LO, GRID_HI),
)

grid_lb = np.random.uniform(sub_lo, sub_hi, size=(N, M))

# optionally jitter throughput and node power
if T_JITTER_FRAC > 0:
    T_draw = T * np.random.uniform(
        1 - T_JITTER_FRAC, 1 + T_JITTER_FRAC, size=(N, M)
    )
else:
    T_draw = np.broadcast_to(T, (N, M))

if P_JITTER_FRAC > 0:
    P_draw = P_NODE * np.random.uniform(
        1 - P_JITTER_FRAC, 1 + P_JITTER_FRAC, size=(N, M)
    )
else:
    P_draw = np.broadcast_to(P_NODE, (N, M))

# compute apei values
g = (P_draw / T_draw) * pue * (grid_lb * LB_TO_G_PER_J)

# identify cleanest and dirtiest models
cleanest = np.argmin(g, axis=1)
dirtiest = np.argmax(g, axis=1)

GROK, LLAMA, GEMINI, GPT, CLAUDE = 0, 1, 2, 3, 4

p_grok_clean = np.mean(cleanest == GROK)
p_claude_dirty = np.mean(dirtiest == CLAUDE)
p_joint = np.mean((cleanest == GROK) & (dirtiest == CLAUDE))

# additional robustness statistics
grok_rank = (g < g[:, [GROK]]).sum(axis=1)
p_grok_top2 = np.mean(grok_rank <= 1)

mid = g[:, [LLAMA, GEMINI, GPT]]
mid_in_order = (mid[:, 0] < mid[:, 1]) & (mid[:, 1] < mid[:, 2])
frac_mid_reorder = np.mean(~mid_in_order)

p05, p95 = np.percentile(g, [5, 95], axis=0)

# print summary
print(f"Monte Carlo Simulation (N = {N})")
print(f"P(Grok cleanest): {p_grok_clean*100:.1f}%")
print(f"P(Claude dirtiest): {p_claude_dirty*100:.1f}%")
print(f"P(Grok cleanest and Claude dirtiest): {p_joint*100:.1f}%")
print(f"P(Grok in top 2 cleanest): {p_grok_top2*100:.1f}%")
print(f"Middle-model reorder rate: {frac_mid_reorder:.4f}\n")

g_point = (
    (P_NODE / T)
    * 0.5 * (PUE_LO + PUE_HI)
    * (GRID_PT * LB_TO_G_PER_J)
)

print("gCO2e/token (5th percentile, reference, 95th percentile)")
for i, name in enumerate(NAMES):
    print(f"{name:18s}: [{p05[i]:.4e}, {g_point[i]:.4e}, {p95[i]:.4e}]")

# plot distributions
fig, ax = plt.subplots(figsize=(8, 5))

bins = np.linspace(g.min(), g.max(), 60)

for i, name in enumerate(NAMES):
    ax.hist(
        g[:, i],
        bins=bins,
        color=COLORS[i],
        alpha=0.55,
        label=name,
        edgecolor="none",
    )

ax.set_xlabel("Carbon emissions (gCO₂e/token)")
ax.set_ylabel("Frequency")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("apei_montecarlo.pdf")
fig.savefig("apei_montecarlo.png", dpi=200)

print("Saved: apei_montecarlo.pdf, apei_montecarlo.png")
