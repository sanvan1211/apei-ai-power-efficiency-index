"""
Break-even sensitivity analysis for the Trainium2 per-chip power estimate.
Supports Section 6.2.7 of the APEI paper.

The Monte Carlo simulation (Section 5.4.1) varies grid intensity and PUE but
holds the two inferred hardware figures (Trainium2 at 500 W, TPU v7 at 1,000 W
per chip) at their point estimates. Because each figure derives from a single
technical source (Patel et al., 2024 for Trainium2) rather than a characterized
probability distribution, a break-even threshold is the appropriate robustness
tool: instead of assuming an arbitrary spread, it asks how far the point
estimate would have to move to change the result.

Claude Opus 4.8's final per-token carbon is strictly linear in Trainium2
per-chip power: node power = chips_per_node * chip_watts * overhead, and every
downstream quantity (J/token, gCO2e/token) is a direct multiplication of node
power. Therefore the break-even chip power at which Claude's APEI equals the
next-highest model (GPT-5.5, on confirmed H100 hardware, so unaffected by this
perturbation) is:

    breakeven_W = 500 * (APEI_gpt / APEI_claude)

The cleanest endpoint (Grok 4.3 vs. Llama 4 Maverick) involves only confirmed
H100 hardware and depends on no inferred power figure.
"""

# Final APEI values (gCO2e per generated token), Section 5.4 of the paper.
# If the upstream pipeline produces higher-precision unrounded values, replace
# these with those outputs and re-run.
APEI_CLAUDE = 0.02273   # Claude Opus 4.8 (Trainium2, 500 W inferred)
APEI_GPT = 0.01481      # GPT-5.5 (H100, 700 W confirmed TDP)

TRAINIUM2_ESTIMATE_W = 500.0  # Patel et al. (2024), SemiAnalysis

breakeven_W = TRAINIUM2_ESTIMATE_W * (APEI_GPT / APEI_CLAUDE)
pct_reduction = (TRAINIUM2_ESTIMATE_W - breakeven_W) / TRAINIUM2_ESTIMATE_W * 100

print("Break-even sensitivity: Trainium2 per-chip power")
print(f"  Claude Opus 4.8 APEI at 500 W estimate : {APEI_CLAUDE} gCO2e/token")
print(f"  GPT-5.5 APEI (confirmed H100 hardware)  : {APEI_GPT} gCO2e/token")
print(f"  Break-even Trainium2 power              : {breakeven_W:.1f} W")
print(f"  Reduction from 500 W estimate           : {pct_reduction:.1f}%")
print()
print("Interpretation: Claude Opus 4.8 remains the highest-emitting model")
print(f"unless the true Trainium2 per-chip draw is below ~{breakeven_W:.0f} W,")
print(f"i.e. unless the Patel et al. (2024) estimate is overstated by more")
print(f"than ~{pct_reduction:.0f}%. The cleanest endpoint (Grok 4.3) depends on no")
print("inferred power figure: Grok and Llama 4 Maverick both run on confirmed")
print("H100 hardware.")
