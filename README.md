# APEI — AI Power Efficiency Index

An open source carbon efficiency benchmark for frontier LLMs (GPT 5.5, Claude Opus 4.8, Llama 4 Maverick, Grok 4.3, Gemini 3.1 Pro). Breaking down gCO2-equivalent/token across compute, cooling, and grid carbon pillars. 

**Measured result:** 3.998 J/token on an NVIDIA Tesla T4, instrumented with [Zeus](https://ml.energy/zeus/), against **4.785 J/token predicted** by the APEI model, a 16% overprediction. 

Most published AI-carbon figures are arithmetic on vendor spec sheets. This repo contains the measurement harness so you can check the arithmetic against a real device.

---

## What APEI computes

APEI decomposes per-token emissions into three physically multiplicative pillars:

| Pillar | Quantity | Unit |
|---|---|---|
| Compute efficiency | Hardware energy per output token | J/token |
| Cooling overhead | Facility power usage effectiveness (PUE) | dimensionless |
| Grid carbon intensity | Emissions per unit energy, by eGRID subregion | gCO₂e/J |

These are multiplied rather than combined into a weighted composite. The product is a single physical quantity — **gCO₂e per output token** — which is invariant to query length and requires no arbitrary weighting choices.

---

## What is measured vs. what is modeled

This distinction matters for interpreting every number below.

**Directly measured.** The 3.998 J/token figure. One NVIDIA Tesla T4, energy sampled via Zeus during inference.

**Modeled.** All five frontier-model results. These extrapolate from published throughput and time-to-first-token benchmarks, mapped to reported data center infrastructure and eGRID subregion carbon factors. No frontier model was measured directly — vendor inference stacks are not accessible for instrumentation.

**Bounded.** A Monte Carlo simulation over the uncertain inputs shows model rankings hold in 99.8–100% of runs. A break-even sensitivity analysis finds the top-emitter result reverses only if Trainium2 power draw is overstated by more than 35%. 

The T4 is a single, older-generation device and is not representative of frontier inference hardware. It was the silicon available. Its role is to test whether the APEI model predicts a real measurement at all — not to stand in for an H100 or a Trainium2.

---

## Results

Per-token emissions across five frontier models:

| Model | gCO₂e/token | Dominant factor |
|---|---|---|
| Grok 4.3 | 0.0081 | High throughput efficiency offsets carbon-intense on-site gas turbine grid |
| TODO | TODO | TODO |
| TODO | TODO | TODO |
| TODO | TODO | TODO |
| Claude Opus 4.8 | 0.0227 | Highest reasoning overhead in the set, on a coal-intensive grid |

**Two findings.**

Compute efficiency, not grid location, dominates per-token emissions. Grok 4.3 runs on a grid substantially more carbon-intense than the U.S. average and still achieves the lowest emissions in the set, because throughput efficiency more than compensates. The spread across the five models is 2.8×.

No model leads on all three pillars. Compute architecture, energy sourcing, and cooling infrastructure appear to be optimized independently rather than jointly in current frontier deployments.

---

## Conference Poster (draft- not finalized): 

<img width="3888" height="2592" alt="pytorch poster-3" src="https://github.com/user-attachments/assets/e7b4f81f-18bb-4ae8-89a3-d74950307130" />

## Citation

```bibtex
@misc{vandara2026apei,
  author       = {Vandara, Sanhith},
  title        = {AI Power Efficiency Index (APEI): A Framework for Measuring
                  Per-Token CO2 Emissions in Frontier LLMs},
  year         = {2026},
  howpublished = {\url{https://github.com/sanvan1211/apei-ai-power-efficiency-index}}
}
```

## Abstract: 
Frontier AI models are typically evaluated on capability and cost, with environmental impact being a secondary consideration or reported in aggregate. The AI power efficiency index (APEI) is a framework for estimating per-token carbon emissions at inference time. APEI consists of 3 quantifiable pillars: compute efficiency (hardware throughput per watt), cooling overhead (facility PUE- power usage effectiveness), and carbon intensity(gCO2 equivalent /Joule). Rather than combining these into a weighted composite score, APEI computes their physical product; the result is a single unit, gCO₂-equivalent per output token, that remains invariant to query length.

APEI is applied to five of the world's leading frontier large language models (OpenAI GPT 5.5, Anthropic Claude Opus 4.8, Google Deepmind Gemini 3.1 Pro, Meta Llama 4 Maverick, xAI Grok 4.3) using published throughput and time-to-first-answer-token (TTFAT) benchmarks mapped to model-specific data center infrastructure and eGRID subregion carbon emissions. Following analysis, it is determined that, compute efficiency, not grid location, is the dominant factor of per-token emissions: Grok 4.3, despite running on an on-site gas turbine grid with with a substantially higher carbon intensity than the U.S. average, achieves the lower emissions per token (0.0081 gCO2e token) due to higher throughput efficiency, which compensates for increased grid carbon intensity. On the other hand, Opus 4.8, operating on a coal-intensive grid and exhibiting the highest compute (reasoning) overhead in the set, is the highest emitter (0.0227 gCO₂e/token), yielding a 2.8× range in per-token emissions across the five models.

No model in the sample dominates across all three pillars, suggesting that compute architecture, energy sourcing, and cooling infrastructure are not integrated nor optimized in current frontier systems. 
