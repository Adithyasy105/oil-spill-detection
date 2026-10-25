# Novelty Analysis & Prior Art Differentiation

## 1. Established Prior Work vs. Our Implementation
- **Prior Art**: Decision-level gating (e.g. mixture of experts, entropy weighting) has been used in vision and speech, but typically assumes equal expert quality and pixel-registered sensors.
- **Our Implementation**: CAFE-Fusion v2 addresses the unique challenge of **asymmetric multimodal sensor quality** without pixel registration.
- **Novelty Core**:
  1. **Asymmetric Logit-Space Prior**: Anchoring fusion in the stronger expert's logit space ($z_{\text{HSI}}$) with bias initialization $g_{\text{init}} \approx 0.12$.
  2. **Hard-Masked Correctness Supervision**: Isolating the $\sim 25\%$ of samples where experts disagree in correctness, eliminating gradient noise from agreement samples.
  3. **Registration-Free Local Statistics**: 11-dimensional gate feature representation capturing texture and confidence statistics without assuming spatial alignment.
