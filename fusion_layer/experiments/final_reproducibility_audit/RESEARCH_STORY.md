# The Research Story of CAFE-Fusion: Overcoming Multimodal Degradation in Marine Oil Spill Detection

## 1. Introduction & Problem Context
Marine oil spill detection using satellite remote sensing has historically relied on two distinct modalities: Synthetic Aperture Radar (SAR), which provides day-and-night, all-weather coverage of ocean surface roughness, and Hyperspectral Imaging (HSI), which captures fine diagnostic hydrocarbon absorption bands. While both modalities provide unique physical insights, their spatial resolutions, coordinate systems, and acquisition times differ substantially, making rigorous pixel-level co-registration impossible in operational maritime monitoring.

## 2. The Dominant Expert Dilemma
Our initial unimodal benchmarks revealed a fundamental asymmetry:
- **Hyperspectral (HMOSDN)**: $F_1 = 0.9261 \pm 0.0307$ (High global accuracy)
- **SAR (CBD-Net)**: $F_1 = 0.8358 \pm 0.0429$ (Noisy due to look-alike biogenic slicks and wind shadows)

When early multimodal fusion architectures were evaluated, they resulted in catastrophic degradation: **Legacy Fusion achieved $F_1 = 0.9172$**, performing worse than HSI alone ($-0.0089$). The weaker modality's false positives directly contaminated the predictions of the stronger expert.

## 3. The Research Iteration Cycle
1. **Calibration Attempts**: Platt probability scaling slightly improved score calibration ($0.9205$) but failed to resolve spatial conflicts dynamically.
2. **Hard Disagreement Router (Variant C)**: Enforcing $g_{\text{effective}} = 0$ in low-disagreement samples guaranteed exact HSI identity reproduction (error $< 10^{-9}$), but hard step thresholding created boundary artifacts ($0.9253$).
3. **CAFE-Fusion v2 Formulation**: Transitioning to logit-space geometric mixing, initializing gate bias to $-2.0$ ($g_{\text{init}} \approx 0.12$), and supervising the gate with hard-masked correctness conflicts produced **$F_1 = 0.9288 \pm 0.0270$**, beating HSI on **5 out of 6 cross-validation folds** and recovering **$11.34\%$** of the Oracle gap.

## 4. Honest Scientific Limitations
- With $n=6$ folds, paired hypothesis testing yields $p = 0.4532$ (t-test) and $p = 0.3125$ (Wilcoxon). While the improvement is consistent across 83.3% of folds, statistical significance at $\alpha = 0.05$ cannot be claimed under the small sample size.
- In Fold 3, validation threshold shift caused a $-0.0110$ drop on GM08/GM09 despite high gate ranking accuracy (ROC-AUC $= 0.833$).
