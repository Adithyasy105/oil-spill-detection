# Confidence-Aware Feature Ensemble (CAFE-Fusion) for Multimodal Marine Oil Spill Detection

## Abstract
Multimodal fusion of Synthetic Aperture Radar (SAR) and Hyperspectral Imagery (HSI) offers the potential to combine wide-area structural ocean monitoring with precise spectral hydrocarbon identification. However, naive multimodal fusion frequently suffers from negative transfer, where false alarms from the weaker modality degrade the performance of the dominant expert. In this paper, we propose **CAFE-Fusion v2 (Confidence-Aware Feature Ensemble)**, a registration-free, logit-space fusion architecture that dynamically gates modality influence while anchoring predictions in the stronger expert. Trained with a hard-masked correctness reliability loss that supervises the gate exclusively on conflict regions, CAFE-Fusion achieves a mean $F_1$ score of $\mathbf{0.9288 \pm 0.0270}$ and $\text{IoU}$ of $\mathbf{0.8679 \pm 0.0465}$ across an 18-scene, 6-fold cross-validation benchmark, outperforming the unimodal HSI baseline ($F_1 = 0.9261$) on 5 of 6 folds and recovering $11.34\%$ of the Oracle complementary gap without requiring pixel-level spatial registration.

**Keywords**: Multimodal Fusion, Oil Spill Detection, Hyperspectral Imaging, Synthetic Aperture Radar, Reliability Gating.

---

## 1. Main Benchmark Results (6-Fold Cross-Validation)
| Model | Mean $F_1 \pm \text{Std}$ | Mean $\text{IoU} \pm \text{Std}$ | $\Delta F_1$ vs HSI | Folds Won | Oracle Gap Recovery |
|---|:---:|:---:|:---:|:---:|:---:|
| **SAR Only (CBD-Net)** | $0.8358 \pm 0.0429$ | $0.7199 \pm 0.0636$ | $-0.0903$ | $0 / 6$ | — |
| **Legacy Registration-Free Fusion** | $0.9172 \pm 0.0286$ | $0.8490 \pm 0.0435$ | $-0.0089$ | $1 / 6$ | Negative |
| **CAFE Variant C (Hard Router)** | $0.9253 \pm 0.0247$ | $0.8618 \pm 0.0428$ | $-0.0007$ | $4 / 6$ | $-3.07\%$ |
| **HSI Only Baseline (HMOSDN)** | $0.9261 \pm 0.0307$ | $0.8636 \pm 0.0528$ | Baseline | — | — |
| **CAFE-Fusion v2 (Ours)** | $\mathbf{0.9288 \pm 0.0270}$ | $\mathbf{0.8679 \pm 0.0465}$ | $\mathbf{+0.0027}$ | $\mathbf{5 / 6}$ | $\mathbf{+11.34\%}$ |
| *Oracle Bound (Theoretical Ceiling)* | $0.9501 \pm 0.0270$ | $0.9060 \pm 0.0479$ | $+0.0240$ | $6 / 6$ | $100.0\%$ |

---

## 2. Statistical Analysis & Defensible Reporting
- **Mean Paired Difference**: `+0.002730`
- **Median Paired Difference**: `+0.002026`
- **95% Confidence Interval**: `[-0.005903, +0.011363]`
- **Paired Two-Tailed Student's t-test**: $t = 0.8129, \; p = 0.4532$
- **Wilcoxon Signed-Rank Test**: $W = 5.0, \; p = 0.3125$
- **Scientific Claim**: CAFE-Fusion v2 demonstrates consistent empirical improvements (beating HSI on 5 of 6 folds and outperforming previous fusion by $+0.0116$), but due to $n=6$ folds, the difference is not statistically significant at $\alpha = 0.05$.
