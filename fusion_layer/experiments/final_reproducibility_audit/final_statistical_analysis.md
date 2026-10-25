# Authoritative Final Statistical Analysis

## 1. Primary Benchmark (n = 6 Independent Folds)
- **HSI Baseline (HMOSDN)**: $F_1 = 0.9261 \pm 0.0307$, $\text{IoU} = 0.8636 \pm 0.0528$
- **SAR Baseline (CBD-Net)**: $F_1 = 0.8358 \pm 0.0429$, $\text{IoU} = 0.7199 \pm 0.0636$
- **CAFE-Fusion v2 (Ours)**: $F_1 = \mathbf{0.9288 \pm 0.0270}$, $\text{IoU} = \mathbf{0.8680 \pm 0.0467}$
- **Oracle Upper Bound**: $F_1 = 0.9501 \pm 0.0270$, $\text{IoU} = 0.9060 \pm 0.0479$

## 2. Paired Statistical Evaluation
- **Mean Paired Difference ($\Delta F_1$)**: `+0.002730`
- **Median Paired Difference**: `+0.002026`
- **Standard Error**: `0.003358`
- **95% Confidence Interval**: `[-0.005903, +0.011363]`
- **Paired Two-Tailed Student's t-test**: $t = 0.8129, \quad p = 0.4532$
- **Two-Sided Wilcoxon Signed-Rank Test**: $W = 5.0, \quad p = 0.3125$
- **Fold Record**: Won 5 / 6 (83.3%), Lost 1 / 6, Tied 0 / 6
- **Oracle Complementary Gap Recovered**: `11.34%`

## 3. Power Analysis & Honest Reporting
With $n=6$ folds, statistical power is naturally constrained. The improvement ($+0.0027$) is consistent across 5/6 folds, but two-tailed hypothesis tests do not achieve $p < 0.05$. We transparently report this without overclaiming statistical significance.
