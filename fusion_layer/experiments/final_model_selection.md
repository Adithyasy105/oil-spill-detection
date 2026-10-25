# Final Model Selection Report

## 1. Objective Comparison: Ablation B (lambda=0.05) vs Primary CAFE v2 (lambda=0.10)

Using the exact six paired fold F1 scores produced during the frozen cross-validation:

| Fold | Held-out Scenes | Primary CAFE v2 (lambda=0.10) | Ablation B (lambda=0.05) | Paired Difference (Delta_i) | Winner |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | GM01, GM02, GM03 | **0.955573** | 0.955447 | -0.000126 | Primary (lambda=0.10) |
| **Fold 2** | GM04, GM05, GM06 | **0.954407** | 0.953838 | -0.000569 | Primary (lambda=0.10) |
| **Fold 3** | GM07, GM08, GM09 | 0.939733 | **0.940244** | +0.000511 | Ablation B (lambda=0.05) |
| **Fold 4** | GM10, GM11, GM12 | **0.894782** | 0.894714 | -0.000067 | Primary (lambda=0.10) |
| **Fold 5** | GM13, GM14, GM15 | 0.931023 | **0.933423** | +0.002400 | Ablation B (lambda=0.05) |
| **Fold 6** | GM16, GM17, GM18 | 0.897239 | **0.897300** | +0.000062 | Ablation B (lambda=0.05) |

## 2. Statistical Analysis
- **Mean Paired Difference**: `+0.000369`
- **Median Paired Difference**: `-0.000003`
- **Standard Deviation**: `0.001054`
- **Standard Error**: `0.000430`
- **95% Confidence Interval**: `[-0.000738, +0.001475]` (straddles zero)
- **Paired Two-Tailed Student's t-test**: t = 0.8566, p = 0.4308 (p >> 0.05)
- **Two-Sided Wilcoxon Signed-Rank Test**: W = 10.0, p = 1.0000 (p = 1.0000)
- **Fold Record**: Won 3, Lost 3, Tied 0 (Split 3 vs 3)

## 3. Decision & Scientific Justification
Following the selection protocol:
1. Ablation B (lambda=0.05) wins on 3 folds while Primary (lambda=0.10) wins on 3 folds (exact tie).
2. The median difference is -0.000003 (favoring lambda=0.10), and the two-tailed paired t-test (p=0.4308) and Wilcoxon test (p=1.0000) show no statistically or practically meaningful difference.
3. **Decision**: **Retain Primary CAFE-Fusion v2 (lambda_rel = 0.10) as the primary frozen proposed method**, reporting lambda=0.05 as a robust, consistent configuration confirming low sensitivity to the regularization coefficient.
