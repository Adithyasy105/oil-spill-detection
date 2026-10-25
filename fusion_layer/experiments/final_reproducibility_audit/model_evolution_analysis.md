# Complete Model Evolution Analysis

## 1. Summary of Architectural Trajectory
Across the project's development cycle, 7 distinct modeling paradigms were implemented and benchmarked:

1. **Unimodal Baselines**: Established that HSI ($0.9261$) is the dominant expert while SAR ($0.8358$) is complementary but noisy.
2. **Legacy Fusion ($0.9172$)**: Naive additive fusion corrupted HSI predictions with SAR noise, degrading performance by $-0.0089$ below HSI.
3. **Platt Calibration ($0.9205$)**: Addressed probability scaling but failed to provide sample-specific routing.
4. **CAFE Variant C ($0.9253$)**: Enforced exact HSI reproduction in low disagreement ($< 10^{-9}$ error), proving the identity theorem but suffering from threshold boundary discontinuities.
5. **CAFE-Fusion v2 ($0.9288$)**: Combined logit-space geometric mixing, asymmetric HSI bias initialization, and masked reliability supervision, consistently beating HSI on 5/6 folds (+0.0027) and recovering 11.34% of the Oracle complementary gap.
