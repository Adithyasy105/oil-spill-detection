# Numerical Consistency & Audit Report

## 1. Audit Methodology
Every artifact in `fusion_layer/experiments/` was cross-checked against the raw cached feature arrays and the primary execution logs to identify any historical discrepancy, rounding variation, or test-leakage contamination.

## 2. Comprehensive Metric Verification Table
| Source File | Metric | Reported Value | Recomputed Value | Discrepancy Status | Authoritative Source | Notes |
|---|---|:---:|:---:|:---:|:---:|---|
| `summary.json` | Mean CAFE F1 | `0.9288` | `0.9288` | Exact Match | `primary_results/summary.json` | authoritatively verified |
| `summary.json` | Mean HSI F1 | `0.9261` | `0.9261` | Exact Match | `primary_results/summary.json` | authoritatively verified |
| `summary.json` | Mean SAR F1 | `0.8358` | `0.8358` | Exact Match | `primary_results/summary.json` | authoritatively verified |
| `statistical_analysis.md` | Paired t-test p | `0.4532` | `0.4532` | Exact Match | Recomputed Stats | df=5 two-tailed |
| `statistical_analysis.md` | Wilcoxon p | `0.3125` | `0.3125` | Exact Match | Recomputed Stats | two-sided signed-rank |
| `fold3_threshold_sweep.csv` | Fold 3 Post-hoc F1 | `0.9567` | `0.9567` | Diagnostic Only | `fold3_threshold_sweep.csv` | Labeled diagnostic only |

## 3. Discrepancy Findings
- **No silent discrepancies detected**: All metrics in the primary results match the underlying prediction logs across all 18 scenes.
- **Clear boundary established**: Post-hoc threshold sweeps on Fold 3 ($F_1 = 0.9567$) are quarantined to diagnostic reports and never mixed into the cross-validation benchmark table.
