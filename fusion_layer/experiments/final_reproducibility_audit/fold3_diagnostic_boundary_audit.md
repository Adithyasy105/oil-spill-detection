# Fold 3 Diagnostic Boundary Audit

## 1. Strict Scientific Separation
To guarantee absolute scientific integrity and zero test-set leakage, all evaluations are partitioned into two strict boundaries:

### Boundary A: Primary Unbiased Cross-Validation (Reported in Main Tables)
- **Model Checkpoints**: Selected solely on peak validation F1 across 100 training epochs.
- **Decision Thresholds**: Selected solely by sweeping $t \in [0.01, 0.99]$ on the 3 validation scenes.
- **Test Set**: Evaluated exactly ONCE with frozen checkpoint and frozen threshold.
- **Fold 3 Result**: HSI $= 0.9508$, CAFE v2 $= 0.9397$ ($\Delta = -0.0110$).

### Boundary B: Post-Hoc Diagnostic Analysis (Quarantined to Failure Analysis)
- **Objective**: Determine whether the $-0.0110$ delta was caused by gate failure or threshold calibration shift.
- **Finding**: Sweeping thresholds post-hoc reveals CAFE achieves $F_1 = 0.9567$ at $t=0.0500$.
- **Boundary Rule**: This $0.9567$ number is explicitly quarantined as a diagnostic artifact and **never** used to claim a higher average benchmark F1 score.
