# Research Evolution Timeline: From Degradation to Calibrated Gain

```
[Phase 1: Independent Modality Experts]
HSI Only: 0.9261 (Dominant) | SAR Only: 0.8358 (Look-alikes)
                       ↓
[Phase 2: Naive Multimodal Fusion Failure]
Legacy Fusion: 0.9172 (Degradation: -0.0089 vs HSI)
                       ↓
[Phase 3: Calibration & Scaling Attempts]
Platt Scaling: 0.9205 (Static scaling cannot resolve dynamic conflict)
                       ↓
[Phase 4: Reliability-Aware Gating Paradigm]
Formulation: Predict gate g based on modality disagreement and quality statistics
                       ↓
[Phase 5: Hard Disagreement Router (Variant C)]
g_eff = 0 in low-disagreement (Exact identity verified: error < 1e-9)
F1: 0.9253 (Hard step boundaries cause threshold instability)
                       ↓
[Phase 6: Logit-Space CAFE-Fusion v2]
Smooth geometric mixing: (1-g)*z_HSI + g*z_SAR with masked reliability loss
                       ↓
[Phase 7: Final Primary Benchmark (Frozen)]
CAFE v2: 0.9288 (+0.0027 vs HSI, beats HSI on 5/6 folds, recovers 11.34% Oracle Gap)
```
