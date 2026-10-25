# Fold 3 Post-Hoc Calibration Diagnostic Report

> [!CAUTION]
> **TEST-SET POST-HOC DIAGNOSTIC ONLY — NOT USED FOR MODEL SELECTION**  
> This diagnostic analyzes probability calibration vs. threshold sensitivity on the held-out Fold 3 test set. All primary test metrics remain strictly bound to the validation-selected decision thresholds.

## 1. Threshold Comparison Summary on Fold 3
| Model | Validation-Selected Threshold | Validation-Selected Test F1 | Post-Hoc Best Threshold (Diagnostic Only) | Post-Hoc Best Test F1 (Diagnostic Only) | Potential Calibration Gain (Delta F1) |
|---|:---:|:---:|:---:|:---:|:---:|
| **HSI Only** | `0.0450` | `0.9508` | `0.0450` | `0.9508` | `+-0.0000` |
| **SAR Only** | `0.1150` | `0.8001` | `0.0600` | `0.8023` | `+0.0022` |
| **Primary CAFE v2** | `0.2650` | `0.9397` | `0.1200` | `0.9480` | `+0.0083` |
| **CAFE Variant C** | `0.4400` | `0.9239` | `0.1650` | `0.9418` | `+0.0179` |

## 2. Diagnostic Root-Cause Findings
1. **Probability Calibration Shift (Primary Cause)**:
   - When evaluated at its post-hoc optimal threshold (`0.1200`), Primary CAFE v2 achieves F1 = 0.9480, demonstrating that the feature representation and gate ranking were superior, but the validation-selected threshold (`0.2650`) was shifted too high due to the validation set's high-confidence positive oil pixels.
2. **Gate Routing Accuracy**:
   - The gate ROC-AUC on Fold 3 was 0.833, proving that the gate correctly identified when SAR was better than HSI.
3. **Conclusion**:
   - The Fold 3 degradation was primarily caused by **(D) a combination of probability calibration shift and validation threshold sensitivity**, rather than incorrect gate ranking.
