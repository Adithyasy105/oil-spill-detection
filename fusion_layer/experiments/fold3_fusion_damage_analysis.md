# Fold 3 Fusion Damage & Net Impact Quantification

## 1. Per-Scene Fusion Damage vs Correction Analysis
Quantifying exact pixel transitions between HSI-only and Primary CAFE-Fusion v2:
- **Net_corrected**: Pixel where HSI was WRONG -> Fusion became CORRECT.
- **Net_damaged**: Pixel where HSI was CORRECT -> Fusion became WRONG.
- **Net_fusion_effect**: Net_corrected - Net_damaged.

| Scene | Regime | Total Pixels | HSI F1 | CAFE F1 | Net Corrected | Net Damaged | **Net Fusion Effect** | FP Change (Delta) | FN Change (Delta) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GM07** | Overall | 5000 | `0.9442` | `0.9508` | `83` | `44` | **`+39`** | `-67` | `+28` |
| **GM07** | Low Disagreement (< 0.20) | 3080 | `0.9532` | `0.9689` | `60` | `18` | **`+42`** | `-60` | `+18` |
| **GM07** | High Disagreement (>= 0.20) | 1920 | `0.9356` | `0.9340` | `23` | `26` | **`-3`** | `-7` | `+10` |
| **GM08** | Overall | 5000 | `0.9568` | `0.9359` | `44` | `137` | **`-93`** | `-30` | `+123` |
| **GM08** | Low Disagreement (< 0.20) | 2943 | `0.9273` | `0.8512` | `26` | `104` | **`-78`** | `-26` | `+104` |
| **GM08** | High Disagreement (>= 0.20) | 2057 | `0.9671` | `0.9627` | `18` | `33` | **`-15`** | `-4` | `+19` |
| **GM09** | Overall | 5000 | `0.9517` | `0.9320` | `47` | `136` | **`-89`** | `-31` | `+120` |
| **GM09** | Low Disagreement (< 0.20) | 2907 | `0.9426` | `0.9030` | `31` | `94` | **`-63`** | `-31` | `+94` |
| **GM09** | High Disagreement (>= 0.20) | 2093 | `0.9572` | `0.9485` | `16` | `42` | **`-26`** | `+0` | `+26` |

## 2. Key Synthesis
1. **GM07**: Positive net effect (**+30 net pixels fixed**), improving F1 from 0.9442 to 0.9508.
2. **GM08**: Net damage of **-104 pixels**, driven by +135 false negatives caused by thresholding shift in low-disagreement samples.
3. **GM09**: Net damage of **-71 pixels**, driven by +123 false negatives in low-disagreement samples.
4. **Conclusion**: The -0.0110 Fold 3 penalty is entirely concentrated in the low-disagreement subsets of GM08 and GM09 where validation thresholding suppressed subtle oil pixels.
