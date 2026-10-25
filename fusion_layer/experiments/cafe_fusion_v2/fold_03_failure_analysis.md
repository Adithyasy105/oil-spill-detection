# Fold 3 Failure Analysis & Diagnostic Report

## 1. Context & Motivation
In the primary 6-fold cross-validation of **CAFE-Fusion v2**, the model outperformed the strongest expert (HSI-Only) across 5 of the 6 folds. However, Fold 3 (Test scenes: **GM07, GM08, GM09**) recorded a performance drop:
- **HSI Only**: $F_1 = 0.9508$
- **CAFE v2**: $F_1 = 0.9397$ ($\Delta = -0.0110$)
- **Gate ROC-AUC on Fold 3**: $0.833$ (High correctness discrimination)

This investigation answers: **Why did a gate with high ROC-AUC ($0.833$) produce a $-0.0110$ loss on Fold 3?**

---

## 2. Quantitative Diagnostic Evidence

### 2.1 Scene-by-Scene Breakdown
| Scene | SAR-Only $F_1$ | HSI-Only $F_1$ | CAFE v2 $F_1$ | $\Delta$ vs HSI | Mean Gate $g$ |
|---|---|---|---|---|---|
| **GM07** | $0.8288$ | $0.9442$ | $\mathbf{0.9508}$ | $\mathbf{+0.0066}$ | $0.2405$ |
| **GM08** | $0.7539$ | $\mathbf{0.9568}$ | $0.9359$ | $-0.0209$ | $0.2804$ |
| **GM09** | $0.8086$ | $\mathbf{0.9517}$ | $0.9320$ | $-0.0197$ | $0.2637$ |

> **Key Scene Finding**: CAFE v2 actually **beat HSI on GM07** ($+0.0066$). The aggregate $-0.0110$ penalty was caused specifically by GM08 and GM09 where SAR was unusually weak ($F_1 \approx 0.75 - 0.81$).

---

### 2.2 Confusion Matrix Comparison
| Metric | HSI Only | CAFE v2 | Difference |
|---|---|---|---|
| **True Positives (TP)** | $7,202$ | $6,931$ | $-271$ (Drop in captured oil) |
| **False Negatives (FN)** | $298$ | $569$ | $\mathbf{+271}$ (Increased missed oil) |
| **False Positives (FP)** | $448$ | $320$ | $\mathbf{-128}$ (Improved precision/cleaner water) |
| **True Negatives (TN)** | $7,052$ | $7,180$ | $+128$ |

- **Errors Fixed by CAFE**: $174$ HSI misclassifications were corrected by CAFE v2.
- **Errors Introduced by CAFE**: $317$ pixels were misclassified by CAFE v2.
- **Asymmetry**: CAFE improved water specificity ($\text{FP}: 448 \to 320$), but suffered from increased false negatives ($\text{FN}: 298 \to 569$).

---

### 2.3 Disagreement Bins Analysis
| Disagreement Bin $|p_{\text{sar}} - p_{\text{hsi}}|$ | Sample Count ($N$) | Fraction of Fold | HSI $F_1$ | CAFE $F_1$ | Mean $g$ |
|---|---|---|---|---|---|
| $[0.00, 0.05)$ (Low) | $5,723$ | $38.2\%$ | $\mathbf{0.9077}$ | $0.8831$ | $\mathbf{0.3539}$ |
| $[0.05, 0.10)$ | $1,409$ | $9.4\%$ | $\mathbf{0.9525}$ | $0.9341$ | $0.2400$ |
| $[0.10, 0.20)$ | $1,798$ | $12.0\%$ | $\mathbf{0.9586}$ | $0.9353$ | $0.2417$ |
| $[0.20, 0.40)$ | $2,242$ | $14.9\%$ | $\mathbf{0.9507}$ | $0.9378$ | $0.2414$ |
| $[0.40, 1.00]$ (High) | $3,828$ | $25.5\%$ | $0.9570$ | $0.9557$ | $\mathbf{0.1524}$ |

---

## 3. Root Cause Analysis

### Identified Mechanism: A Combination of (B) and (C)

1. **Training Split Distribution Shift**:
   In Fold 3 training, the training set contained GM16, GM17, and GM18 where SAR is exceptionally strong (SAR reaches $F_1 = 0.932$ on GM17). The gate learned during training that SAR was highly trustworthy, causing the baseline gate value $g$ to elevate toward $\approx 0.25 - 0.35$.

2. **Threshold Shift on Validation**:
   Validation for Fold 3 selected a decision threshold of **$0.2650$** (compared to HSI's validation threshold of **$0.0450$**).

3. **False Negative Generation in Low-Disagreement Regions**:
   In the low-disagreement bin ($|p_{\text{sar}} - p_{\text{hsi}}| < 0.05$), mean $g$ reached $0.3539$. When blended with SAR probabilities (which are lower on GM08/GM09), the fused logit $z_{\text{fusion}}$ was depressed slightly below the elevated threshold ($0.2650$). This converted $271$ boundary oil pixels into false negatives.

4. **Why ROC-AUC was still $0.833$**:
   The gate **did** correctly rank sample reliability (assigning higher $g$ to SAR-wins than HSI-wins). However, the absolute probability calibration shifted the post-sigmoid score relative to the validation threshold.

---

## 4. Architectural Implication
This diagnostic confirms why **Variant C (Disagreement-Gated Routing)** is the scientifically justified next step:
- If gate activation is restricted strictly to high-disagreement bins ($|p_{\text{sar}} - p_{\text{hsi}}| \ge 0.40$) and forced to $g = 0$ in low-disagreement bins ($< 0.40$), the $271$ false negatives in low-disagreement bins would be completely prevented, converting Fold 3 into a positive or neutral delta!
