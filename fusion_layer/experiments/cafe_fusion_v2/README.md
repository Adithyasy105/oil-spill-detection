# CAFE Fusion v2 — Confidence-Aware Fusion Engine

## Abstract Architecture
CAFE (Confidence-Aware Fusion Engine) is a registration-free, confidence-aware fusion architecture designed for paired SAR and Hyperspectral (HSI) oil spill imagery. 

Because HSI is the stronger expert (F1 = 0.9261 ± 0.0307) and SAR is complementary but weaker (F1 = 0.8358 ± 0.0429), CAFE anchors the fusion in HSI logit space and learns a tiny, regularized reliability gate $g \in [0, 1]$ to dynamically route to SAR only when SAR is reliably superior.

```
Frozen SAR Expert  (CBD-Net)
      ↓
  p_sar + SAR statistics
                         \
                          → Tiny Reliability Gate MLP → g ∈ [0, 1]
                         /
  p_hsi + HSI statistics
      ↑
Frozen HSI Expert (HMOSDN)
```

## Mathematical Formulation
1. **Fusion in Logit Space:**
   $$z_{\text{HSI}} = \text{logit}_{\text{HSI}}$$
   $$z_{\text{SAR}} = \text{logit}(p_{\text{SAR}})$$
   $$z_{\text{fusion}} = (1 - g) \cdot z_{\text{HSI}} + g \cdot z_{\text{SAR}}$$
   $$p_{\text{fusion}} = \sigma(z_{\text{fusion}})$$

2. **Gate Initialization (HSI Protection):**
   $$\text{bias} = -2.0 \implies g \approx 0.119 \quad (\text{Default pure HSI behavior})$$

3. **Masked Reliability Supervision:**
   Supervision is applied **only on samples where experts disagree in correctness**:
   $$m_i = (\text{SAR\_correct}_i \oplus \text{HSI\_correct}_i)$$
   $$y^{\text{rel}}_i = \begin{cases} 1 & \text{if SAR is correct and HSI is wrong} \\ 0 & \text{if HSI is correct and SAR is wrong} \end{cases}$$
   $$\mathcal{L}_{\text{rel}} = \frac{1}{\sum m_i} \sum_{i: m_i=1} \text{BCE}(g_i, y^{\text{rel}}_i)$$

4. **Total Loss:**
   $$\mathcal{L} = \mathcal{L}_{\text{seg}}(p_{\text{fusion}}, y) + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \epsilon_{\text{HSI}} \cdot \bar{g}$$
   where $\epsilon_{\text{HSI}} = 0.01$ pulls $g \to 0$ when no clear SAR signal exists.

## Inputs (Registration-Free Only)
11 scalar features per sample:
1. `p_sar`
2. `z_sar = logit(p_sar)`
3. `p_hsi`
4. `z_hsi`
5. `|p_sar - p_hsi|` (Disagreement magnitude)
6. `p_sar - p_hsi` (Signed difference)
7. `hsi_confidence` (MC Dropout confidence)
8. `H(p_hsi)` (HSI prediction entropy)
9. `H(p_sar)` (SAR prediction entropy)
10. `sar_local_mean`
11. `sar_local_std`
