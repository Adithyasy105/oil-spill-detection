# Mathematical Foundation of CAFE-Fusion

## 1. Unimodal Prediction Spaces
Let $\mathbf{x}_{\text{SAR}}$ and $\mathbf{x}_{\text{HSI}}$ represent the inputs to the frozen unimodal experts:
- **SAR Expert ($f_{\text{SAR}}$)**: Predicts probability $p_{\text{SAR}} \in [10^{-6}, 1 - 10^{-6}]$, with logit $z_{\text{SAR}} = \ln \left(\frac{p_{\text{SAR}}}{1 - p_{\text{SAR}}}\right)$.
- **HSI Expert ($f_{\text{HSI}}$)**: Predicts logit $z_{\text{HSI}} \in \mathbb{R}$, with probability $p_{\text{HSI}} = \sigma(z_{\text{HSI}}) = \frac{1}{1 + e^{-z_{\text{HSI}}}}$.

## 2. Gate Input Feature Representation
The gate operates on an 11-dimensional feature vector $\mathbf{u} \in \mathbb{R}^{11}$:
$$\mathbf{u} = [p_{\text{SAR}}, z_{\text{SAR}}, p_{\text{HSI}}, z_{\text{HSI}}, |p_{\text{SAR}} - p_{\text{HSI}}|, (p_{\text{SAR}} - p_{\text{HSI}}), c_{\text{HSI}}, H(p_{\text{HSI}}), H(p_{\text{SAR}}), \mu_{\text{SAR}}, \sigma_{\text{SAR}}]$$
where $H(p) = -(p \ln p + (1-p) \ln (1-p))$ is the entropy, $c_{\text{HSI}}$ is the HSI model confidence, and $\mu_{\text{SAR}}, \sigma_{\text{SAR}}$ are local patch statistics.

## 3. Reliability Gate Network
$$\mathbf{h}_1 = \text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1)), \quad \mathbf{W}_1 \in \mathbb{R}^{64 \times 11}$$
$$\mathbf{h}_2 = \text{GELU}(\text{LayerNorm}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)), \quad \mathbf{W}_2 \in \mathbb{R}^{32 \times 64}$$
$$g = \sigma(\mathbf{w}_3^T \mathbf{h}_2 + b_3), \quad \mathbf{w}_3 \in \mathbb{R}^{32}, \; b_3 = -2.0 \implies g_{\text{init}} \approx 0.119$$

## 4. Geometric Logit Mixing
$$z_{\text{fusion}} = (1 - g) \cdot z_{\text{HSI}} + g \cdot z_{\text{SAR}}$$
$$p_{\text{fusion}} = \sigma(z_{\text{fusion}})$$

## 5. Hard Masked Reliability Supervision Loss
Let $c_{\text{SAR}} = \mathbb{I}(\hat{y}_{\text{SAR}} == y)$ and $c_{\text{HSI}} = \mathbb{I}(\hat{y}_{\text{HSI}} == y)$.
The correctness conflict mask is:
$$m_i = c_{\text{SAR}, i} \oplus c_{\text{HSI}, i}$$
The reliability target is:
$$y_{\text{rel}, i} = \begin{cases} 1 & \text{if } c_{\text{SAR}, i} = 1 \wedge c_{\text{HSI}, i} = 0 \\ 0 & \text{if } c_{\text{HSI}, i} = 1 \wedge c_{\text{SAR}, i} = 0 \end{cases}$$
The masked reliability loss is:
$$\mathcal{L}_{\text{rel}} = - \frac{1}{\sum m_i} \sum_{i: m_i = 1} \left[ y_{\text{rel}, i} \ln g_i + (1 - y_{\text{rel}, i}) \ln (1 - g_i) \right]$$

## 6. Total Objective Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(p_{\text{fusion}}, y) + \mathcal{L}_{\text{Dice}}(p_{\text{fusion}}, y) + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \epsilon_{\text{reg}} \cdot \bar{g}$$
where $\lambda_{\text{rel}} = 0.10$ and $\epsilon_{\text{reg}} = 0.01$.
