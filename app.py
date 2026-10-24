import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import json
import matplotlib.pyplot as plt
from skimage import morphology
from PIL import Image
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, roc_auc_score

# =============================================================================
# Path Configuration & Module Imports
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAR_LAYER_DIR = os.path.join(PROJECT_ROOT, "sar_layer")
HSI_LAYER_DIR = os.path.join(PROJECT_ROOT, "hsi_layer_corrected")
FUSION_EXP_DIR = os.path.join(PROJECT_ROOT, "fusion_layer", "experiments")
CACHE_ROOT = os.path.join(PROJECT_ROOT, "fusion_layer", "cache_features")

if SAR_LAYER_DIR not in sys.path:
    sys.path.insert(0, SAR_LAYER_DIR)
if HSI_LAYER_DIR not in sys.path:
    sys.path.insert(0, HSI_LAYER_DIR)

# 1. SAR Module Loading (CBDNetSAR)
try:
    import importlib.util
    spec_sar_m = importlib.util.spec_from_file_location("model_03_sar", os.path.join(SAR_LAYER_DIR, "03_model.py"))
    m3_sar = importlib.util.module_from_spec(spec_sar_m)
    spec_sar_m.loader.exec_module(m3_sar)
    CBDNetSAR = m3_sar.CBDNetSAR
    
    spec_sar_ds = importlib.util.spec_from_file_location("dataset_02_sar", os.path.join(SAR_LAYER_DIR, "02_dataset.py"))
    ds2_sar = importlib.util.module_from_spec(spec_sar_ds)
    spec_sar_ds.loader.exec_module(ds2_sar)
    preprocess_sar = ds2_sar.preprocess_sar
except Exception as e:
    CBDNetSAR = None
    preprocess_sar = None

# 2. HSI Module Loading (HMOSDNet from hsi_layer_corrected)
try:
    spec_hsi_m = importlib.util.spec_from_file_location("model_03_hsi", os.path.join(HSI_LAYER_DIR, "03_model.py"))
    m3_hsi = importlib.util.module_from_spec(spec_hsi_m)
    spec_hsi_m.loader.exec_module(m3_hsi)
    HMOSDNet = m3_hsi.HMOSDNet
    
    spec_hsi_ds = importlib.util.spec_from_file_location("dataset_02_hsi", os.path.join(HSI_LAYER_DIR, "02_dataset.py"))
    ds2_hsi = importlib.util.module_from_spec(spec_hsi_ds)
    spec_hsi_ds.loader.exec_module(ds2_hsi)
    SpectralProcessor = ds2_hsi.SpectralProcessor
    BANDS_TO_REMOVE_HSI = ds2_hsi.BANDS_TO_REMOVE
    PCA_COMPONENTS_HSI = 25
    PATCH_SIZE_HSI = 5
    import scipy.io as sio
except Exception as e:
    HMOSDNet = None
    SpectralProcessor = None
    BANDS_TO_REMOVE_HSI = []
    PCA_COMPONENTS_HSI = 25
    PATCH_SIZE_HSI = 5
    import scipy.io as sio

# =============================================================================
# App Config & Premium UI Styling (CAFE-Fusion Intelligence Platform)
# =============================================================================
st.set_page_config(
    page_title="Confidence-Aware Multimodal Fusion | Marine Oil Spill Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b132b 0%, #1c2541 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b132b !important;
        border-right: 1px solid #1e293b;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-badge-critical {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #f87171;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .status-badge-nominal {
        background: rgba(45, 212, 191, 0.15);
        border: 1px solid #2dd4bf;
        color: #2dd4bf;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .stDataFrame {
        background-color: #1e293b;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Model Architectures & Checkpoint Loaders
# =============================================================================

@st.cache_resource
def load_sar_expert():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = os.path.join(SAR_LAYER_DIR, "weights", "best_sar_expert.pth")
    if not os.path.exists(weights_path):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        url = "https://huggingface.co/adi106/marine-oil-spill-weights/resolve/main/best_sar_expert.pth"
        try:
            with st.spinner("Downloading SAR expert weights (~165MB)..."):
                torch.hub.download_url_to_file(url, weights_path)
        except Exception as e:
            st.sidebar.error(f"Failed to auto-download SAR weights: {e}")
    if CBDNetSAR is None or not os.path.exists(weights_path):
        return None, device
    try:
        model = CBDNetSAR().to(device)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd)
        model.eval()
        return model, device
    except Exception as e:
        st.sidebar.error(f"⚠️ SAR Weight Load Error: {e}")
        return None, device


@st.cache_resource
def load_hsi_expert():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_candidates = [
        os.path.join(HSI_LAYER_DIR, "weights", "frozen_hsi_expert.pth"),
        os.path.join(HSI_LAYER_DIR, "weights", "best_hsi_expert.pth"),
    ]
    weights_path = next((p for p in weights_candidates if os.path.exists(p)), weights_candidates[0])
    pca_path = os.path.join(HSI_LAYER_DIR, "weights", "spectral_processor.pkl")
    
    processor = None
    if SpectralProcessor is not None and os.path.exists(pca_path):
        try:
            processor = SpectralProcessor(n_components=PCA_COMPONENTS_HSI, bands_to_remove=BANDS_TO_REMOVE_HSI)
            processor.load(pca_path)
        except Exception:
            processor = None

    if HMOSDNet is None or not os.path.exists(weights_path):
        return None, processor, device

    try:
        model = HMOSDNet(in_channels=1, num_bands=PCA_COMPONENTS_HSI, mam_reduction=2).to(device)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd)
        model.eval()
        return model, processor, device
    except Exception as e:
        st.sidebar.error(f"⚠️ HSI Weight Load Error: {e}")
        return None, processor, device


class TinyReliabilityGate(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class CAFEFusionModel(nn.Module):
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, fusion_mode: str = "logit"):
        super().__init__()
        self.gate = TinyReliabilityGate(input_dim=input_dim, hidden_dim=hidden_dim)
        self.fusion_mode = fusion_mode

    def forward(self, gate_inputs: torch.Tensor, z_sar: torch.Tensor, z_hsi: torch.Tensor):
        g = self.gate(gate_inputs)
        z_fused = (1.0 - g) * z_hsi + g * z_sar
        p_fused = torch.sigmoid(z_fused)
        return p_fused, g, z_fused


def enable_mc_dropout(model: nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@st.cache_resource
def load_cafe_fusion_model(fold: int = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = os.path.join(
        FUSION_EXP_DIR, "cafe_fusion_v2", "primary_results",
        f"fold_{fold:02d}", "lambda_0.10", "best_cafe_model.pth"
    )
    model = CAFEFusionModel(input_dim=11, hidden_dim=64).to(device)
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(sd)
            model.eval()
            return model, device, True
        except Exception as e:
            return model, device, False
    return model, device, False


# Preprocessing Helpers
def enhance_realistic_sar(data):
    image = data.astype(np.float32)
    image_db = 10.0 * np.log10(np.abs(image) + 1e-6)
    v_min, v_max = np.percentile(image_db, 2), np.percentile(image_db, 98)
    real_view = np.clip(255 * (image_db - v_min) / (v_max - v_min + 1e-6), 0, 255).astype(np.uint8)
    return real_view, image_db


def preprocess_for_ai(image_raw):
    if preprocess_sar is not None:
        image_norm = preprocess_sar(image_raw)
    else:
        v_min, v_max = np.percentile(image_raw, 1), np.percentile(image_raw, 99)
        image_norm = np.clip((image_raw - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)
    return np.stack([image_norm, image_norm, image_norm], axis=0)


def build_gate_features_live(p_sar_flat, p_hsi_flat, hsi_conf_flat=None, sar_mean_flat=None, sar_std_flat=None):
    p_sar = np.clip(p_sar_flat, 1e-6, 1.0 - 1e-6).astype(np.float32)
    p_hsi = np.clip(p_hsi_flat, 1e-6, 1.0 - 1e-6).astype(np.float32)
    z_sar = np.log(p_sar / (1.0 - p_sar))
    z_hsi = np.log(p_hsi / (1.0 - p_hsi))
    abs_diff = np.abs(p_sar - p_hsi)
    signed_diff = p_sar - p_hsi
    hsi_conf = hsi_conf_flat.astype(np.float32) if hsi_conf_flat is not None else p_hsi
    hsi_ent = -(p_hsi * np.log(p_hsi + 1e-12) + (1.0 - p_hsi) * np.log(1.0 - p_hsi + 1e-12))
    sar_ent = -(p_sar * np.log(p_sar + 1e-12) + (1.0 - p_sar) * np.log(1.0 - p_sar + 1e-12))
    s_mean = sar_mean_flat.astype(np.float32) if sar_mean_flat is not None else np.full_like(p_sar, np.mean(p_sar))
    s_std = sar_std_flat.astype(np.float32) if sar_std_flat is not None else np.full_like(p_sar, np.std(p_sar))
    
    return np.column_stack([
        p_sar, z_sar, p_hsi, z_hsi, abs_diff, signed_diff,
        hsi_conf, hsi_ent, sar_ent, s_mean, s_std
    ]).astype(np.float32)


# Visualization Tile Renderer Helper
def render_tile(arr, title, cmap='viridis', vmin=0.0, vmax=1.0):
    if arr.ndim == 1:
        side = int(np.sqrt(len(arr)))
        if side * side == len(arr):
            arr_2d = arr.reshape(side, side)
        else:
            arr_2d = arr[:side * side].reshape(side, side)
    else:
        arr_2d = arr

    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    im = ax.imshow(arr_2d, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title, color='#f8fafc', fontsize=9, fontweight='bold', pad=4)
    fig.tight_layout(pad=0.2)
    return fig


# Cached Scene Data Loader
@st.cache_data
def load_all_cached_scenes():
    scenes = {}
    scene_names_by_split = {
        "train": ["GM01", "GM02", "GM06", "GM07", "GM08", "GM12", "GM13", "GM14", "GM18"],
        "validation": ["GM03", "GM09", "GM15"],
        "test": ["GM04", "GM05", "GM10", "GM11", "GM16", "GM17"],
    }
    for split, s_names in scene_names_by_split.items():
        npz_path = os.path.join(CACHE_ROOT, f"{split}_features.npz")
        if os.path.exists(npz_path):
            with np.load(npz_path, allow_pickle=False) as d:
                for idx, sname in enumerate(s_names):
                    mask = d["scene_indices"] == idx
                    scenes[sname] = {
                        "split": split,
                        "sar_features": d["sar_features"][mask],
                        "hsi_logits": d["hsi_logits"][mask],
                        "hsi_confidence": d["hsi_confidence"][mask],
                        "labels": d["labels"][mask],
                    }
    return scenes


@st.cache_data
def load_cafe_summary():
    summary_path = os.path.join(FUSION_EXP_DIR, "cafe_fusion_v2", "primary_results", "summary.json")
    if not os.path.exists(summary_path):
        return {
            "mean_f1_fusion": None,
            "std_f1_fusion": None,
            "mean_f1_hsi": None,
            "std_f1_hsi": None,
            "mean_f1_sar": None,
            "std_f1_sar": None,
            "oracle_gap_recovered_pct": None,
            "paired_t_p_value": None,
            "wilcoxon_p_value": None,
        }
    with open(summary_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# App Header & Status Center
# =============================================================================
model_sar, device_sar = load_sar_expert()
model_hsi, processor_hsi, device_hsi = load_hsi_expert()
model_cafe, device_cafe, cafe_loaded = load_cafe_fusion_model(fold=1)

cached_scenes = load_all_cached_scenes()

st.title("Confidence-Aware Multimodal Fusion for Oil Spill Detection")
st.markdown("<p style='color:#94a3b8; font-size:14px; margin-top:-10px;'>CAFE-Fusion v2 Platform | SAR (CBD-Net) + HSI (HMOSDNet) + Logit Gating + MC Dropout Epistemic Uncertainty</p>", unsafe_allow_html=True)
st.divider()

# Sidebar
st.sidebar.markdown("### SYSTEM STATUS & CONTROLS")

if model_sar is not None:
    st.sidebar.markdown("<div style='background:rgba(45,212,191,0.1); border:1px solid #2dd4bf; color:#2dd4bf; padding:8px 12px; border-radius:6px; font-size:12px; font-weight:600;'>CBD-Net SAR Expert: ONLINE</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='background:rgba(239,68,68,0.1); border:1px solid #ef4444; color:#f87171; padding:8px 12px; border-radius:6px; font-size:12px; font-weight:600;'>SAR Expert: OFFLINE</div>", unsafe_allow_html=True)

if model_hsi is not None:
    st.sidebar.markdown("<div style='background:rgba(45,212,191,0.1); border:1px solid #2dd4bf; color:#2dd4bf; padding:8px 12px; border-radius:6px; font-size:12px; font-weight:600; margin-top:8px;'>HMOSDNet HSI Expert: ONLINE</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='background:rgba(239,68,68,0.1); border:1px solid #ef4444; color:#f87171; padding:8px 12px; border-radius:6px; font-size:12px; font-weight:600; margin-top:8px;'>HSI Expert: OFFLINE</div>", unsafe_allow_html=True)

if cafe_loaded:
    st.sidebar.markdown("<div style='background:rgba(168,85,247,0.1); border:1px solid #a855f7; color:#c084fc; padding:8px 12px; border-radius:6px; font-size:12px; font-weight:600; margin-top:8px;'>CAFE-Fusion v2: ONLINE (MC Dropout)</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div style='background:rgba(239,68,68,0.1); border:1px solid #ef4444; color:#f87171; padding:8px 12px; border-radius:6px; font-size:12px; font-weight:600; margin-top:8px;'>CAFE-Fusion: OFFLINE</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### SAR Expert Parameters")
threshold_sar = st.sidebar.slider("SAR Sensitivity Threshold", 0.1, 0.95, 0.75, 0.05, key="sar_t")
min_size_sar = st.sidebar.slider("Minimum Cluster Area (px²)", 10, 500, 50, 10, key="sar_m")

st.sidebar.markdown("#### HSI Expert Parameters")
threshold_hsi = st.sidebar.slider("HSI Sensitivity Threshold", 0.1, 0.95, 0.50, 0.05, key="hsi_t")

st.sidebar.markdown("#### Fusion Engine Parameters")
fusion_threshold = st.sidebar.slider("Decision Threshold (t*)", 0.01, 0.99, 0.50, 0.01, key="fus_t")
mc_passes = st.sidebar.slider("MC Dropout Stochastic Passes (T)", 5, 50, 30, 5, key="mc_t")
selected_fold = st.sidebar.selectbox("Cross-Validation Checkpoint Fold", [1, 2, 3, 4, 5, 6], index=0)

if selected_fold != 1:
    model_cafe, device_cafe, cafe_loaded = load_cafe_fusion_model(fold=selected_fold)


# =============================================================================
# Navigation Tabs
# =============================================================================
tab_fusion, tab_sar, tab_hsi = st.tabs([
    "CAFE-Fusion Intelligence & Multimodal Comparison",
    "SAR Intelligence (CBD-Net)",
    "HSI Spectral Expert (HMOSDNet)"
])


# =============================================================================
# TAB 1: CAFE-FUSION INTELLIGENCE & MULTIMODAL COMPARISON
# =============================================================================
with tab_fusion:
    benchmark_summary = load_cafe_summary()
    st.markdown("""
    <div style='background:linear-gradient(90deg,rgba(168,85,247,0.15),rgba(56,189,248,0.10));
                border:1px solid #a855f7; border-radius:12px; padding:16px 22px; margin-bottom:20px;'>
        <h3 style='color:#c084fc; margin:0 0 6px 0; font-size:18px;'>CAFE-Fusion v2 Multimodal Decision & Epistemic Uncertainty Hub</h3>
        <p style='color:#94a3b8; margin:0; font-size:13px;'>Convex Logit-Space Mixing: <code>z_fusion = (1 - g) · z_HSI + g · z_SAR</code> | Epistemic Uncertainty via Monte Carlo Dropout (T={t})</p>
    </div>
    """.format(t=mc_passes), unsafe_allow_html=True)

    if benchmark_summary.get("mean_f1_fusion") is not None:
        summary_rows = [
            ("SAR-only expert", benchmark_summary.get("mean_f1_sar", 0.0), benchmark_summary.get("std_f1_sar", 0.0), "#38bdf8"),
            ("HSI-only expert", benchmark_summary.get("mean_f1_hsi", 0.0), benchmark_summary.get("std_f1_hsi", 0.0), "#2dd4bf"),
            ("CAFE fusion v2", benchmark_summary.get("mean_f1_fusion", 0.0), benchmark_summary.get("std_f1_fusion", 0.0), "#c084fc"),
            ("Oracle upper bound", 0.9501, 0.0270, "#f59e0b"),
        ]
        benchmark_df = pd.DataFrame(summary_rows, columns=["Model", "Mean F1", "Std", "Color"])
        fig_bench, ax_bench = plt.subplots(figsize=(9, 4), facecolor="#0f172a")
        ax_bench.set_facecolor("#0f172a")
        colors = [row[3] for row in summary_rows]
        bars = ax_bench.bar(benchmark_df["Model"], benchmark_df["Mean F1"], color=colors, edgecolor="white", linewidth=1.0, alpha=0.9)
        ax_bench.set_ylim(0.0, 1.05)
        ax_bench.set_ylabel("Mean F1", color="#94a3b8", fontsize=10)
        ax_bench.set_title("6-Fold Scene-CV Benchmark (validated CAFE summary)", color="#f8fafc", fontsize=11, fontweight="bold")
        ax_bench.tick_params(axis='x', colors="#e2e8f0", labelsize=9, rotation=18)
        ax_bench.tick_params(axis='y', colors="#94a3b8")
        ax_bench.grid(axis='y', linestyle='--', color="#334155", alpha=0.35)
        for b, v in zip(bars, benchmark_df["Mean F1"]):
            ax_bench.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha='center', va='bottom', color="#e2e8f0", fontsize=8)
        fig_bench.tight_layout()
        st.pyplot(fig_bench, width='stretch')

        st.markdown("#### Best validated benchmark summary")
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.metric("SAR-only expert", f"{benchmark_summary.get('mean_f1_sar',0.0):.4f}", f"± {benchmark_summary.get('std_f1_sar',0.0):.4f}")
        with b2: st.metric("HSI-only expert", f"{benchmark_summary.get('mean_f1_hsi',0.0):.4f}", f"± {benchmark_summary.get('std_f1_hsi',0.0):.4f}")
        with b3: st.metric("CAFE fusion v2", f"{benchmark_summary.get('mean_f1_fusion',0.0):.4f}", f"± {benchmark_summary.get('std_f1_fusion',0.0):.4f}")
        with b4: st.metric("Oracle upper bound", f"{0.9501:.4f}", f"recovery {benchmark_summary.get('oracle_gap_recovered_pct',0.0):.1f}%")

    input_mode = st.radio("Select Analysis Mode:", ["Benchmark Scene Explorer (18 AVIRIS/SAR Scenes)", "Custom Sensor Upload (.npy / .mat)"], horizontal=True)

    if input_mode == "Benchmark Scene Explorer (18 AVIRIS/SAR Scenes)":
        scene_list = list(cached_scenes.keys()) if cached_scenes else [f"GM{i:02d}" for i in range(1, 19)]
        sc_col1, sc_col2 = st.columns([2, 3])
        with sc_col1:
            chosen_scene = st.selectbox("Select Target Marine Scene:", scene_list, index=scene_list.index("GM04") if "GM04" in scene_list else 0)
        
        if chosen_scene in cached_scenes:
            s_data = cached_scenes[chosen_scene]
            n_px = len(s_data["labels"])
            n_oil = int(np.sum(s_data["labels"]))
            oil_pct = (n_oil / n_px) * 100 if n_px > 0 else 0
            
            with sc_col2:
                st.markdown(f"""
                <div style='background:#1e293b; border:1px solid #334155; border-radius:10px; padding:10px 16px; margin-top:24px;'>
                    <span style='color:#94a3b8; font-size:12px;'>Scene: <b>{chosen_scene}</b> | Dataset Split: <b style='color:#38bdf8;'>{s_data['split'].upper()}</b> | Total Pixels: <b>{n_px:,}</b> | Ground Truth Oil: <b style='color:#f43f5e;'>{n_oil:,} px ({oil_pct:.1f}%)</b></span>
                </div>
                """, unsafe_allow_html=True)

            # 1. Feature Extraction & Model Inference
            sar_p_raw = np.clip(s_data["sar_features"][:, 0], 1e-6, 1.0 - 1e-6)
            sar_z_raw = np.log(sar_p_raw / (1.0 - sar_p_raw))
            hsi_z_raw = s_data["hsi_logits"].reshape(-1)
            hsi_p_raw = np.clip(1.0 / (1.0 + np.exp(-hsi_z_raw)), 1e-6, 1.0 - 1e-6)
            gt_y = s_data["labels"].reshape(-1)

            u_feats = build_gate_features_live(
                sar_p_raw, hsi_p_raw,
                hsi_conf_flat=s_data["hsi_confidence"].reshape(-1),
                sar_mean_flat=s_data["sar_features"][:, 1],
                sar_std_flat=s_data["sar_features"][:, 2]
            )

            u_tensor = torch.tensor(u_feats, dtype=torch.float32, device=device_cafe)
            z_sar_t = torch.tensor(sar_z_raw, dtype=torch.float32, device=device_cafe).unsqueeze(1)
            z_hsi_t = torch.tensor(hsi_z_raw, dtype=torch.float32, device=device_cafe).unsqueeze(1)

            # Deterministic Pass
            model_cafe.eval()
            with torch.no_grad():
                p_det_t, g_det_t, _ = model_cafe(u_tensor, z_sar_t, z_hsi_t)
                p_det = p_det_t.squeeze().cpu().numpy()
                g_det = g_det_t.squeeze().cpu().numpy()

            # MC Dropout Passes
            enable_mc_dropout(model_cafe)
            mc_p_samples = np.zeros((mc_passes, len(gt_y)), dtype=np.float32)
            mc_g_samples = np.zeros((mc_passes, len(gt_y)), dtype=np.float32)

            torch.manual_seed(42 + selected_fold)
            with torch.no_grad():
                for t_i in range(mc_passes):
                    p_t, g_t, _ = model_cafe(u_tensor, z_sar_t, z_hsi_t)
                    mc_p_samples[t_i] = p_t.squeeze().cpu().numpy()
                    mc_g_samples[t_i] = g_t.squeeze().cpu().numpy()

            p_mc_mean = np.mean(mc_p_samples, axis=0)
            p_mc_var = np.var(mc_p_samples, axis=0)  # Epistemic Uncertainty
            p_mc_std = np.std(mc_p_samples, axis=0)
            g_mc_mean = np.mean(mc_g_samples, axis=0)

            # Metrics Computation
            y_pred_sar = (sar_p_raw >= fusion_threshold).astype(int)
            y_pred_hsi = (hsi_p_raw >= fusion_threshold).astype(int)
            y_pred_det = (p_det >= fusion_threshold).astype(int)
            y_pred_mc = (p_mc_mean >= fusion_threshold).astype(int)

            f1_sar = f1_score(gt_y, y_pred_sar, zero_division=0)
            f1_hsi = f1_score(gt_y, y_pred_hsi, zero_division=0)
            f1_det = f1_score(gt_y, y_pred_det, zero_division=0)
            f1_mc = f1_score(gt_y, y_pred_mc, zero_division=0)

            iou_sar = jaccard_score(gt_y, y_pred_sar, zero_division=0)
            iou_hsi = jaccard_score(gt_y, y_pred_hsi, zero_division=0)
            iou_det = jaccard_score(gt_y, y_pred_det, zero_division=0)
            iou_mc = jaccard_score(gt_y, y_pred_mc, zero_division=0)

            # Oracle
            err_sar = np.abs(sar_p_raw - gt_y)
            err_hsi = np.abs(hsi_p_raw - gt_y)
            p_oracle = np.where(err_sar < err_hsi, sar_p_raw, hsi_p_raw)
            f1_oracle = f1_score(gt_y, (p_oracle >= fusion_threshold).astype(int), zero_division=0)

            # Uncertainty alignment on this scene
            is_corr = (y_pred_mc == gt_y).astype(int)
            unc_corr = float(np.mean(p_mc_var[is_corr == 1]))
            unc_err = float(np.mean(p_mc_var[is_corr == 0])) if (is_corr == 0).sum() > 0 else 0.0
            unc_ratio = unc_err / (unc_corr + 1e-12)

            # =========================================================
            # 1. KPI STRIP
            # =========================================================
            st.markdown("#### Executive Performance Matrix")
            kp1, kp2, kp3, kp4, kp5, kp6 = st.columns(6)
            kpi_box = lambda label, val, sub, col: f"""
            <div style='background:#1e293b; border:1px solid {col}; border-top:3px solid {col}; border-radius:10px; padding:12px 10px; text-align:center;'>
                <p style='color:#94a3b8; font-size:10px; margin:0; font-weight:700; text-transform:uppercase;'>{label}</p>
                <p style='color:#f8fafc; font-size:20px; font-weight:800; margin:4px 0 0 0;'>{val}</p>
                <p style='color:{col}; font-size:11px; margin:2px 0 0 0; font-weight:600;'>{sub}</p>
            </div>
            """
            with kp1: st.markdown(kpi_box("SAR (CBD-Net)", f"{f1_sar:.4f}", f"IoU: {iou_sar:.4f}", "#38bdf8"), unsafe_allow_html=True)
            with kp2: st.markdown(kpi_box("HSI (HMOSDNet)", f"{f1_hsi:.4f}", f"IoU: {iou_hsi:.4f}", "#2dd4bf"), unsafe_allow_html=True)
            with kp3: 
                diff_hsi = f1_mc - f1_hsi
                diff_col = "#2dd4bf" if diff_hsi >= 0 else "#ef4444"
                st.markdown(kpi_box("CAFE-Fusion v2", f"{f1_mc:.4f}", f"Δ vs HSI: {diff_hsi:+.4f}", "#c084fc"), unsafe_allow_html=True)
            with kp4: st.markdown(kpi_box("Theoretical Oracle", f"{f1_oracle:.4f}", "Ceiling Bound", "#f59e0b"), unsafe_allow_html=True)
            with kp5: st.markdown(kpi_box("Mean Gate Weight (g)", f"{np.mean(g_mc_mean):.3f}", f"Std: {np.std(g_mc_mean):.3f}", "#38bdf8"), unsafe_allow_html=True)
            with kp6: st.markdown(kpi_box("Uncertainty Ratio", f"{unc_ratio:.2f}x", f"Err: {unc_err:.1e}", "#ef4444"), unsafe_allow_html=True)

            st.divider()

            # =========================================================
            # 2. 5-PANEL HIGH-RESOLUTION AUDIT FIGURE
            # =========================================================
            audit_fig_path = os.path.join(PROJECT_ROOT, "final_fusion_audit", "figures", "fusion", f"{chosen_scene}.png")
            if os.path.exists(audit_fig_path):
                st.markdown(f"#### Authoritative 5-Panel High-Resolution Audit Map ({chosen_scene})")
                st.image(audit_fig_path, caption=f"CAFE-Fusion v2 Publication Figure — {chosen_scene} (Ground Truth | SAR Prob | HSI Prob | CAFE-Fusion Prob | Fusion Prediction)")
            
            st.divider()

            # =========================================================
            # 3. DIAGNOSTIC CHARTS: REGIMES, GATE & UNCERTAINTY
            # =========================================================
            st.markdown("#### Diagnostic Analytics & Uncertainty Profiles")
            d_c1, d_c2, d_c3 = st.columns([1.2, 1, 1])

            with d_c1:
                # Gate Distribution
                fig_g, ax_g = plt.subplots(figsize=(7, 4), facecolor='#0f172a')
                ax_g.set_facecolor('#0f172a')
                ax_g.hist(g_mc_mean, bins=50, color='#c084fc', alpha=0.75, edgecolor='none', label='Gate Weight (g)')
                ax_g.axvline(np.mean(g_mc_mean), color='#f59e0b', linestyle='--', linewidth=2, label=f'Mean g = {np.mean(g_mc_mean):.3f}')
                ax_g.axvline(0.119, color='#2dd4bf', linestyle=':', linewidth=1.5, label='HSI Safe Anchor (g=0.119)')
                ax_g.set_title("Reliability Gate Weight Distribution", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
                ax_g.set_xlabel("g value (0 = Full HSI, 1 = Full SAR)", color='#94a3b8', fontsize=9)
                ax_g.set_ylabel("Pixel Count", color='#94a3b8', fontsize=9)
                ax_g.tick_params(colors='#94a3b8', labelsize=8)
                ax_g.grid(color='#334155', linestyle='--', alpha=0.25)
                leg_g = ax_g.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8)
                for t in leg_g.get_texts(): t.set_color('#e2e8f0')
                fig_g.tight_layout()
                st.pyplot(fig_g)

            with d_c2:
                # 4-Way Correctness Regimes
                c_s = (y_pred_sar == gt_y)
                c_h = (y_pred_hsi == gt_y)
                g_A = int(np.sum(c_h & c_s))
                g_B = int(np.sum(c_h & ~c_s))
                g_C = int(np.sum(~c_h & c_s))
                g_D = int(np.sum(~c_h & ~c_s))

                fig_pie, ax_pie = plt.subplots(figsize=(4.5, 4), facecolor='#0f172a')
                ax_pie.set_facecolor('#0f172a')
                sizes = [g_A, g_B, g_C, g_D]
                labels = ["Group A\nBoth Correct", "Group B\nHSI✓ SAR✗", "Group C\nHSI✗ SAR✓", "Group D\nBoth Wrong"]
                colors = ['#2dd4bf', '#38bdf8', '#f59e0b', '#ef4444']
                wedges, texts, autotexts = ax_pie.pie(
                    sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=140, textprops={'color': '#e2e8f0', 'fontsize': 8},
                    wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
                )
                for at in autotexts: at.set_color('#0f172a'); at.set_fontweight('bold')
                ax_pie.set_title("Modality Agreement Regimes", color='#f8fafc', fontsize=10, fontweight='bold', pad=6)
                fig_pie.tight_layout()
                st.pyplot(fig_pie)

            with d_c3:
                # Risk-Coverage Selective Rejection
                fig_rc, ax_rc = plt.subplots(figsize=(5, 4), facecolor='#0f172a')
                ax_rc.set_facecolor('#0f172a')
                covs = np.linspace(0.10, 1.00, 30)
                sort_idx = np.argsort(p_mc_var)
                err_rates = []
                for c in covs:
                    n_k = max(1, int(c * len(p_mc_var)))
                    k_idx = sort_idx[:n_k]
                    err_rates.append((1.0 - np.mean(is_corr[k_idx])) * 100)

                ax_rc.plot(covs * 100, err_rates, color='#ef4444', linewidth=2.5, label='Error Rate (Risk %)')
                ax_rc.fill_between(covs * 100, err_rates, color='#ef4444', alpha=0.15)
                ax_rc.set_title("Selective Risk-Coverage", color='#f8fafc', fontsize=10, fontweight='bold', pad=8)
                ax_rc.set_xlabel("Data Coverage (%)", color='#94a3b8', fontsize=9)
                ax_rc.set_ylabel("Error Rate (%)", color='#94a3b8', fontsize=9)
                ax_rc.tick_params(colors='#94a3b8', labelsize=8)
                ax_rc.grid(color='#334155', linestyle='--', alpha=0.25)
                fig_rc.tight_layout()
                st.pyplot(fig_rc)

            # Detailed Scene Metrics Table
            st.markdown("#### Quantitative Model Performance Summary")
            scene_summary_df = pd.DataFrame([
                {"Model Architecture": "SAR Expert (CBD-Net)", "F1 Score": f"{f1_sar:.4f}", "IoU Metric": f"{iou_sar:.4f}", "Precision": f"{precision_score(gt_y, y_pred_sar, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_sar, zero_division=0):.4f}", "Role": "Spatial Expert"},
                {"Model Architecture": "HSI Expert (HMOSDNet)", "F1 Score": f"{f1_hsi:.4f}", "IoU Metric": f"{iou_hsi:.4f}", "Precision": f"{precision_score(gt_y, y_pred_hsi, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_hsi, zero_division=0):.4f}", "Role": "Dominant Spectral Anchor"},
                {"Model Architecture": "CAFE-Fusion v2 (Deterministic)", "F1 Score": f"{f1_det:.4f}", "IoU Metric": f"{iou_det:.4f}", "Precision": f"{precision_score(gt_y, y_pred_det, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_det, zero_division=0):.4f}", "Role": "Logit-Space Fused"},
                {"Model Architecture": f"CAFE-Fusion v2 + MC Dropout (T={mc_passes})", "F1 Score": f"{f1_mc:.4f}", "IoU Metric": f"{iou_mc:.4f}", "Precision": f"{precision_score(gt_y, y_pred_mc, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_mc, zero_division=0):.4f}", "Role": "Confidence-Aware + Epistemic Unc."},
                {"Model Architecture": "Theoretical Oracle Bound", "F1 Score": f"{f1_oracle:.4f}", "IoU Metric": f"{jaccard_score(gt_y, (p_oracle>=fusion_threshold).astype(int), zero_division=0):.4f}", "Precision": "-", "Recall": "-", "Role": "Upper Limit"},
            ])
            st.dataframe(scene_summary_df, width='stretch', hide_index=True)

            # Mathematical Foundations Explainer
            with st.expander("📐 Mathematical Formulations & Architecture Deep-Dive"):
                st.markdown("##### 1. Convex Asymmetric Mixing in Logit Space")
                st.latex(r"z_{\text{SAR}} = \ln\left(\frac{p_{\text{SAR}}}{1 - p_{\text{SAR}}}\right), \quad z_{\text{HSI}} = \ln\left(\frac{p_{\text{HSI}}}{1 - p_{\text{HSI}}}\right)")
                st.latex(r"z_{\text{fused}} = (1 - g) \cdot z_{\text{HSI}} + g \cdot z_{\text{SAR}}, \quad p_{\text{fused}} = \sigma(z_{\text{fused}})")

                st.markdown("##### 2. 11-D Registration-Free Gate Feature Vector ($u$)")
                st.latex(r"u = \left[ p_{\text{SAR}}, z_{\text{SAR}}, p_{\text{HSI}}, z_{\text{HSI}}, |p_{\text{SAR}}-p_{\text{HSI}}|, p_{\text{SAR}}-p_{\text{HSI}}, q_{\text{HSI}}, H(p_{\text{HSI}}), H(p_{\text{SAR}}), \mu_{\text{SAR}}, \sigma_{\text{SAR}} \right]")

                st.markdown("##### 3. Asymmetric Anchor Inductive Bias")
                st.latex(r"\text{Bias}_{\text{init}} = -2.0 \implies g_{\text{init}} = \sigma(-2.0) \approx 0.1192 \quad (\sim 88\%\text{ HSI Dominance by default})")

                st.markdown(f"##### 4. Epistemic Uncertainty via Monte Carlo Dropout ($T={mc_passes}$ passes)")
                st.latex(r"\bar{p}(x) = \frac{1}{T} \sum_{t=1}^T p_t(x), \quad \sigma^2_{\text{epistemic}}(x) = \frac{1}{T} \sum_{t=1}^T \left(p_t(x) - \bar{p}(x)\right)^2")

    else:
        st.markdown("""
        <div style='background:#1e293b; border:1px solid #38bdf8; border-left:4px solid #38bdf8; border-radius:8px; padding:12px 16px; margin-bottom:16px;'>
            <h5 style='color:#38bdf8; margin:0 0 4px 0; font-weight:700;'>Custom Sensor Upload Guide</h5>
            <p style='color:#94a3b8; font-size:12px; margin:0;'>
                Provide paired unaligned sensor streams for live inference:<br>
                • <b>SAR Stream (Spatial Expert)</b>: Grayscale PNG / JPG / NPY array of radar backscatter intensity &sigma;<sub>0</sub> (e.g., from <code>new_dataset/OSD/TrainSAR/GM01.png</code>).<br>
                • <b>HSI Cube (Spectral Expert)</b>: MATLAB <code>.mat</code> file containing 224-band radiance data <code>img</code> (e.g., from <code>new_dataset/OSD/TrainCopy/GM01.mat</code>).
            </p>
        </div>
        """, unsafe_allow_html=True)
        u1, u2 = st.columns(2)
        with u1:
            up_sar = st.file_uploader("Upload SAR Data Stream (.npy or Image)", type=["npy", "jpg", "png", "jpeg"], key="up_sar_fus")
        with u2:
            up_hsi = st.file_uploader("Upload Hyperspectral Cube (.mat)", type=["mat"], key="up_hsi_fus")

        if up_sar is not None and up_hsi is not None and model_sar is not None and model_hsi is not None:
            # Process SAR with exact CBDNet dB normalization
            if up_sar.name.endswith(".npy"):
                s_raw = np.load(up_sar)
                if s_raw.ndim == 3: s_raw = s_raw[0]
            else:
                s_raw = np.array(Image.open(up_sar).convert("L"))
            
            sar_256 = cv2.resize(s_raw, (256, 256)).astype(np.float32)
            eps_sar = 1e-6
            sar_db = 10.0 * np.log10(np.abs(sar_256) + eps_sar)
            sar_norm = (sar_db - sar_db.mean()) / (sar_db.std() + eps_sar)
            sar_tensor = torch.from_numpy(np.stack([sar_norm, sar_norm, sar_norm], axis=0)).unsqueeze(0).to(device_sar).float()
            with torch.no_grad():
                s_log = model_sar(sar_tensor)
                s_logits = s_log[0] if isinstance(s_log, (tuple, list)) else s_log
                p_sar_live = torch.sigmoid(s_logits).squeeze().cpu().numpy().astype(np.float32)

            # Process HSI
            h_mat = sio.loadmat(up_hsi)
            k_h = [k for k in h_mat.keys() if not k.startswith('__') and k not in ['gt', 'map', 'label', 'labels', 'GroundTruth', 'ROI', 'index']][0]
            raw_cube = np.nan_to_num(h_mat[k_h].astype(np.float32), 0)
            h_h, w_h = raw_cube.shape[:2]
            n_b = raw_cube.shape[2] if raw_cube.ndim == 3 else 1
            
            # Adaptive Band Alignment: Ensure compatibility with fitted 193-band PCA
            if n_b == 224:
                cube_pca = processor_hsi.transform(raw_cube)
            elif n_b == 193:
                cube_pca = processor_hsi.pca.transform(raw_cube.reshape(-1, 193)).reshape(h_h, w_h, 25)
            elif n_b > 193:
                trimmed_cube = raw_cube[:, :, :193]
                cube_pca = processor_hsi.pca.transform(trimmed_cube.reshape(-1, 193)).reshape(h_h, w_h, 25)
            else:
                pad_b = 193 - n_b
                padded_cube = np.pad(raw_cube, ((0, 0), (0, 0), (0, pad_b)), mode='edge')
                cube_pca = processor_hsi.pca.transform(padded_cube.reshape(-1, 193)).reshape(h_h, w_h, 25)

            pad_h = PATCH_SIZE_HSI // 2
            pad_cube = np.pad(cube_pca, ((pad_h, pad_h), (pad_h, pad_h), (0, 0)), mode='reflect')
            p_hsi_full = np.zeros((h_h, w_h), dtype=np.float32)

            with torch.no_grad():
                for r_i in range(h_h):
                    patches = [np.expand_dims(pad_cube[r_i:r_i+PATCH_SIZE_HSI, c_j:c_j+PATCH_SIZE_HSI, :].transpose(2, 0, 1), 0) for c_j in range(w_h)]
                    b_t = torch.from_numpy(np.array(patches)).to(device_hsi).float()
                    p_hsi_full[r_i, :] = torch.sigmoid(model_hsi(b_t, return_features=False)).squeeze(1).cpu().numpy().flatten()

            p_hsi_live = cv2.resize(p_hsi_full, (256, 256)).astype(np.float32)

            # Build Gate Features & Run MC Dropout
            u_live = build_gate_features_live(p_sar_live.flatten(), p_hsi_live.flatten())
            u_t = torch.tensor(u_live, dtype=torch.float32, device=device_cafe)
            z_s_t = torch.tensor(np.log(np.clip(p_sar_live.flatten(), 1e-6, 1-1e-6) / (1-np.clip(p_sar_live.flatten(), 1e-6, 1-1e-6))), dtype=torch.float32, device=device_cafe).unsqueeze(1)
            z_h_t = torch.tensor(np.log(np.clip(p_hsi_live.flatten(), 1e-6, 1-1e-6) / (1-np.clip(p_hsi_live.flatten(), 1e-6, 1-1e-6))), dtype=torch.float32, device=device_cafe).unsqueeze(1)

            enable_mc_dropout(model_cafe)
            mc_live = np.zeros((mc_passes, 256*256), dtype=np.float32)
            mc_g_live = np.zeros((mc_passes, 256*256), dtype=np.float32)
            with torch.no_grad():
                for t_l in range(mc_passes):
                    p_l, g_l, _ = model_cafe(u_t, z_s_t, z_h_t)
                    mc_live[t_l] = p_l.squeeze().cpu().numpy()
                    mc_g_live[t_l] = g_l.squeeze().cpu().numpy()

            p_fus_live = np.mean(mc_live, axis=0).reshape(256, 256)
            p_unc_live = np.var(mc_live, axis=0).reshape(256, 256)
            g_live_mean = np.mean(mc_g_live, axis=0).reshape(256, 256)

            # Check for Ground Truth in the uploaded HSI file
            gt_key = next((k for k in ['gt', 'map', 'label', 'labels', 'GroundTruth', 'ROI', 'y'] if k in h_mat), None)
            
            if gt_key is not None:
                gt_raw = h_mat[gt_key]
                gt_256 = cv2.resize(gt_raw.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                gt_y = (gt_256.flatten() > 0).astype(int)

                # Quantitative Metrics Computation
                y_pred_sar = (p_sar_live.flatten() >= fusion_threshold).astype(int)
                y_pred_hsi = (p_hsi_live.flatten() >= fusion_threshold).astype(int)
                y_pred_mc = (p_fus_live.flatten() >= fusion_threshold).astype(int)

                f1_sar = f1_score(gt_y, y_pred_sar, zero_division=0)
                f1_hsi = f1_score(gt_y, y_pred_hsi, zero_division=0)
                f1_mc = f1_score(gt_y, y_pred_mc, zero_division=0)

                iou_sar = jaccard_score(gt_y, y_pred_sar, zero_division=0)
                iou_hsi = jaccard_score(gt_y, y_pred_hsi, zero_division=0)
                iou_mc = jaccard_score(gt_y, y_pred_mc, zero_division=0)

                # Oracle Upper Bound
                err_sar = np.abs(p_sar_live.flatten() - gt_y)
                err_hsi = np.abs(p_hsi_live.flatten() - gt_y)
                p_oracle = np.where(err_sar < err_hsi, p_sar_live.flatten(), p_hsi_live.flatten())
                f1_oracle = f1_score(gt_y, (p_oracle >= fusion_threshold).astype(int), zero_division=0)

                # Uncertainty Error Elevation
                is_corr = (y_pred_mc == gt_y).astype(int)
                unc_corr = float(np.mean(p_unc_live.flatten()[is_corr == 1])) if (is_corr == 1).sum() > 0 else 1e-12
                unc_err = float(np.mean(p_unc_live.flatten()[is_corr == 0])) if (is_corr == 0).sum() > 0 else 0.0
                unc_ratio = unc_err / (unc_corr + 1e-12)

                # Executive KPI Strip
                st.markdown("#### Executive Performance Matrix")
                kp1, kp2, kp3, kp4, kp5, kp6 = st.columns(6)
                kpi_box = lambda label, val, sub, col: f"""
                <div style='background:#1e293b; border:1px solid {col}; border-top:3px solid {col}; border-radius:10px; padding:12px 10px; text-align:center;'>
                    <p style='color:#94a3b8; font-size:10px; margin:0; font-weight:700; text-transform:uppercase;'>{label}</p>
                    <p style='color:#f8fafc; font-size:20px; font-weight:800; margin:4px 0 0 0;'>{val}</p>
                    <p style='color:{col}; font-size:11px; margin:2px 0 0 0; font-weight:600;'>{sub}</p>
                </div>
                """
                diff_hsi = f1_mc - f1_hsi
                diff_col = "#2dd4bf" if diff_hsi >= 0 else "#ef4444"
                with kp1: st.markdown(kpi_box("SAR (CBD-Net)", f"{f1_sar:.4f}", f"IoU: {iou_sar:.4f}", "#38bdf8"), unsafe_allow_html=True)
                with kp2: st.markdown(kpi_box("HSI (HMOSDNet)", f"{f1_hsi:.4f}", f"IoU: {iou_hsi:.4f}", "#2dd4bf"), unsafe_allow_html=True)
                with kp3: st.markdown(kpi_box("CAFE-Fusion v2", f"{f1_mc:.4f}", f"Δ vs HSI: {diff_hsi:+.4f}", "#c084fc"), unsafe_allow_html=True)
                with kp4: st.markdown(kpi_box("Theoretical Oracle", f"{f1_oracle:.4f}", "Ceiling Bound", "#f59e0b"), unsafe_allow_html=True)
                with kp5: st.markdown(kpi_box("Mean Gate Weight (g)", f"{np.mean(g_live_mean):.3f}", f"Std: {np.std(g_live_mean):.3f}", "#38bdf8"), unsafe_allow_html=True)
                with kp6: st.markdown(kpi_box("Uncertainty Ratio", f"{unc_ratio:.2f}x", f"Err: {unc_err:.1e}", "#ef4444"), unsafe_allow_html=True)

                st.divider()

            # Publication-Grade 5-Panel Audit Figure (matches GM17.png exactly)
            st.markdown("#### Live Sensor Output Comparison & Decision Audit")
            
            # Compute binary prediction
            pred_binary_live = (p_fus_live >= fusion_threshold).astype(np.float32)
            mean_g_live = float(np.mean(g_live_mean))

            # Build the exact 5-panel figure matching generate_fusion_audit_visualizations.py
            H_fig, W_fig = p_fus_live.shape

            if gt_key is not None:
                has_gt = True
                gt_2d = gt_256.astype(np.float32) / max(gt_256.max(), 1)
                f1_header = f"SAR $F_1$ = {f1_sar:.4f} → HSI $F_1$ = {f1_hsi:.4f} → Fusion $F_1$ = {f1_mc:.4f}  |  Mean Gate $\\bar{{g}}$ = {mean_g_live:.4f}"
                panels = [
                    ("Ground Truth", gt_2d, "gray", False, "#0f172a"),
                    (f"SAR Expert Prob\n($F_1$ = {f1_sar:.4f})", p_sar_live, "viridis", True, "#0f172a"),
                    (f"HSI Expert Prob\n($F_1$ = {f1_hsi:.4f})", p_hsi_live, "viridis", True, "#0f172a"),
                    (f"CAFE-Fusion Prob\n($\\bar{{g}}$ = {mean_g_live:.3f})", p_fus_live, "plasma", True, "#0284c7"),
                    (f"Fusion Prediction @ {fusion_threshold:.3f}\n($F_1$ = {f1_mc:.4f})", pred_binary_live, "gray", False, "#0284c7"),
                ]
            else:
                has_gt = False
                f1_header = f"Custom Upload  |  Mean Gate $\\bar{{g}}$ = {mean_g_live:.4f}"
                panels = [
                    ("SAR Input (Uploaded)", (sar_256 - sar_256.min()) / (sar_256.max() - sar_256.min() + 1e-6), "gray", False, "#0f172a"),
                    ("SAR Expert Prob\n(CBD-Net)", p_sar_live, "viridis", True, "#0f172a"),
                    ("HSI Expert Prob\n(HMOSDNet)", p_hsi_live, "viridis", True, "#0f172a"),
                    (f"CAFE-Fusion Prob\n($\\bar{{g}}$ = {mean_g_live:.3f})", p_fus_live, "plasma", True, "#0284c7"),
                    (f"Fusion Prediction @ {fusion_threshold:.3f}", pred_binary_live, "gray", False, "#0284c7"),
                ]

            fig_pub, gs_pub = plt.subplots(1, 5, figsize=(20, 5.25), facecolor="white")
            fig_pub.suptitle(f1_header, fontsize=12, fontweight="bold", color="#0f172a", y=0.98)
            for ax_p, (title, data, cmap_name, show_cbar, t_color) in zip(gs_pub, panels):
                ax_p.set_facecolor("white")
                im_p = ax_p.imshow(data, cmap=cmap_name, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="auto")
                ax_p.set_title(title, fontsize=10, fontweight="bold", color=t_color, pad=6)
                ax_p.axis("off")
                if show_cbar:
                    cbar_p = fig_pub.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.025)
                    cbar_p.ax.tick_params(labelsize=7)
                    cbar_p.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            fig_pub.tight_layout()
            st.pyplot(fig_pub, width='stretch')

            # Quantitative Table
            st.markdown("#### Quantitative Performance & Area Breakdown")
            sar_area_live = int(np.sum(p_sar_live > fusion_threshold))
            hsi_area_live = int(np.sum(p_hsi_live > fusion_threshold))
            fus_area_live = int(np.sum(p_fus_live > fusion_threshold))
            mean_unc_live = float(np.mean(p_unc_live))

            if gt_key is not None:
                live_df = pd.DataFrame([
                    {"Model Architecture": "SAR Expert (CBD-Net)", "F1 Score": f"{f1_sar:.4f}", "IoU Metric": f"{iou_sar:.4f}", "Precision": f"{precision_score(gt_y, y_pred_sar, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_sar, zero_division=0):.4f}", "Detected Area": f"{sar_area_live:,} px"},
                    {"Model Architecture": "HSI Expert (HMOSDNet)", "F1 Score": f"{f1_hsi:.4f}", "IoU Metric": f"{iou_hsi:.4f}", "Precision": f"{precision_score(gt_y, y_pred_hsi, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_hsi, zero_division=0):.4f}", "Detected Area": f"{hsi_area_live:,} px"},
                    {"Model Architecture": f"CAFE-Fusion v2 (MC T={mc_passes})", "F1 Score": f"{f1_mc:.4f}", "IoU Metric": f"{iou_mc:.4f}", "Precision": f"{precision_score(gt_y, y_pred_mc, zero_division=0):.4f}", "Recall": f"{recall_score(gt_y, y_pred_mc, zero_division=0):.4f}", "Detected Area": f"{fus_area_live:,} px"},
                    {"Model Architecture": "Theoretical Oracle Bound", "F1 Score": f"{f1_oracle:.4f}", "IoU Metric": f"{jaccard_score(gt_y, (p_oracle>=fusion_threshold).astype(int), zero_division=0):.4f}", "Precision": "-", "Recall": "-", "Detected Area": "-"},
                ])

            else:
                live_df = pd.DataFrame([
                    {"Model Architecture": "SAR Expert (CBD-Net)", "Detected Area": f"{sar_area_live:,} px", "Mean Confidence": f"{np.mean(p_sar_live)*100:.1f}%", "Uncertainty": "N/A"},
                    {"Model Architecture": "HSI Expert (HMOSDNet)", "Detected Area": f"{hsi_area_live:,} px", "Mean Confidence": f"{np.mean(p_hsi_live)*100:.1f}%", "Uncertainty": "N/A"},
                    {"Model Architecture": f"CAFE-Fusion v2 (MC T={mc_passes})", "Detected Area": f"{fus_area_live:,} px", "Mean Confidence": f"{np.mean(p_fus_live)*100:.1f}%", "Uncertainty": f"{mean_unc_live:.2e}"},
                ])
            st.dataframe(live_df, width='stretch', hide_index=True)


# =============================================================================
# TAB 2: SAR INTELLIGENCE (CBD-Net)
# =============================================================================
with tab_sar:
    uploaded_file = st.file_uploader("Select SAR Data Stream (.npy / .jpg)", type=["npy", "jpg", "png", "jpeg"], key="sar_single")

    if uploaded_file is not None and model_sar:
        # 1. Pipeline: Load & Process
        if uploaded_file.name.endswith(".npy"):
            data = np.load(uploaded_file)
            if data.ndim == 3: data = data[0]
            data = cv2.resize(data, (256, 256))
        else:
            data = np.array(Image.open(uploaded_file).convert("L"))
            data = cv2.resize(data, (256, 256))
        
        real_view, image_db = enhance_realistic_sar(data)
        ai_input = preprocess_for_ai(image_db)
        input_tensor = torch.from_numpy(ai_input).unsqueeze(0).to(device_sar).float()
    
        # 2. Engine: Inference
        with torch.no_grad():
            seg_logits, boundary_logits, _, mask_conf = model_sar(input_tensor)
            prob_sar = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            boundary_prob = torch.sigmoid(boundary_logits[0]).squeeze().cpu().numpy()
    
        mask_sar = morphology.remove_small_objects(prob_sar > threshold_sar, min_size=min_size_sar)
        detected_area_sar = np.sum(mask_sar)
        confidence_score_sar = mask_conf.item() * 100
    
        # --- 3-COLUMN MASTER VIEW ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<p style='text-align:center; font-weight:700; color: #38bdf8;'>{uploaded_file.name}</p>", unsafe_allow_html=True)
            st.image(real_view, width='stretch')
        with c2:
            st.markdown("<p style='text-align:center; font-weight:700; color: #38bdf8;'>Expert Edge Vision</p>", unsafe_allow_html=True)
            fig_edge, ax_edge = plt.subplots(figsize=(5,5), facecolor='#0f172a')
            ax_edge.set_facecolor('#0f172a')
            ax_edge.imshow(boundary_prob, cmap='magma')
            ax_edge.axis('off')
            fig_edge.tight_layout()
            st.pyplot(fig_edge, width='stretch')
        with c3:
            st.markdown("<p style='text-align:center; font-weight:700; color: #38bdf8;'>Expert Detection</p>", unsafe_allow_html=True)
            img_rgb = cv2.cvtColor(real_view, cv2.COLOR_GRAY2RGB)
            overlay = img_rgb.copy()
            overlay[mask_sar] = [139, 0, 0] # Deep Red for oil
            edge_mask = boundary_prob > 0.5
            overlay[edge_mask] = [0, 255, 218] # Cyan for edges
            final_blend = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)
            st.image(final_blend, width='stretch')
        
        # --- EXPERT ANALYSIS PANEL ---
        st.divider()
        st.markdown("### Expert Intelligence Suite")
    
        # Advanced Slick Metadata Extraction
        bin_mask = (mask_sar * 255).astype(np.uint8)
        conts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        slick_info = []
        for i, cnt in enumerate(conts):
            area = cv2.contourArea(cnt)
            if area < min_size_sar: continue
        
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
            c_mask = np.zeros_like(prob_sar, dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 1, -1)
            m_conf = np.mean(prob_sar[c_mask == 1])
        
            comp = "High" if circularity < 0.3 else "Medium" if circularity < 0.6 else "Low"
        
            slick_info.append({
                "Cluster ID": f"Slick #{i+1:02d}",
                "Area (px²)": int(area),
                "Mean Confidence": f"{m_conf*100:.1f}%",
                "Complexity": comp,
                "Intelligence Score": round(circularity, 2)
            })
    
        num_slicks = len(slick_info)
        avg_conf = np.mean([float(s["Mean Confidence"].replace('%','')) for s in slick_info]) if slick_info else 0
        
        status_title = "Clear Water Signature"
        badge_class = "status-badge-nominal"
        if detected_area_sar > 1000 or num_slicks > 4:
            status_title = "Significant Slick Cluster"
            badge_class = "status-badge-critical"
        elif detected_area_sar > 200:
            status_title = "Moderate Anomaly Detected"
            badge_class = "status-badge-critical"

        # Top Metrics Row
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Total Population", f"{num_slicks} Clusters")
        with m_c2:
            st.metric("Detection Confidence", f"{avg_conf:.1f}%")
        with m_c3:
            st.metric("System Integrity", f"{confidence_score_sar:.1f}%")
        with m_c4:
            st.markdown(f"""
            <div style='background:#1e293b; padding:12px 16px; border-radius:10px; border:1px solid #334155; text-align:center;'>
                <p style='color:#94a3b8; font-size:11px; margin:0 0 4px 0; font-weight:600;'>TARGET STATUS</p>
                <span class='{badge_class}'>{status_title}</span>
            </div>
            """, unsafe_allow_html=True)

        # Middle Analytics Row
        a1, a2 = st.columns([1, 1])
        with a1:
            fig_h, ax_h = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
            ax_h.set_facecolor('#0f172a')
            ax_h.hist(prob_sar.flatten(), bins=50, color='#38bdf8', alpha=0.7, label="Pixel Confidence Distribution")
            ax_h.axvline(threshold_sar, color='#f43f5e', linestyle='--', linewidth=1.8, label=f"Sensitivity Cutoff ({threshold_sar:.2f})")
            ax_h.set_title("Probability Distribution Density", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
            ax_h.set_xlabel("Confidence Score (0.0 to 1.0)", color='#94a3b8', fontsize=9)
            ax_h.set_ylabel("Pixel Count", color='#94a3b8', fontsize=9)
            ax_h.tick_params(colors='#94a3b8', labelsize=8)
            ax_h.grid(color='#334155', linestyle='--', alpha=0.3)
            legend_h = ax_h.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8)
            for text in legend_h.get_texts(): text.set_color('#e2e8f0')
            fig_h.tight_layout()
            st.pyplot(fig_h, width='stretch')
    
        with a2:
            if slick_info:
                df_slick = pd.DataFrame(slick_info)
                fig_g, ax_g = plt.subplots(figsize=(8, 4), facecolor='#0f172a')
                ax_g.set_facecolor('#0f172a')
                bars = ax_g.bar(df_slick["Cluster ID"], df_slick["Intelligence Score"], color='#2dd4bf', alpha=0.85, width=0.5)
                ax_g.set_title("Geometric Circularity (Slick Surface Regularity)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
                ax_g.set_ylabel("Circularity Score (4πA/P²)", color='#94a3b8', fontsize=9)
                ax_g.tick_params(axis='x', rotation=30, colors='#94a3b8', labelsize=8)
                ax_g.tick_params(axis='y', colors='#94a3b8', labelsize=8)
                ax_g.grid(color='#334155', linestyle='--', alpha=0.3)
                fig_g.tight_layout()
                st.pyplot(fig_g, width='stretch')
            else:
                st.info("🌊 Clear ocean background. No complex clusters detected.")

        # Deep Intelligence Table
        if slick_info:
            st.markdown("#### 📓 Cluster Intelligence Breakdown")
            df_display = pd.DataFrame(slick_info).drop(columns=["Intelligence Score"])
            st.dataframe(df_display, width='stretch', hide_index=True)
    
        # Executive Summary Status
        st.markdown(f"""
        <div style='background:#1e293b; padding:16px 20px; border-radius:10px; margin-top:20px; border:1px solid #334155; border-left:6px solid #38bdf8;'>
            <h4 style='color:#38bdf8; margin:0 0 6px 0; font-weight:600;'>Executive Analysis Summary</h4>
            <p style='color:#94a3b8; margin:0; font-size:14px;'>CBD-Net SAR Expert isolated {num_slicks} cluster candidate(s) covering {detected_area_sar:,} spatial pixels with a mean detection confidence of {avg_conf:.1f}%.</p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: HSI SPECTRAL EXPERT (HMOSDNet)
# =============================================================================
with tab_hsi:
    uploaded_hsi = st.file_uploader("Select Hyperspectral Cube (.mat)", type=["mat"], key="hsi_upload")
    
    if uploaded_hsi is not None and model_hsi and processor_hsi:
        try:
            data_hsi = sio.loadmat(uploaded_hsi)
            img_key = [k for k in data_hsi.keys() if not k.startswith('__') and k not in ['gt', 'map', 'index']][0]
            raw_img = data_hsi[img_key].astype(np.float32)
            raw_img = np.nan_to_num(raw_img, 0)
        except Exception as e:
            st.error(f"Failed to read hyperspectral data: {e}")
            st.stop()
            
        st.markdown("<p style='color:#94a3b8; font-size:13px;'>HMOSDNet Initialized from corrected weights. Performing batched 3D-CNN patch inference & PCA spatial reduction...</p>", unsafe_allow_html=True)
        
        # 1. Generate False Color Composite (RGB)
        try:
            r_band = raw_img[:, :, min(120, raw_img.shape[2]-1)]
            g_band = raw_img[:, :, min(60, raw_img.shape[2]-1)]
            b_band = raw_img[:, :, min(28, raw_img.shape[2]-1)]
            
            def norm_band(b):
                v_min, v_max = np.percentile(b, 2), np.percentile(b, 98)
                stretched = np.clip(255 * (b - v_min) / (v_max - v_min + 1e-6), 0, 255)
                return stretched.astype(np.uint8)

            false_color = np.stack([norm_band(r_band), norm_band(g_band), norm_band(b_band)], axis=-1)
        except Exception:
            false_color = np.zeros((raw_img.shape[0], raw_img.shape[1], 3), dtype=np.uint8)
            
        # 2. Preprocess & Inference (Batched Sliding Window)
        h, w, c = raw_img.shape
        if c == 224:
            img_pca = processor_hsi.transform(raw_img)
        elif c == 193:
            img_pca = processor_hsi.pca.transform(raw_img.reshape(-1, 193)).reshape(h, w, 25)
        elif c > 193:
            st.info(f"ℹ️ Input cube contains {c} bands. Adaptively aligning to 193 transmission channels.")
            trimmed = raw_img[:, :, :193]
            img_pca = processor_hsi.pca.transform(trimmed.reshape(-1, 193)).reshape(h, w, 25)
        else:
            st.info(f"ℹ️ Input cube contains {c} bands. Padding to 193 transmission channels.")
            padded = np.pad(raw_img, ((0, 0), (0, 0), (0, 193 - c)), mode='edge')
            img_pca = processor_hsi.pca.transform(padded.reshape(-1, 193)).reshape(h, w, 25)

        pad = PATCH_SIZE_HSI // 2
        img_pad = np.pad(img_pca, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
        prob_map_hsi = np.zeros((h, w), dtype=np.float32)
        
        batch_size = 512
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with torch.no_grad():
            for i in range(h):
                status_text.text(f"Scanning Hyperspectral Geometries... {int((i/h)*100)}%")
                progress_bar.progress(int((i/h)*100))
                
                patches = []
                for j in range(w):
                    patch = img_pad[i:i+PATCH_SIZE_HSI, j:j+PATCH_SIZE_HSI, :]
                    patch = np.expand_dims(patch.transpose(2, 0, 1), axis=0)
                    patches.append(patch)
                
                row_probs = []
                for b_start in range(0, len(patches), batch_size):
                    batch_np = np.array(patches[b_start:b_start+batch_size])
                    batch = torch.from_numpy(batch_np).to(device_hsi).float()
                    logits = model_hsi(batch, return_features=False)
                    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                    row_probs.append(probs)
                prob_map_hsi[i, :] = np.concatenate(row_probs).flatten()
                
        progress_bar.empty()
        status_text.empty()
        
        mask_hsi = morphology.remove_small_objects(prob_map_hsi > threshold_hsi, min_size=min_size_sar)
        detected_area_hsi = np.sum(mask_hsi)
        mean_hsi_conf = np.mean(prob_map_hsi[mask_hsi]) * 100 if detected_area_hsi > 0 else 0.0
        
        hsi_status_title = "Clear Seawater Baseline"
        hsi_badge_class = "status-badge-nominal"
        if detected_area_hsi > 500:
            hsi_status_title = "Hydrocarbon Signature Identified"
            hsi_badge_class = "status-badge-critical"
        elif detected_area_hsi > 50:
            hsi_status_title = "Moderate Spectral Anomaly"
            hsi_badge_class = "status-badge-critical"
            
        # Top HSI Executive Metrics
        hm1, hm2, hm3, hm4 = st.columns(4)
        with hm1:
            st.metric("Detected Slick Area", f"{detected_area_hsi:,} px²")
        with hm2:
            st.metric("Mean Hydrocarbon Conf.", f"{mean_hsi_conf:.1f}%")
        with hm3:
            st.metric("Bands Analyzed", "224 Bands (25 PCA)")
        with hm4:
            st.markdown(f"""
            <div style='background:#1e293b; padding:12px 16px; border-radius:10px; border:1px solid #334155; text-align:center;'>
                <p style='color:#94a3b8; font-size:11px; margin:0 0 4px 0; font-weight:600;'>TARGET STATUS</p>
                <span class='{hsi_badge_class}'>{hsi_status_title}</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # --- 3-COLUMN SPECTRAL VIEW ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<p style='text-align:center; font-weight:600; color: #38bdf8; font-size:13px;'>{uploaded_hsi.name} (False Color RGB)</p>", unsafe_allow_html=True)
            st.image(false_color, width='stretch')
        with c2:
            st.markdown("<p style='text-align:center; font-weight:600; color: #38bdf8; font-size:13px;'>HMOSDNet Probability Field</p>", unsafe_allow_html=True)
            fig_hm, ax_hm = plt.subplots(figsize=(5,5), facecolor='#0f172a')
            ax_hm.set_facecolor('#0f172a')
            im_hm = ax_hm.imshow(prob_map_hsi, cmap='viridis', vmin=0, vmax=1)
            ax_hm.axis('off')
            cbar = fig_hm.colorbar(im_hm, ax=ax_hm, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='#94a3b8', labelsize=8)
            cbar.outline.set_edgecolor('#334155')
            fig_hm.tight_layout()
            st.pyplot(fig_hm, width='stretch')
        with c3:
            st.markdown("<p style='text-align:center; font-weight:600; color: #38bdf8; font-size:13px;'>Confidence Density Overlay</p>", unsafe_allow_html=True)
            overlay_hsi = false_color.copy()
            
            core_threshold = min(threshold_hsi + 0.25, 0.95)
            mask_core = (prob_map_hsi >= core_threshold)
            mask_periphery = (prob_map_hsi >= threshold_hsi) & (prob_map_hsi < core_threshold)
            
            mask_core = morphology.remove_small_objects(mask_core, min_size=min_size_sar)
            mask_periphery = morphology.remove_small_objects(mask_periphery, min_size=min_size_sar)
            
            overlay_hsi[mask_periphery] = [45, 212, 191]   # Soft Teal (Marginal/Thin Periphery)
            overlay_hsi[mask_core] = [244, 63, 94]        # Soft Red/Rose (High Confidence Core)
            
            final_blend_hsi = cv2.addWeighted(false_color, 0.4, overlay_hsi, 0.6, 0)
            st.image(final_blend_hsi, width='stretch')
            
        # --- Native Scientific Legends ---
        st.markdown(
            f"""
            <div style='display: flex; justify-content: space-around; background-color: #1e293b; padding: 12px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 20px;'>
                <div style='text-align: center;'><span style='color: #f43f5e; font-weight: bold;'>■</span> <b>High AI Confidence (>{core_threshold*100:.0f}%)</b><br><span style='font-size: 12px; color: #94a3b8;'>Dense Core / Strong Hydrocarbon Signature</span></div>
                <div style='text-align: center;'><span style='color: #2dd4bf; font-weight: bold;'>■</span> <b>Marginal AI Confidence ({threshold_hsi*100:.0f}-{core_threshold*100:.0f}%)</b><br><span style='font-size: 12px; color: #94a3b8;'>Thin Periphery / Mixed Signature</span></div>
                <div style='text-align: center;'><span style='color: #38bdf8; font-weight: bold;'>■</span> <b>Baseline Water (0-{threshold_hsi*100:.0f}%)</b><br><span style='font-size: 12px; color: #94a3b8;'>Clean Seawater Reference</span></div>
            </div>
            """, unsafe_allow_html=True
        )
            
        st.divider()
        st.markdown("### Hyperspectral Physics & Spectral Divergence Proof")
        st.markdown("Quantifies solar radiance reflectance across all 224 AVIRIS channels (400–2500 nm), confirming the chemical hydrocarbon absorption signature against clean seawater baseline.")
        
        # Dual Analytics Row
        g1, g2 = st.columns([1.3, 1])
        
        with g1:
            if detected_area_hsi > 10:
                oil_pixels = raw_img[mask_hsi]
                water_mask = (prob_map_hsi < 0.1)
                water_pixels = raw_img[water_mask]
                
                if len(water_pixels) > 0:
                    np.random.shuffle(water_pixels)
                    water_pixels = water_pixels[:10000]
                    
                    oil_mean = np.mean(oil_pixels, axis=0)
                    oil_std = np.std(oil_pixels, axis=0)
                    water_mean = np.mean(water_pixels, axis=0)
                    water_std = np.std(water_pixels, axis=0)
                    
                    wavelengths = np.linspace(400, 2500, len(oil_mean))
                    
                    fig_s, ax_s = plt.subplots(figsize=(8, 4.2), facecolor='#0f172a')
                    ax_s.set_facecolor('#0f172a')
                    
                    ax_s.axvspan(1350, 1430, color='#64748b', alpha=0.2, hatch='//', label='Atmospheric H₂O Absorption')
                    ax_s.axvspan(1800, 1950, color='#64748b', alpha=0.2, hatch='//')
                    ax_s.axvspan(750, 1100, color='#0284c7', alpha=0.15, label='VNIR Divergence Zone')
                    
                    ax_s.plot(wavelengths, oil_mean, color='#f43f5e', linewidth=2.0, label='Oil Slick Signature')
                    ax_s.fill_between(wavelengths, oil_mean - oil_std, oil_mean + oil_std, color='#f43f5e', alpha=0.12)
                    
                    ax_s.plot(wavelengths, water_mean, color='#38bdf8', linewidth=2.0, label='Seawater Baseline')
                    ax_s.fill_between(wavelengths, water_mean - water_std, water_mean + water_std, color='#38bdf8', alpha=0.12)
                    
                    max_diff_idx = np.argmax(np.abs(oil_mean - water_mean))
                    peak_wave = wavelengths[max_diff_idx]
                    peak_val_oil = oil_mean[max_diff_idx]
                    ax_s.annotate(
                        f'Peak Divergence\n({peak_wave:.0f} nm)',
                        xy=(peak_wave, peak_val_oil),
                        xytext=(peak_wave + 60, peak_val_oil + 0.12 * max(oil_mean)),
                        arrowprops=dict(facecolor='#38bdf8', shrink=0.08, width=1.2, headwidth=5),
                        color='#e2e8f0', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', edgecolor='#38bdf8', alpha=0.9)
                    )
                    
                    ax_s.set_title("Wavelength Reflectance Analysis (400 - 2500 nm)", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
                    ax_s.set_xlabel("Wavelength (nm)", color='#94a3b8', fontsize=9)
                    ax_s.set_ylabel("Radiance Intensity", color='#94a3b8', fontsize=9)
                    ax_s.tick_params(colors='#94a3b8', labelsize=8)
                    ax_s.grid(color='#334155', linestyle='--', alpha=0.3)
                    
                    legend = ax_s.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8, loc='upper right')
                    for text in legend.get_texts(): text.set_color('#e2e8f0')
                        
                    fig_s.tight_layout()
                    st.pyplot(fig_s, width='stretch')
                else:
                    st.info("Insufficient water background to plot a comparative graph.")
            else:
                st.info("No significant oil clusters detected to generate a target signature.")
                
        with g2:
            # PCA Variance Distribution Chart
            if processor_hsi is not None and hasattr(processor_hsi.pca, 'explained_variance_ratio_'):
                var_ratio = processor_hsi.pca.explained_variance_ratio_ * 100
                fig_p, ax_p = plt.subplots(figsize=(7, 4.2), facecolor='#0f172a')
                ax_p.set_facecolor('#0f172a')
                
                bands_idx = np.arange(1, len(var_ratio) + 1)
                ax_p.bar(bands_idx, var_ratio, color='#38bdf8', alpha=0.7, label='Band Variance %')
                ax_p.plot(bands_idx, np.cumsum(var_ratio), color='#2dd4bf', marker='o', markersize=3, linewidth=1.5, label='Cumulative Variance')
                
                ax_p.set_title("PCA Component Energy Distribution", color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
                ax_p.set_xlabel("PCA Principal Component", color='#94a3b8', fontsize=9)
                ax_p.set_ylabel("Variance Explained (%)", color='#94a3b8', fontsize=9)
                ax_p.tick_params(colors='#94a3b8', labelsize=8)
                ax_p.grid(color='#334155', linestyle='--', alpha=0.3)
                
                legend_p = ax_p.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8, loc='center right')
                for text in legend_p.get_texts(): text.set_color('#e2e8f0')
                
                fig_p.tight_layout()
                st.pyplot(fig_p, width='stretch')
            else:
                st.info("PCA model variance details unavailable.")

st.divider()
st.caption("CAFE-Fusion Intelligence Suite | Multimodal Marine Oil Spill Detection | SAR + HSI + Reliability Gating + MC Dropout")

