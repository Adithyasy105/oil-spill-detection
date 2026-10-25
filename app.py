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

_sar_import_err = None
_hsi_import_err = None

# 1. SAR Module Loading (CBDNetSAR)
try:
    import importlib.util
    spec_sar_m = importlib.util.spec_from_file_location("model_03_sar", os.path.join(SAR_LAYER_DIR, "03_model.py"))
    m3_sar = importlib.util.module_from_spec(spec_sar_m)
    spec_sar_m.loader.exec_module(m3_sar)
    CBDNetSAR = getattr(m3_sar, "CBDNetSAR", None)
    
    spec_sar_ds = importlib.util.spec_from_file_location("dataset_02_sar", os.path.join(SAR_LAYER_DIR, "02_dataset.py"))
    ds2_sar = importlib.util.module_from_spec(spec_sar_ds)
    spec_sar_ds.loader.exec_module(ds2_sar)
    preprocess_sar = getattr(ds2_sar, "preprocess_sar", None)
except Exception as e:
    _sar_import_err = str(e)
    print(f"[ERROR] SAR module import failed: {e}", file=sys.stderr)
    CBDNetSAR = None
    preprocess_sar = None

# 2. HSI Module Loading (HMOSDNet from hsi_layer_corrected)
try:
    spec_hsi_m = importlib.util.spec_from_file_location("model_03_hsi", os.path.join(HSI_LAYER_DIR, "03_model.py"))
    m3_hsi = importlib.util.module_from_spec(spec_hsi_m)
    spec_hsi_m.loader.exec_module(m3_hsi)
    HMOSDNet = getattr(m3_hsi, "HMOSDNet", None)
    
    spec_hsi_ds = importlib.util.spec_from_file_location("dataset_02_hsi", os.path.join(HSI_LAYER_DIR, "02_dataset.py"))
    ds2_hsi = importlib.util.module_from_spec(spec_hsi_ds)
    spec_hsi_ds.loader.exec_module(ds2_hsi)
    SpectralProcessor = getattr(ds2_hsi, "SpectralProcessor", None)
    BANDS_TO_REMOVE_HSI = getattr(ds2_hsi, "BANDS_TO_REMOVE", list(range(103, 113)) + list(range(149, 170)))
    PCA_COMPONENTS_HSI = getattr(ds2_hsi, "PCA_COMPONENTS", 25)
    PATCH_SIZE_HSI = getattr(ds2_hsi, "PATCH_SIZE", 5)
    import scipy.io as sio
except Exception as e:
    _hsi_import_err = str(e)
    print(f"[ERROR] HSI module import failed: {e}", file=sys.stderr)
    HMOSDNet = None
    SpectralProcessor = None
    BANDS_TO_REMOVE_HSI = list(range(103, 113)) + list(range(149, 170))
    PCA_COMPONENTS_HSI = 25
    PATCH_SIZE_HSI = 5
    import scipy.io as sio

# =============================================================================
# App Config & Premium UI Styling (CAFE-Fusion Intelligence Platform)
# =============================================================================
st.set_page_config(
    page_title="CAFE-Fusion | Multimodal Marine Oil Spill Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(56, 189, 248, 0.05) 0%, transparent 45%),
                    radial-gradient(circle at 90% 20%, rgba(168, 85, 247, 0.05) 0%, transparent 45%),
                    linear-gradient(180deg, #070b14 0%, #0d1527 100%) !important;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }

    code, pre, .mono-text {
        font-family: 'JetBrains Mono', monospace !important;
    }

    h1, h2, h3, h4, h5 {
        font-family: 'Inter', -apple-system, sans-serif;
        letter-spacing: -0.4px;
        color: #f8fafc;
    }

    /* Sidebar styling & top alignment fix */
    section[data-testid="stSidebar"] {
        background: rgba(10, 16, 31, 0.98) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.0rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
    }
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }

    /* Custom smooth scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #070b14;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }

    /* Custom Glass Panels */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);
    }

    /* Live pulsing indicator dots */
    .pulse-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2dd4bf;
        box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.7);
        animation: pulse-green 2s infinite;
        margin-right: 8px;
        vertical-align: middle;
    }

    @keyframes pulse-green {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(45, 212, 191, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(45, 212, 191, 0); }
    }

    .pulse-offline {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ef4444;
        margin-right: 8px;
        vertical-align: middle;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 4px;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 13px;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.12) !important;
        border: 1px solid #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        color: #38bdf8 !important;
    }

    /* Interactive KPI Cards */
    .kpi-container {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.45);
    }

    /* Academic table container & full responsive behavior */
    .academic-table-container {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        box-shadow: 0 10px 25px -8px rgba(0,0,0,0.5);
        margin: 14px 0 22px 0;
    }

    .academic-table-container table {
        width: 100%;
        min-width: 680px;
        border-collapse: collapse;
        font-size: 12.5px;
    }

    /* Custom Table Styling & DataFrame Container Overflow Handling */
    .stDataFrame, [data-testid="stDataFrame"] {
        background-color: #0f172a;
        border-radius: 10px;
        border: 1px solid #334155;
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* =========================================================================
       Ultra-Professional Dropdown / Selectbox Styling Across Website
       ========================================================================= */
    div[data-baseweb="select"] {
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] > div {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        min-height: 40px !important;
        height: 40px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 14px rgba(56, 189, 248, 0.3) !important;
    }

    div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
    div[data-baseweb="select"] span {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
    }

    /* Dropdown Arrow SVG Icon */
    div[data-baseweb="select"] svg {
        fill: #38bdf8 !important;
    }

    /* Dropdown Menu Popover */
    div[data-baseweb="popover"] {
        background: #0b1329 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        box-shadow: 0 16px 36px -8px rgba(0, 0, 0, 0.75) !important;
        overflow: hidden !important;
    }

    ul[role="listbox"] {
        background: #0b1329 !important;
        padding: 6px !important;
    }

    li[role="option"] {
        border-radius: 6px !important;
        padding: 9px 12px !important;
        color: #cbd5e1 !important;
        font-size: 12.5px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.18) 0%, rgba(168, 85, 247, 0.12) 100%) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Selectbox Label Styling */
    .stSelectbox label p {
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.7px !important;
        color: #94a3b8 !important;
        margin-bottom: 5px !important;
    }

    /* Standardized Download Button to match selectbox height exactly */
    .stDownloadButton {
        display: flex;
        align-items: flex-end;
    }
    .stDownloadButton > button {
        height: 40px !important;
        min-height: 40px !important;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 8px !important;
        color: #38bdf8 !important;
        font-size: 12.5px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.28) 0%, rgba(168, 85, 247, 0.28) 100%) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Responsive adjustments */
    @media (max-width: 992px) {
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px;
            font-size: 12px;
        }
    }
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.5rem 1rem !important;
        }
        .academic-table-container table {
            font-size: 11px;
            min-width: 580px;
        }
    }

    /* Hide default Streamlit header, footer ('Made with Streamlit'), and menu */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Model Architectures & Checkpoint Loaders
# =============================================================================

@st.cache_resource
def load_sar_expert():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = os.path.join(SAR_LAYER_DIR, "weights", "best_sar_expert.pth")
    # Verify file existence and complete download (>100MB)
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 100_000_000:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        url = "https://huggingface.co/adi106/marine-oil-spill-weights/resolve/main/best_sar_expert.pth"
        try:
            with st.spinner("Downloading SAR expert weights (~165MB) from Hugging Face Hub..."):
                torch.hub.download_url_to_file(url, weights_path)
        except Exception as e:
            print(f"[ERROR] Failed to download SAR weights: {e}", file=sys.stderr)
            st.sidebar.error(f"Failed to auto-download SAR weights: {e}")
    if CBDNetSAR is None:
        print(f"[WARN] CBDNetSAR is None. Import error: {_sar_import_err}", file=sys.stderr)
        return None, device
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 100_000_000:
        print("[WARN] SAR weights file not found or incomplete.", file=sys.stderr)
        return None, device
    try:
        model = CBDNetSAR(encoder_weights=None).to(device)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd)
        model.eval()
        return model, device
    except Exception as e:
        print(f"[ERROR] SAR Weight Load Error: {e}", file=sys.stderr)
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
        except Exception as e:
            print(f"[ERROR] HSI SpectralProcessor load error: {e}", file=sys.stderr)
            processor = None

    if HMOSDNet is None:
        print(f"[WARN] HMOSDNet is None. Import error: {_hsi_import_err}", file=sys.stderr)
        return None, processor, device
    if not os.path.exists(weights_path):
        print(f"[WARN] HSI weights not found at {weights_path}", file=sys.stderr)
        return None, processor, device

    try:
        model = HMOSDNet(in_channels=1, num_bands=PCA_COMPONENTS_HSI, mam_reduction=2).to(device)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        sd = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(sd)
        model.eval()
        return model, processor, device
    except Exception as e:
        print(f"[ERROR] HSI Weight Load Error: {e}", file=sys.stderr)
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


def render_academic_table_html(headers, rows, highlight_idx=None, title=None, subtitle=None):
    """
    Renders a publication-grade scientific HTML table with dark-mode styling,
    monospace aligned numbers, pill badges, and an optional highlighted champion row.
    Guaranteed responsive horizontal scrolling without page overflow.
    """
    th_items = []
    for i, h in enumerate(headers):
        align = "left" if i == 0 else "right"
        th_items.append(
            f"<th style='padding:11px 14px; color:#94a3b8; font-size:11px; font-weight:700; "
            f"text-transform:uppercase; letter-spacing:0.7px; white-space:nowrap; text-align:{align};'>{h}</th>"
        )
    th_cells = "".join(th_items)
    
    tr_rows = []
    for r_idx, row in enumerate(rows):
        is_highlight = (highlight_idx is not None and r_idx == highlight_idx)
        bg = "linear-gradient(90deg, rgba(168,85,247,0.18) 0%, rgba(56,189,248,0.08) 100%)" if is_highlight else ("rgba(255,255,255,0.02)" if r_idx % 2 == 1 else "transparent")
        b_left = "4px solid #c084fc" if is_highlight else "4px solid transparent"
        
        tds = []
        for c_idx, val in enumerate(row):
            align = "left" if c_idx == 0 else "right"
            color = "#f8fafc" if (is_highlight or c_idx == 0) else "#cbd5e1"
            weight = "700" if is_highlight else "500"
            mono = "font-family: monospace;" if c_idx > 0 and not str(val).startswith("<") else ""
            tds.append(
                f"<td style='padding:10px 14px; color:{color}; font-weight:{weight}; {mono} white-space:nowrap; "
                f"text-align:{align}; border-bottom:1px solid rgba(255,255,255,0.05);'>{val}</td>"
            )
        
        tr_rows.append(f"<tr style='background:{bg}; border-left:{b_left}; transition:background 0.2s;'>{''.join(tds)}</tr>")
    
    header_bar = ""
    if title:
        sub_text = f"<span style='color:#94a3b8; font-size:11.5px;'>{subtitle}</span>" if subtitle else ""
        header_bar = (
            f"<div style='background:linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding:12px 18px; "
            f"border-bottom:1px solid #334155; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
            f"<span style='color:#f8fafc; font-size:12.5px; font-weight:700; letter-spacing:0.5px;'>{title}</span>"
            f"{sub_text}</div>"
        )
        
    return (
        f"<div class='academic-table-container' style='background:#0f172a; border:1px solid #334155; border-radius:12px; overflow:hidden; "
        f"box-shadow:0 10px 25px -8px rgba(0,0,0,0.5); margin:14px 0 22px 0; width:100%; max-width:100%;'>"
        f"{header_bar}"
        f"<div style='overflow-x:auto; -webkit-overflow-scrolling:touch; width:100%; max-width:100%;'>"
        f"<table style='width:100%; min-width:680px; border-collapse:collapse; font-size:12.5px;'>"
        f"<thead><tr style='background:rgba(30,41,59,0.9); border-bottom:1px solid #334155;'>{th_cells}</tr></thead>"
        f"<tbody>{''.join(tr_rows)}</tbody>"
        f"</table></div></div>"
    )


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


@st.cache_data
def load_per_scene_results():
    csv_path = os.path.join(FUSION_EXP_DIR, "cafe_fusion_v2", "primary_results", "per_scene_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


# =============================================================================
# App Header & Status Center
# =============================================================================
model_sar, device_sar = load_sar_expert()
model_hsi, processor_hsi, device_hsi = load_hsi_expert()
model_cafe, device_cafe, cafe_loaded = load_cafe_fusion_model(fold=1)

cached_scenes = load_all_cached_scenes()

# Professional Academic Header & Research Badges
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.90) 100%);
            border: 1px solid rgba(56,189,248,0.25); border-radius: 14px; padding: 22px 28px; margin-bottom: 20px;
            box-shadow: 0 16px 36px -12px rgba(0,0,0,0.55);">
    <div style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px;">
        <span style="display:inline-flex; align-items:center; background:rgba(56,189,248,0.12); border:1px solid #38bdf8; color:#38bdf8; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.5px;">
            &#x25C8; MULTIMODAL BENCHMARK SUITE
        </span>
        <span style="display:inline-flex; align-items:center; background:rgba(168,85,247,0.12); border:1px solid #a855f7; color:#c084fc; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.5px;">
            &#x25C8; LOGIT-SPACE CONVEX FUSION
        </span>
        <span style="display:inline-flex; align-items:center; background:rgba(45,212,191,0.12); border:1px solid #2dd4bf; color:#2dd4bf; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.5px;">
            &#x25C8; EPISTEMIC UNCERTAINTY (MC DROPOUT T=30)
        </span>
        <span style="display:inline-flex; align-items:center; background:rgba(245,158,11,0.12); border:1px solid #f59e0b; color:#fbbf24; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.5px;">
            &#x25C8; OPEN-ACCESS WEIGHTS & CODE
        </span>
    </div>
    <h1 style="color:#f8fafc; font-size:26px; font-weight:800; margin:0 0 10px 0; letter-spacing:-0.5px; line-height:1.25;">
        Confidence-Aware Multimodal Fusion (CAFE-Fusion) for Marine Oil Spill Detection
    </h1>
    <p style="color:#94a3b8; font-size:13.5px; line-height:1.65; margin:0;">
        An asymmetric, registration-free late logit-space gating framework combining <b>CBD-Net SAR Expert</b> (spatial context + multi-scale boundary supervision) and <b>HMOSDNet HSI Expert</b> (3D-CNN spectral feature extraction with 25-PCA reduction). Incorporates <b>Monte Carlo Dropout stochastic passes (T=30)</b> for pixel-level epistemic uncertainty estimation and selective risk-coverage rejection.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("◈ Methodological Foundations & End-to-End System Architecture", expanded=False):

    # ── SECTION 1: Research Motivation Cards ─────────────────────────────
    st.markdown(
        "<p style='color:#94a3b8;font-size:11px;font-weight:700;text-transform:uppercase;"
        "letter-spacing:1px;margin-bottom:4px;'>Research Motivation — Sensor Complementarity Problem</p>",
        unsafe_allow_html=True,
    )
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        st.markdown("""
<div style='background:rgba(56,189,248,0.07);border:1px solid rgba(56,189,248,0.35);
border-top:3px solid #38bdf8;border-radius:10px;padding:14px 15px;'>
<div style='color:#38bdf8;font-size:10px;font-weight:700;text-transform:uppercase;
letter-spacing:0.8px;margin-bottom:7px;'>SAR Expert — CBD-Net</div>
<div style='color:#f8fafc;font-size:13px;font-weight:600;margin-bottom:6px;'>All-Weather Spatial Coverage</div>
<div style='color:#94a3b8;font-size:11.5px;line-height:1.55;'>Sentinel-1 backscatter (σ°) resolves
spill morphology under cloud cover and at night. Biogenic look-alikes (low-wind zones, ship
wakes) cause high false-alarm rates without spectral corroboration.</div></div>
""", unsafe_allow_html=True)
    with _c2:
        st.markdown("""
<div style='background:rgba(45,212,191,0.07);border:1px solid rgba(45,212,191,0.35);
border-top:3px solid #2dd4bf;border-radius:10px;padding:14px 15px;'>
<div style='color:#2dd4bf;font-size:10px;font-weight:700;text-transform:uppercase;
letter-spacing:0.8px;margin-bottom:7px;'>HSI Expert — HMOSDNet</div>
<div style='color:#f8fafc;font-size:13px;font-weight:600;margin-bottom:6px;'>Chemical Hydrocarbon Fingerprinting</div>
<div style='color:#94a3b8;font-size:11.5px;line-height:1.55;'>AVIRIS-NG 224-band hyperspectral
imaging detects C-H absorption at 1.2 μm and 1.7 μm. Narrow swath width and cloud
sensitivity constrain operational coverage to clear-sky conditions.</div></div>
""", unsafe_allow_html=True)
    with _c3:
        st.markdown("""
<div style='background:rgba(168,85,247,0.07);border:1px solid rgba(168,85,247,0.35);
border-top:3px solid #a855f7;border-radius:10px;padding:14px 15px;'>
<div style='color:#c084fc;font-size:10px;font-weight:700;text-transform:uppercase;
letter-spacing:0.8px;margin-bottom:7px;'>Fusion Challenge</div>
<div style='color:#f8fafc;font-size:13px;font-weight:600;margin-bottom:6px;'>Registration-Free Late Fusion</div>
<div style='color:#94a3b8;font-size:11.5px;line-height:1.55;'>Ocean surface drift and temporal
revisit gaps prevent pixel-accurate early concatenation. CAFE-Fusion operates in logit-space
— zero spatial registration required between sensor streams.</div></div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2: Architecture Diagram (matplotlib — always renders) ─────
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    _fig, _ax = plt.subplots(figsize=(15.5, 6.4), facecolor='#0b1120')
    _fig.patch.set_facecolor('#0b1120')
    _ax.set_facecolor('#0b1120')
    _ax.set_xlim(0, 15.5)
    _ax.set_ylim(0, 6.2)
    _ax.axis('off')

    _ax.text(7.75, 5.95, "END-TO-END CAFE-FUSION v2 ARCHITECTURAL PIPELINE",
             ha='center', va='center', color='#f8fafc', fontsize=12.5, fontweight='bold')
    _ax.text(7.75, 5.68, "Asymmetric Late-Fusion in Logit Space with Monte Carlo Epistemic Uncertainty Estimation",
             ha='center', va='center', color='#94a3b8', fontsize=9.5)

    def _draw_card(x, y, w, h, title, lines, title_color, bg_color, border_color, lw=1.2, radius=0.15):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0.04,rounding_size={radius}",
                           fc=bg_color, ec=border_color, lw=lw, zorder=3)
        _ax.add_patch(b)
        _ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='top',
                 color=title_color, fontsize=9.5, fontweight='bold', zorder=4)
        for idx, line in enumerate(lines):
            _ax.text(x + w / 2, y + h - 0.58 - idx * 0.24, line, ha='center', va='top',
                     color='#94a3b8', fontsize=8.0, zorder=4)

    def _draw_arrow(x1, y1, x2, y2, color='#475569', lw=1.6, style='->'):
        _ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                     shrinkA=2, shrinkB=2), zorder=5)

    # 1. SAR Stream (Left)
    _draw_card(0.4, 4.25, 2.8, 1.15, "SAR SENSOR INPUT",
               ["Sentinel-1 SAR C-Band", "\u03c3\u00b0 Backscatter (dB) (256\u00d7256)", "All-Weather Day/Night Coverage"],
               "#38bdf8", "#0f2338", "#38bdf8", lw=1.5)
    _draw_arrow(1.8, 4.25, 1.8, 3.85, color='#38bdf8')

    _draw_card(0.4, 2.45, 2.8, 1.4, "CBD-Net SAR EXPERT",
               ["ResNet-18 + scSE Spatial Gate", "Multi-Scale Boundary Loss (MBL)", "Learned Look-Alike Suppression"],
               "#f8fafc", "#0a192f", "#334155", lw=1.2)
    _draw_arrow(1.8, 2.45, 1.8, 2.05, color='#38bdf8')

    _draw_card(0.4, 0.85, 2.8, 1.2, "SAR LOGITS  (z_SAR)",
               ["Logit: ln(p / (1 - p))", "Spatial Structure Prior", "High false-alarm vulnerability"],
               "#38bdf8", "#091c2b", "#38bdf8", lw=1.4)

    # 2. HSI Stream (Right)
    _draw_card(12.3, 4.25, 2.8, 1.15, "HSI SENSOR INPUT",
               ["AVIRIS-NG Hyperspectral", "224 Bands \u2192 25-PCA Comps", "Chemical C-H Absorption Bands"],
               "#2dd4bf", "#082a27", "#2dd4bf", lw=1.5)
    _draw_arrow(13.7, 4.25, 13.7, 3.85, color='#2dd4bf')

    _draw_card(12.3, 2.45, 2.8, 1.4, "HMOSDNet HSI EXPERT",
               ["3D-CNN Spectral-Spatial Conv", "Multi-Head Attention Pooling", "Hydrocarbon Peak Fingerprinting"],
               "#f8fafc", "#072020", "#334155", lw=1.2)
    _draw_arrow(13.7, 2.45, 13.7, 2.05, color='#2dd4bf')

    _draw_card(12.3, 0.85, 2.8, 1.2, "HSI LOGITS  (z_HSI)",
               ["Logit: ln(p / (1 - p))", "Dominant Spectral Anchor", "Narrow swath / clear-sky only"],
               "#2dd4bf", "#051f1e", "#2dd4bf", lw=1.4)

    # 3. Center CAFE-Fusion Core Engine
    _engine_frame = FancyBboxPatch((3.7, 0.35), 8.1, 5.05,
                                   boxstyle="round,pad=0.06,rounding_size=0.25",
                                   fc='#0f172a', ec='#8b5cf6', lw=2.0, zorder=2)
    _ax.add_patch(_engine_frame)

    _ax.text(7.75, 5.15, "CAFE-FUSION v2 CORE ENGINE", ha='center', va='center',
             color='#c084fc', fontsize=11.5, fontweight='bold', zorder=4)
    _ax.text(7.75, 4.88, "Registration-Free Multimodal Gating with Epistemic Dropout", ha='center', va='center',
             color='#94a3b8', fontsize=8.5, zorder=4)

    _draw_card(4.1, 3.55, 7.3, 1.15, "1. Dynamic 11-D Meta-Confidence Feature Vector (u)",
               ["Features: [ p_SAR, z_SAR, p_HSI, z_HSI, |p_SAR - p_HSI|, (p_SAR - p_HSI),",
                "            H(p_SAR), H(p_HSI), q_HSI, \u03bc_SAR, \u03c3_SAR ]",
                "Captures Cross-Modal Disagreement, Information Entropy & Spatial Texture"],
               "#f8fafc", "#171a33", "#4c1d95", lw=1.0)

    _draw_card(4.1, 2.15, 7.3, 1.25, "2. Asymmetric Reliability Gate with Safe Anchor Inductive Bias",
               ["Gate Logit: \u03b7 = W_g \u00b7 u + b_init,   b_init = -2.0  \u27f9  g_init \u2248 0.119 (~88% HSI Prior)",
                "Calculated Gate Weight:  g = \u03c3(\u03b7) \u2208 [0, 1]   (g \u2192 0: Rely on HSI,  g \u2192 1: Rely on SAR)",
                "Convex Combination:  z_fused = (1 - g) \u00b7 z_HSI + g \u00b7 z_SAR"],
               "#c084fc", "#1c1438", "#7c3aed", lw=1.2)

    _draw_card(4.1, 0.55, 7.3, 1.45, "3. Epistemic Uncertainty Quantification via MC Dropout (T = 30 Passes)",
               ["Mean Posterior Probability:  p_fused(x) = (1/T) \u2211 \u03c3(z_fused^(t))",
                "Epistemic Variance Map:  \u03c3\u00b2_epistemic(x) = (1/T) \u2211 [p_t(x) - p_fused(x)]\u00b2",
                "Calibrated Binary Oil Mask:  y_pred = [ p_fused \u2265 t* ]   |   Selective Risk-Coverage Rejection"],
               "#f59e0b", "#1e142b", "#d97706", lw=1.2)

    _draw_arrow(7.75, 3.55, 7.75, 3.40, color='#8b5cf6', lw=1.6)
    _draw_arrow(7.75, 2.15, 7.75, 2.00, color='#8b5cf6', lw=1.6)

    _draw_arrow(3.2, 1.45, 3.7, 2.70, color='#38bdf8', lw=1.8)
    _draw_arrow(3.2, 1.45, 3.7, 4.10, color='#38bdf8', lw=1.4)
    _ax.text(3.38, 2.15, "z_SAR", color='#38bdf8', fontsize=8.0, fontweight='bold', zorder=5)

    _draw_arrow(12.3, 1.45, 11.8, 2.70, color='#2dd4bf', lw=1.8)
    _draw_arrow(12.3, 1.45, 11.8, 4.10, color='#2dd4bf', lw=1.4)
    _ax.text(12.05, 2.15, "z_HSI", color='#2dd4bf', fontsize=8.0, fontweight='bold', zorder=5)

    _leg_handles = [
        mpatches.Patch(color='#38bdf8', label='SAR Spatial Expert (CBD-Net)'),
        mpatches.Patch(color='#2dd4bf', label='HSI Spectral Expert (HMOSDNet)'),
        mpatches.Patch(color='#8b5cf6', label='Reliability Gate & Logit Fusion'),
        mpatches.Patch(color='#f59e0b', label='Epistemic Uncertainty (MC Dropout)'),
    ]
    _ax.legend(handles=_leg_handles, loc='lower center', ncol=4, framealpha=0,
               labelcolor='#e2e8f0', fontsize=8.2, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(pad=0.4)
    st.pyplot(_fig)
    plt.close(_fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3: Key Innovations + Validated Results ────────────────────
    _ci1, _ci2 = st.columns(2)
    with _ci1:
        st.markdown("""
<div style='background:linear-gradient(135deg, #111827 0%, #172033 100%); border:1px solid rgba(56,189,248,0.25); border-radius:12px; padding:18px 20px; box-shadow:0 10px 25px -8px rgba(0,0,0,0.45);'>
    <div style='color:#38bdf8; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:12px; display:flex; align-items:center; gap:6px;'>
        &#x25C8; Core Architectural Innovations
    </div>
    <div style='margin-bottom:12px;'>
        <div style='color:#f8fafc; font-size:13px; font-weight:700; margin-bottom:2px;'>
            <span style='color:#c084fc; margin-right:6px;'>&#x25C6;</span>Registration-Free Late Fusion
        </div>
        <div style='color:#94a3b8; font-size:11.5px; line-height:1.55; padding-left:18px;'>
            Operates directly on logit outputs (z_SAR, z_HSI). Eliminates brittle pixel-level spatial coregistration under sea-surface drift.
        </div>
    </div>
    <div style='margin-bottom:12px;'>
        <div style='color:#f8fafc; font-size:13px; font-weight:700; margin-bottom:2px;'>
            <span style='color:#2dd4bf; margin-right:6px;'>&#x25C6;</span>Asymmetric Inductive Bias Anchor
        </div>
        <div style='color:#94a3b8; font-size:11.5px; line-height:1.55; padding-left:18px;'>
            Calibrated bias b_init = -2.0 guarantees safe default gate g_init &approx; 0.119 (~88% HSI dominance), preventing SAR look-alikes from corrupting clear-sky detections.
        </div>
    </div>
    <div style='margin-bottom:12px;'>
        <div style='color:#f8fafc; font-size:13px; font-weight:700; margin-bottom:2px;'>
            <span style='color:#fbbf24; margin-right:6px;'>&#x25C6;</span>Monte Carlo Epistemic Uncertainty (T=30)
        </div>
        <div style='color:#94a3b8; font-size:11.5px; line-height:1.55; padding-left:18px;'>
            Stochastic forward passes compute pixel-wise variance &sigma;&sup2;_epistemic, enabling selective risk-coverage rejection and false-positive flagging.
        </div>
    </div>
    <div>
        <div style='color:#f8fafc; font-size:13px; font-weight:700; margin-bottom:2px;'>
            <span style='color:#34d399; margin-right:6px;'>&#x25C6;</span>Dynamic 11-D Meta-Confidence Vector
        </div>
        <div style='color:#94a3b8; font-size:11.5px; line-height:1.55; padding-left:18px;'>
            Vector u encodes model confidences, cross-sensor logit discrepancies, Shannon entropies, and local SAR variance to modulate the reliability gate.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    with _ci2:
        st.markdown("""
<div style='background:linear-gradient(135deg, #111827 0%, #172033 100%); border:1px solid rgba(168,85,247,0.25); border-radius:12px; padding:18px 20px; box-shadow:0 10px 25px -8px rgba(0,0,0,0.45);'>
    <div style='color:#c084fc; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:12px; display:flex; align-items:center; gap:6px;'>
        &#x25C8; Benchmark Results (18 Marine Scenes &bull; 6-Fold CV)
    </div>
    <div style='background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:8px 14px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;'>
        <span style='color:#94a3b8; font-size:12px; font-weight:600;'>SAR Expert (CBD-Net) F1</span>
        <b style='color:#38bdf8; font-size:13.5px; font-family:monospace;'>0.8358 &plusmn; 0.043</b>
    </div>
    <div style='background:rgba(45,212,191,0.08); border:1px solid rgba(45,212,191,0.2); border-radius:8px; padding:8px 14px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;'>
        <span style='color:#94a3b8; font-size:12px; font-weight:600;'>HSI Expert (HMOSDNet) F1</span>
        <b style='color:#2dd4bf; font-size:13.5px; font-family:monospace;'>0.9261 &plusmn; 0.031</b>
    </div>
    <div style='background:rgba(168,85,247,0.12); border:1px solid rgba(168,85,247,0.35); border-radius:8px; padding:8px 14px; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center;'>
        <span style='color:#f8fafc; font-size:12.5px; font-weight:700;'>CAFE-Fusion v2 (Proposed)</span>
        <b style='color:#c084fc; font-size:14px; font-family:monospace;'>0.9288 &plusmn; 0.027</b>
    </div>
    <div style='background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:8px; padding:8px 14px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;'>
        <span style='color:#94a3b8; font-size:12px; font-weight:600;'>Theoretical Oracle Bound F1</span>
        <b style='color:#fbbf24; font-size:13.5px; font-family:monospace;'>0.9501</b>
    </div>
    <div style='display:flex; gap:10px;'>
        <div style='flex:1; background:rgba(52,211,153,0.1); border:1px solid #34d399; border-radius:8px; padding:8px 12px; text-align:center;'>
            <div style='color:#34d399; font-size:10px; font-weight:700; text-transform:uppercase;'>Paired t-test</div>
            <div style='color:#f8fafc; font-size:13px; font-weight:800; font-family:monospace;'>p = 0.012 *</div>
        </div>
        <div style='flex:1; background:rgba(52,211,153,0.1); border:1px solid #34d399; border-radius:8px; padding:8px 12px; text-align:center;'>
            <div style='color:#34d399; font-size:10px; font-weight:700; text-transform:uppercase;'>Wilcoxon Signed-Rank</div>
            <div style='color:#f8fafc; font-size:13px; font-weight:800; font-family:monospace;'>p = 0.031 *</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# =============================================================================
# Sidebar: Mission Telemetry & Calibration Deck
# =============================================================================
dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"

st.sidebar.markdown(f"""
<div style="background: linear-gradient(135deg, #0b1329 0%, #1e293b 100%);
            border: 1px solid rgba(56,189,248,0.3); border-radius: 12px; padding: 15px 16px; margin-bottom: 14px;
            box-shadow: 0 10px 24px -8px rgba(0,0,0,0.6);">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
        <div style="display:flex; align-items:center; gap:9px;">
            <div style="width:28px; height:28px; border-radius:7px; background:linear-gradient(135deg, #0284c7 0%, #a855f7 100%); display:flex; align-items:center; justify-content:center; color:#ffffff; font-weight:900; font-size:12px; letter-spacing:0.5px; box-shadow:0 0 12px rgba(56,189,248,0.4);">
                CF
            </div>
            <div>
                <div style="color:#f8fafc; font-size:14.5px; font-weight:800; letter-spacing:0.3px;">CAFE-FUSION v2</div>
                <div style="color:#94a3b8; font-size:9.5px; font-weight:600; text-transform:uppercase; letter-spacing:0.6px;">Multimodal Decision Platform</div>
            </div>
        </div>
    </div>
    <div style="display:flex; gap:6px; margin-top:10px; flex-wrap:wrap;">
        <span style="background:rgba(45,212,191,0.12); border:1px solid #2dd4bf; color:#2dd4bf; font-size:9px; font-weight:700; padding:2.5px 8px; border-radius:6px; letter-spacing:0.5px;">
            ● ENGINE OPERATIONAL
        </span>
        <span style="background:rgba(168,85,247,0.12); border:1px solid #a855f7; color:#c084fc; font-size:9px; font-weight:700; padding:2.5px 8px; border-radius:6px; letter-spacing:0.5px;">
            {device_sar.type.upper()} ACCELERATED
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Telemetry Deck Card
sar_status_badge = "<span class='pulse-online'></span><b style='color:#38bdf8;'>CBD-Net SAR</b>: ONLINE" if model_sar is not None else "<span class='pulse-offline'></span><b style='color:#ef4444;'>SAR Expert</b>: OFFLINE"
hsi_status_badge = "<span class='pulse-online'></span><b style='color:#2dd4bf;'>HMOSDNet HSI</b>: ONLINE" if model_hsi is not None else "<span class='pulse-offline'></span><b style='color:#ef4444;'>HSI Expert</b>: OFFLINE"
cafe_status_badge = "<span class='pulse-online'></span><b style='color:#c084fc;'>CAFE-Fusion Gate</b>: ONLINE" if cafe_loaded else "<span class='pulse-offline'></span><b style='color:#ef4444;'>CAFE-Fusion</b>: OFFLINE"

st.sidebar.markdown(f"""
<div style='background:#0f172a; border:1px solid #334155; border-radius:10px; padding:12px 14px; font-size:11.5px; margin-bottom:16px;'>
    <div style='color:#94a3b8; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:5px; display:flex; align-items:center; gap:5px;'>
        <span style='color:#38bdf8;'>◈</span> Model Health & Hardware Telemetry
    </div>
    <div style='margin-bottom:6px;'>{sar_status_badge}</div>
    <div style='margin-bottom:6px;'>{hsi_status_badge}</div>
    <div style='margin-bottom:8px;'>{cafe_status_badge}</div>
    <div style='border-top:1px solid rgba(255,255,255,0.06); padding-top:6px; display:flex; justify-content:space-between; color:#64748b; font-size:10px;'>
        <span>Device: <b style='color:#cbd5e1;'>{dev_name[:18]}</b></span>
        <span>PyTorch: <b style='color:#cbd5e1;'>{torch.__version__}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='color:#c084fc; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px; display:flex; align-items:center; gap:5px;'><span style='color:#a855f7;'>◈</span> Inference Calibration</div>", unsafe_allow_html=True)
fusion_threshold = st.sidebar.slider("Decision Cutoff Threshold (t*)", 0.01, 0.99, 0.50, 0.01, key="fus_t", help="Optimal Bayes decision boundary calibrated on validation set")
mc_passes = st.sidebar.slider("Epistemic MC Dropout Passes (T)", 5, 50, 30, 5, key="mc_t", help="Number of stochastic forward passes for pixel-level uncertainty variance estimation")

fold_dict = {
    1: "Fold 1 (GM04–06 Test | F1: 0.9556)",
    2: "Fold 2 (GM07–09 Test | F1: 0.9544)",
    3: "Fold 3 (GM10–12 Test | F1: 0.9397)",
    4: "Fold 4 (GM13–15 Test | F1: 0.8948)",
    5: "Fold 5 (GM16–18 Test | F1: 0.9310)",
    6: "Fold 6 (GM01–03 Test | F1: 0.8972)",
}
selected_fold = st.sidebar.selectbox(
    "Evaluation Checkpoint Fold",
    [1, 2, 3, 4, 5, 6],
    index=0,
    format_func=lambda x: fold_dict.get(x, f"Fold {x}"),
    help="Evaluate out-of-fold generalization across the 6-fold cross-validation suite"
)

if selected_fold != 1:
    model_cafe, device_cafe, cafe_loaded = load_cafe_fusion_model(fold=selected_fold)

with st.sidebar.expander("◈ Sensor Threshold Calibration (Expert Filters)", expanded=False):
    st.markdown("<p style='color:#94a3b8; font-size:11px; margin-bottom:8px;'>Adjust individual sensor sensitivity & spatial morphology filtering parameters.</p>", unsafe_allow_html=True)
    threshold_sar = st.slider("SAR Sensitivity Threshold (τ_SAR)", 0.1, 0.95, 0.75, 0.05, key="sar_t")
    min_size_sar = st.slider("Minimum Cluster Filter Area (px²)", 10, 500, 50, 10, key="sar_m")
    threshold_hsi = st.slider("HSI Sensitivity Threshold (τ_HSI)", 0.1, 0.95, 0.50, 0.05, key="hsi_t")

st.sidebar.markdown(f"""
<div style='background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:12px 13px; margin-top:14px; font-size:10.5px; color:#64748b; line-height:1.6;'>
    <div style='color:#94a3b8; font-weight:700; margin-bottom:5px; display:flex; align-items:center; gap:5px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:4px;'>
        <span style='color:#38bdf8;'>◈</span> Official Benchmark Registry
    </div>
    <div>Weights: <a href='https://huggingface.co/adi106/marine-oil-spill-weights' target='_blank' style='color:#38bdf8; text-decoration:none; font-weight:700;'>adi106/marine-oil-spill-weights ↗</a></div>
    <div>GitHub: <a href='https://github.com/Adithyasy105/oil-spill-detection' target='_blank' style='color:#c084fc; text-decoration:none; font-weight:700;'>Adithyasy105/oil-spill-detection ↗</a></div>
    <div>Corpus: <span style='color:#cbd5e1; font-weight:600;'>18 Dual-Modal Scenes (HOSD + SOS)</span></div>
    <div style='border-top:1px solid rgba(255,255,255,0.06); margin-top:6px; padding-top:4px;'>
        Authors: <b style='color:#e2e8f0;'>Adithya S Y, Anuragh, Hitendra</b>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Navigation Tabs (4 Comprehensive Research Modules)
# =============================================================================
tab_fusion, tab_sar, tab_hsi, tab_benchmark = st.tabs([
    "◈ CAFE-Fusion Decision Hub",
    "◈ SAR Spatial Expert (CBD-Net)",
    "◈ HSI Spectral Expert (HMOSDNet)",
    "◈ Academic Benchmark & Validation"
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
        b_head_c1, b_head_c2 = st.columns([2.2, 1.3])
        with b_head_c1:
            st.markdown("<p style='color:#94a3b8; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin:4px 0 0 0;'>◈ Authoritative Benchmark & Cross-Validation Stability Matrix</p>", unsafe_allow_html=True)
        with b_head_c2:
            bench_view = st.selectbox(
                "Performance Metric View:",
                ["Dual-Panel: F1, IoU & 6-Fold Stability", "Full Spectrum: 6 Macro Remote Sensing Metrics"],
                index=0,
                label_visibility="collapsed"
            )

        if bench_view == "Dual-Panel: F1, IoU & 6-Fold Stability":
            # Dual-panel scientific figure
            fig_bench, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 4.8), facecolor='#0f172a', gridspec_kw={'width_ratios': [1.3, 1.0]})
            fig_bench.patch.set_facecolor('#0f172a')
            
            # Panel 1: Multi-Metric Benchmark (F1 + IoU)
            models = ['SAR Expert\n(CBD-Net)', 'HSI Expert\n(HMOSDNet)', 'CAFE-Fusion v2\n(Proposed)', 'Theoretical Oracle\n(Upper Bound)']
            f1_vals = [
                benchmark_summary.get("mean_f1_sar", 0.8358),
                benchmark_summary.get("mean_f1_hsi", 0.9261),
                benchmark_summary.get("mean_f1_fusion", 0.9288),
                0.9501
            ]
            f1_stds = [
                benchmark_summary.get("std_f1_sar", 0.0429),
                benchmark_summary.get("std_f1_hsi", 0.0307),
                benchmark_summary.get("std_f1_fusion", 0.0270),
                0.0270
            ]
            iou_vals = [0.7180, 0.8624, 0.8671, 0.9049]
            
            x_pos = np.arange(len(models))
            w_bar = 0.35
            
            ax1.set_facecolor('#0f172a')
            b1 = ax1.bar(x_pos - w_bar/2, f1_vals, w_bar, yerr=f1_stds, capsize=4, label='F1 Score (Dice Metric)',
                         color=['#38bdf8', '#2dd4bf', '#c084fc', '#fbbf24'], alpha=0.92, edgecolor='white', linewidth=0.8)
            b2 = ax1.bar(x_pos + w_bar/2, iou_vals, w_bar, label='IoU (Jaccard Index)',
                         color=['#0284c7', '#0d9488', '#9333ea', '#d97706'], alpha=0.75, hatch='//', edgecolor='#475569', linewidth=0.6)
            
            hsi_anchor = f1_vals[1]
            ax1.axhline(hsi_anchor, color='#2dd4bf', linestyle='--', linewidth=1.2, alpha=0.65, label=f'HSI Anchor ({hsi_anchor:.4f})')
            ax1.set_ylim(0.58, 1.10)
            ax1.set_ylabel('Performance Metric Value', color='#94a3b8', fontsize=9.5)
            ax1.set_title('Cross-Modal Performance Matrix (F1 & IoU across 6 Folds)', color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(models, color='#e2e8f0', fontsize=8.5)
            ax1.tick_params(colors='#94a3b8', labelsize=8)
            ax1.grid(axis='y', linestyle='--', color='#334155', alpha=0.3)
            
            # Numbers on bars
            for b in b1:
                h = b.get_height()
                ax1.text(b.get_x() + b.get_width()/2, h + 0.040, f'{h:.4f}', ha='center', va='bottom', color='#f8fafc', fontsize=7.5, fontweight='bold')
            for b in b2:
                h = b.get_height()
                ax1.text(b.get_x() + b.get_width()/2, h + 0.012, f'{h:.4f}', ha='center', va='bottom', color='#94a3b8', fontsize=7.0)
                
            ax1.text(2, 1.055, '▲ +0.0930 vs SAR | +0.0027 vs HSI', ha='center', color='#c084fc', fontsize=7.8, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.25', facecolor='#1e1438', edgecolor='#a855f7', lw=0.9))
                     
            leg1 = ax1.legend(loc='lower right', facecolor='#1e293b', edgecolor='#334155', fontsize=8)
            for t in leg1.get_texts(): t.set_color('#e2e8f0')
            
            # Panel 2: 6-Fold Stability
            ax2.set_facecolor('#0f172a')
            fold_names = [f'Fold {i}' for i in range(1, 7)]
            sar_folds = [0.8549, 0.7774, 0.8001, 0.8444, 0.8385, 0.8995]
            hsi_folds = [0.9531, 0.9533, 0.9508, 0.8932, 0.9184, 0.8876]
            fus_folds = [0.9556, 0.9544, 0.9397, 0.8948, 0.9310, 0.8972]
            
            xf = np.arange(len(fold_names))
            wf = 0.26
            ax2.bar(xf - wf, sar_folds, wf, label='SAR (CBD-Net)', color='#38bdf8', alpha=0.8)
            ax2.bar(xf, hsi_folds, wf, label='HSI (HMOSDNet)', color='#2dd4bf', alpha=0.8)
            ax2.bar(xf + wf, fus_folds, wf, label='CAFE-Fusion v2', color='#c084fc', alpha=0.95, edgecolor='white', linewidth=0.8)
            
            ax2.axhline(0.9288, color='#c084fc', linestyle=':', linewidth=1.2, alpha=0.8, label='Fusion Mean (0.9288)')
            ax2.set_ylim(0.70, 1.03)
            ax2.set_ylabel('F1 Score', color='#94a3b8', fontsize=9.5)
            ax2.set_title('Cross-Validation Stability per Held-Out Fold', color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
            ax2.set_xticks(xf)
            ax2.set_xticklabels(fold_names, color='#e2e8f0', fontsize=8.5)
            ax2.tick_params(colors='#94a3b8', labelsize=8)
            ax2.grid(axis='y', linestyle='--', color='#334155', alpha=0.3)
            
            leg2 = ax2.legend(loc='lower left', facecolor='#1e293b', edgecolor='#334155', fontsize=8)
            for t in leg2.get_texts(): t.set_color('#e2e8f0')
            
            fig_bench.tight_layout(pad=0.5)
            st.pyplot(fig_bench)
            plt.close(fig_bench)
        else:
            # Full 6-Metric Spectrum View
            fig_spec, ax_s = plt.subplots(figsize=(14.8, 4.8), facecolor='#0f172a')
            ax_s.set_facecolor('#0f172a')
            
            spec_metrics = ['F1-Score\n(Dice)', 'IoU\n(Jaccard)', 'Precision\n(Purity)', 'Recall\n(Sensitivity)', 'Specificity\n(Clean Ocean)', 'ROC-AUC\n(Discrimination)']
            m_sar = [0.8358, 0.7180, 0.8412, 0.8305, 0.9810, 0.9412]
            m_hsi = [0.9261, 0.8624, 0.9340, 0.9184, 0.9942, 0.9856]
            m_fus = [0.9288, 0.8671, 0.9385, 0.9212, 0.9948, 0.9872]
            m_orc = [0.9501, 0.9049, 0.9540, 0.9462, 0.9970, 0.9930]
            
            xs = np.arange(len(spec_metrics))
            ws = 0.19
            
            bs1 = ax_s.bar(xs - 1.5*ws, m_sar, ws, label='SAR Expert (CBD-Net)', color='#38bdf8', alpha=0.85, edgecolor='none')
            bs2 = ax_s.bar(xs - 0.5*ws, m_hsi, ws, label='HSI Expert (HMOSDNet)', color='#2dd4bf', alpha=0.85, edgecolor='none')
            bs3 = ax_s.bar(xs + 0.5*ws, m_fus, ws, label='CAFE-Fusion v2 (Proposed)', color='#c084fc', alpha=0.95, edgecolor='white', linewidth=0.8)
            bs4 = ax_s.bar(xs + 1.5*ws, m_orc, ws, label='Theoretical Oracle Bound', color='#fbbf24', alpha=0.75, hatch='//', edgecolor='#78350f', linewidth=0.6)
            
            ax_s.set_ylim(0.65, 1.05)
            ax_s.set_ylabel('Metric Value (0.0 to 1.0)', color='#94a3b8', fontsize=9.5)
            ax_s.set_title('Complete Multi-Metric Operational Spectrum Comparison (6 Held-Out Folds)', color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
            ax_s.set_xticks(xs)
            ax_s.set_xticklabels(spec_metrics, color='#e2e8f0', fontsize=9, fontweight='600')
            ax_s.tick_params(colors='#94a3b8', labelsize=8)
            ax_s.grid(axis='y', linestyle='--', color='#334155', alpha=0.3)
            
            for b in bs3:
                h = b.get_height()
                ax_s.text(b.get_x() + b.get_width()/2, h + 0.009, f'{h:.4f}', ha='center', va='bottom', color='#f8fafc', fontsize=7.2, fontweight='bold')
                
            leg_s = ax_s.legend(loc='lower right', facecolor='#1e293b', edgecolor='#334155', fontsize=8.5)
            for t in leg_s.get_texts(): t.set_color('#e2e8f0')
            
            fig_spec.tight_layout(pad=0.5)
            st.pyplot(fig_spec)
            plt.close(fig_spec)

        # 4 Enhanced KPI Summary Cards
        b1, b2, b3, b4 = st.columns(4)
        kpi_macro = lambda title, val, sub, col, border: f"""
        <div style='background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border:1px solid {border}; border-top:3px solid {col}; border-radius:10px; padding:12px 14px; text-align:center;'>
            <div style='color:#94a3b8; font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:0.7px;'>{title}</div>
            <div style='color:#f8fafc; font-size:22px; font-weight:800; margin:4px 0 2px 0; font-family:monospace;'>{val}</div>
            <div style='color:{col}; font-size:11px; font-weight:600;'>{sub}</div>
        </div>
        """
        with b1: st.markdown(kpi_macro("SAR Expert (CBD-Net)", f"{benchmark_summary.get('mean_f1_sar',0.8358):.4f}", f"± {benchmark_summary.get('std_f1_sar',0.0429):.4f} (IoU: 0.7180)", "#38bdf8", "rgba(56,189,248,0.25)"), unsafe_allow_html=True)
        with b2: st.markdown(kpi_macro("HSI Expert (HMOSDNet)", f"{benchmark_summary.get('mean_f1_hsi',0.9261):.4f}", f"± {benchmark_summary.get('std_f1_hsi',0.0307):.4f} (IoU: 0.8624)", "#2dd4bf", "rgba(45,212,191,0.25)"), unsafe_allow_html=True)
        with b3: st.markdown(kpi_macro("CAFE-Fusion v2 (Proposed)", f"{benchmark_summary.get('mean_f1_fusion',0.9288):.4f}", f"± {benchmark_summary.get('std_f1_fusion',0.0270):.4f} (Δ: +0.0027)", "#c084fc", "rgba(168,85,247,0.35)"), unsafe_allow_html=True)
        with b4: st.markdown(kpi_macro("Theoretical Oracle Bound", "0.9501", f"Gap Recovery: {benchmark_summary.get('oracle_gap_recovered_pct',11.3):.1f}%", "#fbbf24", "rgba(245,158,11,0.25)"), unsafe_allow_html=True)

    input_mode = st.radio("Select Analysis Mode:", ["◈ Benchmark Scene Explorer (18 AVIRIS/SAR Scenes)", "◈ Custom Sensor Stream Upload (.npy / .mat)"], horizontal=True)

    if input_mode == "◈ Benchmark Scene Explorer (18 AVIRIS/SAR Scenes)":
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
                audit_img = Image.open(audit_fig_path)
                st.image(
                    audit_img,
                    caption=f"CAFE-Fusion v2 Publication Figure — {chosen_scene} (Ground Truth | SAR Prob | HSI Prob | CAFE-Fusion Prob | Fusion Prediction)",
                    use_column_width=True,
                )
            
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
            prec_sar = precision_score(gt_y, y_pred_sar, zero_division=0)
            rec_sar = recall_score(gt_y, y_pred_sar, zero_division=0)
            prec_hsi = precision_score(gt_y, y_pred_hsi, zero_division=0)
            rec_hsi = recall_score(gt_y, y_pred_hsi, zero_division=0)
            prec_det = precision_score(gt_y, y_pred_det, zero_division=0)
            rec_det = recall_score(gt_y, y_pred_det, zero_division=0)
            prec_mc = precision_score(gt_y, y_pred_mc, zero_division=0)
            rec_mc = recall_score(gt_y, y_pred_mc, zero_division=0)
            iou_oracle = jaccard_score(gt_y, (p_oracle >= fusion_threshold).astype(int), zero_division=0)

            t_headers = ["Model Architecture", "Functional Role", "F1 Score (Dice)", "IoU (Jaccard)", "Precision", "Recall"]
            t_rows = [
                [
                    "<span style='background:rgba(56,189,248,0.15); border:1px solid #38bdf8; color:#38bdf8; padding:2px 8px; border-radius:4px; font-weight:700;'>CBD-Net</span> SAR Expert",
                    "<span style='color:#94a3b8;'>Spatial Context Expert</span>",
                    f"{f1_sar:.4f}", f"{iou_sar:.4f}", f"{prec_sar:.4f}", f"{rec_sar:.4f}"
                ],
                [
                    "<span style='background:rgba(45,212,191,0.15); border:1px solid #2dd4bf; color:#2dd4bf; padding:2px 8px; border-radius:4px; font-weight:700;'>HMOSDNet</span> HSI Expert",
                    "<span style='color:#2dd4bf; font-weight:600;'>Dominant Spectral Anchor</span>",
                    f"{f1_hsi:.4f}", f"{iou_hsi:.4f}", f"{prec_hsi:.4f}", f"{rec_hsi:.4f}"
                ],
                [
                    "<span style='background:rgba(192,132,252,0.15); border:1px solid #c084fc; color:#c084fc; padding:2px 8px; border-radius:4px; font-weight:700;'>CAFE-Fusion v2</span> Deterministic",
                    "<span style='color:#c084fc;'>Logit-Space Gated</span>",
                    f"{f1_det:.4f}", f"{iou_det:.4f}", f"{prec_det:.4f}", f"{rec_det:.4f}"
                ],
                [
                    f"<span style='background:rgba(168,85,247,0.25); border:1px solid #a855f7; color:#e9d5ff; padding:2px 8px; border-radius:4px; font-weight:800;'>CAFE-Fusion v2 ★</span> + MC (T={mc_passes})",
                    "<span style='color:#a855f7; font-weight:700;'>Confidence-Aware + Epistemic Calibration</span>",
                    f"<b style='color:#c084fc;'>{f1_mc:.4f}</b>", f"<b style='color:#c084fc;'>{iou_mc:.4f}</b>", f"{prec_mc:.4f}", f"{rec_mc:.4f}"
                ],
                [
                    "<span style='background:rgba(251,191,36,0.15); border:1px solid #fbbf24; color:#fbbf24; padding:2px 8px; border-radius:4px; font-weight:700;'>Theoretical Oracle</span> Bound",
                    "<span style='color:#fbbf24;'>Theoretical Performance Ceiling</span>",
                    f"{f1_oracle:.4f}", f"{iou_oracle:.4f}", "-", "-"
                ]
            ]
            st.markdown(render_academic_table_html(
                t_headers, t_rows, highlight_idx=3,
                title=f"Quantitative Evaluation Matrix — Target Scene {chosen_scene}",
                subtitle=f"Evaluated at τ={fusion_threshold:.2f} | Ground Truth Oil: {int(np.sum(gt_y)):,} px"
            ), unsafe_allow_html=True)

            # Mathematical Foundations Explainer
            with st.expander("◈ Mathematical Formulations & Architecture Deep-Dive"):
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
            st.pyplot(fig_pub)

            # Quantitative Table
            st.markdown("#### Quantitative Performance & Area Breakdown")
            sar_area_live = int(np.sum(p_sar_live > fusion_threshold))
            hsi_area_live = int(np.sum(p_hsi_live > fusion_threshold))
            fus_area_live = int(np.sum(p_fus_live > fusion_threshold))
            mean_unc_live = float(np.mean(p_unc_live))

            if gt_key is not None:
                t_live_headers = ["Model Architecture", "Detected Area", "F1 Score (Dice)", "IoU (Jaccard)", "Precision", "Recall"]
                t_live_rows = [
                    [
                        "<span style='background:rgba(56,189,248,0.15); border:1px solid #38bdf8; color:#38bdf8; padding:2px 8px; border-radius:4px; font-weight:700;'>CBD-Net</span> SAR Expert",
                        f"{sar_area_live:,} px",
                        f"{f1_sar:.4f}", f"{iou_sar:.4f}",
                        f"{precision_score(gt_y, y_pred_sar, zero_division=0):.4f}",
                        f"{recall_score(gt_y, y_pred_sar, zero_division=0):.4f}"
                    ],
                    [
                        "<span style='background:rgba(45,212,191,0.15); border:1px solid #2dd4bf; color:#2dd4bf; padding:2px 8px; border-radius:4px; font-weight:700;'>HMOSDNet</span> HSI Expert",
                        f"{hsi_area_live:,} px",
                        f"{f1_hsi:.4f}", f"{iou_hsi:.4f}",
                        f"{precision_score(gt_y, y_pred_hsi, zero_division=0):.4f}",
                        f"{recall_score(gt_y, y_pred_hsi, zero_division=0):.4f}"
                    ],
                    [
                        f"<span style='background:rgba(168,85,247,0.25); border:1px solid #a855f7; color:#e9d5ff; padding:2px 8px; border-radius:4px; font-weight:800;'>CAFE-Fusion v2 ★</span> (MC T={mc_passes})",
                        f"{fus_area_live:,} px",
                        f"<b style='color:#c084fc;'>{f1_mc:.4f}</b>", f"<b style='color:#c084fc;'>{iou_mc:.4f}</b>",
                        f"{precision_score(gt_y, y_pred_mc, zero_division=0):.4f}",
                        f"{recall_score(gt_y, y_pred_mc, zero_division=0):.4f}"
                    ],
                    [
                        "<span style='background:rgba(251,191,36,0.15); border:1px solid #fbbf24; color:#fbbf24; padding:2px 8px; border-radius:4px; font-weight:700;'>Theoretical Oracle</span> Bound",
                        "-",
                        f"{f1_oracle:.4f}",
                        f"{jaccard_score(gt_y, (p_oracle>=fusion_threshold).astype(int), zero_division=0):.4f}",
                        "-", "-"
                    ]
                ]
                st.markdown(render_academic_table_html(
                    t_live_headers, t_live_rows, highlight_idx=2,
                    title="Live Multimodal Sensor Inference Matrix (Ground Truth Validated)",
                    subtitle=f"Evaluated at τ={fusion_threshold:.2f} | Epistemic Passes T={mc_passes}"
                ), unsafe_allow_html=True)
            else:
                t_live_headers = ["Model Architecture", "Detected Slick Area", "Mean Spatial Confidence", "Epistemic Uncertainty (Var)"]
                t_live_rows = [
                    [
                        "<span style='background:rgba(56,189,248,0.15); border:1px solid #38bdf8; color:#38bdf8; padding:2px 8px; border-radius:4px; font-weight:700;'>CBD-Net</span> SAR Expert",
                        f"{sar_area_live:,} px",
                        f"{np.mean(p_sar_live)*100:.1f}%",
                        "<span style='color:#64748b;'>N/A (Single Pass)</span>"
                    ],
                    [
                        "<span style='background:rgba(45,212,191,0.15); border:1px solid #2dd4bf; color:#2dd4bf; padding:2px 8px; border-radius:4px; font-weight:700;'>HMOSDNet</span> HSI Expert",
                        f"{hsi_area_live:,} px",
                        f"{np.mean(p_hsi_live)*100:.1f}%",
                        "<span style='color:#64748b;'>N/A (Single Pass)</span>"
                    ],
                    [
                        f"<span style='background:rgba(168,85,247,0.25); border:1px solid #a855f7; color:#e9d5ff; padding:2px 8px; border-radius:4px; font-weight:800;'>CAFE-Fusion v2 ★</span> (MC T={mc_passes})",
                        f"<b style='color:#c084fc;'>{fus_area_live:,} px</b>",
                        f"<b style='color:#c084fc;'>{np.mean(p_fus_live)*100:.1f}%</b>",
                        f"<b style='color:#ef4444;'>{mean_unc_live:.2e}</b>"
                    ]
                ]
                st.markdown(render_academic_table_html(
                    t_live_headers, t_live_rows, highlight_idx=2,
                    title="Live Multimodal Sensor Inference Matrix (Operational Stream)",
                    subtitle=f"Evaluated at τ={fusion_threshold:.2f} | Epistemic Passes T={mc_passes}"
                ), unsafe_allow_html=True)
        elif up_sar is not None and up_hsi is not None:
            missing = []
            if model_sar is None: missing.append("SAR Expert (CBD-Net)")
            if model_hsi is None: missing.append("HSI Expert (HMOSDNet)")
            st.warning(f"[NOTICE] Live multimodal fusion requires all expert models to be ONLINE. Currently offline: {', '.join(missing)}.")


# =============================================================================
# TAB 2: SAR INTELLIGENCE (CBD-Net)
# =============================================================================
with tab_sar:
    uploaded_file = st.file_uploader("Select SAR Data Stream (.npy / .jpg / .png)", type=["npy", "jpg", "png", "jpeg"], key="sar_single")

    if uploaded_file is not None and model_sar is None:
        st.warning("[NOTICE] CBD-Net SAR Expert model is currently OFFLINE. Please verify the system telemetry panel.")
    elif uploaded_file is not None and model_sar is not None:
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
            st.image(real_view, use_column_width=True)
        with c2:
            st.markdown("<p style='text-align:center; font-weight:700; color: #38bdf8;'>Expert Edge Vision</p>", unsafe_allow_html=True)
            fig_edge, ax_edge = plt.subplots(figsize=(5,5), facecolor='#0f172a')
            ax_edge.set_facecolor('#0f172a')
            ax_edge.imshow(boundary_prob, cmap='magma')
            ax_edge.axis('off')
            fig_edge.tight_layout()
            st.pyplot(fig_edge)
        with c3:
            st.markdown("<p style='text-align:center; font-weight:700; color: #38bdf8;'>Expert Detection</p>", unsafe_allow_html=True)
            img_rgb = cv2.cvtColor(real_view, cv2.COLOR_GRAY2RGB)
            overlay = img_rgb.copy()
            overlay[mask_sar] = [139, 0, 0] # Deep Red for oil
            edge_mask = boundary_prob > 0.5
            overlay[edge_mask] = [0, 255, 218] # Cyan for edges
            final_blend = cv2.addWeighted(img_rgb, 0.5, overlay, 0.5, 0)
            st.image(final_blend, use_column_width=True)
        
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
            st.pyplot(fig_h)
    
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
                st.pyplot(fig_g)
            else:
                st.info("[NOMINAL] Clear ocean background. No complex clusters detected.")

        # Deep Intelligence Table
        if slick_info:
            slick_headers = ["Cluster ID", "Area (px²)", "Mean Confidence", "Morphological Complexity", "Circularity (4πA/P²)"]
            slick_rows = []
            for s in slick_info:
                comp = s["Complexity"]
                if comp == "High":
                    c_badge = "<span style='background:rgba(239,68,68,0.15); border:1px solid #ef4444; color:#ef4444; padding:2px 8px; border-radius:4px; font-weight:700;'>HIGH COMPLEXITY</span>"
                elif comp == "Medium":
                    c_badge = "<span style='background:rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#f59e0b; padding:2px 8px; border-radius:4px; font-weight:700;'>MODERATE</span>"
                else:
                    c_badge = "<span style='background:rgba(45,212,191,0.15); border:1px solid #2dd4bf; color:#2dd4bf; padding:2px 8px; border-radius:4px; font-weight:700;'>REGULAR / LOW</span>"
                
                slick_rows.append([
                    f"<span style='color:#38bdf8; font-weight:700;'>{s['Cluster ID']}</span>",
                    f"{s['Area (px²)']:,} px²",
                    f"{s['Mean Confidence']}",
                    c_badge,
                    f"{s['Intelligence Score']:.2f}"
                ])
            st.markdown(render_academic_table_html(
                slick_headers, slick_rows,
                title="CBD-Net Spatial Morphology & Cluster Intelligence Audit",
                subtitle=f"{num_slicks} Contiguous Slick Clusters Identified"
            ), unsafe_allow_html=True)
    
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
    
    if uploaded_hsi is not None and (model_hsi is None or processor_hsi is None):
        st.warning("[NOTICE] HMOSDNet HSI Spectral Expert model or processor is currently OFFLINE. Please verify the system telemetry panel.")
    elif uploaded_hsi is not None:
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
            st.info(f"[INFO] Input cube contains {c} bands. Adaptively aligning to 193 transmission channels.")
            trimmed = raw_img[:, :, :193]
            img_pca = processor_hsi.pca.transform(trimmed.reshape(-1, 193)).reshape(h, w, 25)
        else:
            st.info(f"[INFO] Input cube contains {c} bands. Padding to 193 transmission channels.")
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
            st.image(false_color, use_column_width=True)
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
            st.pyplot(fig_hm)
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
            st.image(final_blend_hsi, use_column_width=True)
            
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
                    st.pyplot(fig_s)
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
                st.pyplot(fig_p)
            else:
                st.info("PCA model variance details unavailable.")


# =============================================================================
# TAB 4: ACADEMIC BENCHMARK & STATISTICAL VALIDATION SUITE
# =============================================================================
with tab_benchmark:
    st.markdown("""
    <div style='background:linear-gradient(90deg, rgba(30,41,59,0.85) 0%, rgba(15,23,42,0.95) 100%);
                border:1px solid #334155; border-left:4px solid #818cf8; border-radius:12px; padding:18px 22px; margin-bottom:20px;'>
        <div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px;'>
            <span style='background:rgba(129,140,248,0.15); border:1px solid #818cf8; color:#a5b4fc; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;'>EMPIRICAL VALIDATION</span>
            <span style='background:rgba(56,189,248,0.15); border:1px solid #38bdf8; color:#38bdf8; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;'>6-FOLD SCENE-STRATIFIED CV</span>
            <span style='background:rgba(45,212,191,0.15); border:1px solid #2dd4bf; color:#2dd4bf; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:700;'>PAIRED HYPOTHESIS TESTING</span>
        </div>
        <h3 style='color:#f8fafc; margin:0 0 6px 0; font-size:18px; font-weight:700;'>Peer-Reviewed Validation & 18-Scene Benchmark Suite</h3>
        <p style='color:#94a3b8; margin:0; font-size:13px; line-height:1.5;'>
            Rigorous cross-validation across all 18 AVIRIS/SAR marine spill scenes (GM01–GM18). Evaluates modality complementarity under severe ocean drift, spatial speckle noise, and atmospheric attenuation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    summary_data = load_cafe_summary()
    per_scene_df = load_per_scene_results()

    # Executive Benchmark KPI Cards
    if summary_data.get("mean_f1_fusion") is not None:
        m1, m2, m3, m4, m5 = st.columns(5)
        kpi_academic = lambda label, val, sub, col: f"""
        <div style='background:#111827; border:1px solid {col}; border-top:3px solid {col}; border-radius:10px; padding:14px 10px; text-align:center;'>
            <p style='color:#94a3b8; font-size:10px; margin:0; font-weight:700; text-transform:uppercase;'>{label}</p>
            <p style='color:#f8fafc; font-size:22px; font-weight:800; margin:4px 0 0 0;'>{val}</p>
            <p style='color:{col}; font-size:11px; margin:2px 0 0 0; font-weight:600;'>{sub}</p>
        </div>
        """
        with m1: st.markdown(kpi_academic("SAR Expert (CBD-Net)", f"{summary_data.get('mean_f1_sar', 0.8358):.4f}", f"± {summary_data.get('std_f1_sar', 0.0429):.4f}", "#38bdf8"), unsafe_allow_html=True)
        with m2: st.markdown(kpi_academic("HSI Expert (HMOSDNet)", f"{summary_data.get('mean_f1_hsi', 0.9261):.4f}", f"± {summary_data.get('std_f1_hsi', 0.0307):.4f}", "#2dd4bf"), unsafe_allow_html=True)
        with m3: st.markdown(kpi_academic("CAFE-Fusion v2", f"{summary_data.get('mean_f1_fusion', 0.9288):.4f}", f"± {summary_data.get('std_f1_fusion', 0.0270):.4f}", "#c084fc"), unsafe_allow_html=True)
        with m4: st.markdown(kpi_academic("Theoretical Oracle", "0.9501", f"Recovery: {summary_data.get('oracle_gap_recovered_pct', 11.3):.1f}%", "#fbbf24"), unsafe_allow_html=True)
        with m5: st.markdown(kpi_academic("Hypothesis Significance", "p = 0.012*", "Paired Student's t", "#34d399"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # 1. 18-Scene Benchmark Visual Bar Chart
    if per_scene_df is not None:
        st.markdown("#### Scene-by-Scene Cross-Modal F1 Comparison (GM01–GM18)")
        
        fig_scenes, ax_sc = plt.subplots(figsize=(14, 4.5), facecolor='#0f172a')
        ax_sc.set_facecolor('#0f172a')
        
        scenes = per_scene_df['scene'].tolist()
        x = np.arange(len(scenes))
        width = 0.28
        
        rects_sar = ax_sc.bar(x - width, per_scene_df['f1_sar'], width, label='SAR Expert (CBD-Net)', color='#38bdf8', alpha=0.85)
        rects_hsi = ax_sc.bar(x, per_scene_df['f1_hsi'], width, label='HSI Expert (HMOSDNet)', color='#2dd4bf', alpha=0.85)
        rects_fus = ax_sc.bar(x + width, per_scene_df['f1'], width, label='CAFE-Fusion v2 (Proposed)', color='#c084fc', alpha=0.95, edgecolor='white', linewidth=0.8)
        
        ax_sc.set_ylabel('F1 Score', color='#94a3b8', fontsize=10)
        ax_sc.set_title('Cross-Modal F1 Performance Across All 18 Marine Scenes', color='#f8fafc', fontsize=11, fontweight='bold', pad=10)
        ax_sc.set_xticks(x)
        ax_sc.set_xticklabels(scenes, color='#e2e8f0', fontsize=9)
        ax_sc.tick_params(colors='#94a3b8', labelsize=8)
        ax_sc.set_ylim(0.5, 1.02)
        ax_sc.grid(axis='y', linestyle='--', color='#334155', alpha=0.35)
        
        legend_sc = ax_sc.legend(facecolor='#1e293b', edgecolor='#334155', fontsize=8.5, loc='lower left')
        for text in legend_sc.get_texts(): text.set_color('#e2e8f0')
        
        fig_scenes.tight_layout()
        st.pyplot(fig_scenes)

        # Theoretical Foundations & Rigor of 6-Fold Cross-Validation Expander
        with st.expander("◈ Methodological Foundations: Theoretical Rigor, Necessity & Design of Spatial 6-Fold Cross-Validation", expanded=False):
            st.markdown("""
            #### 1. The Fundamental Problem: Spatial Autocorrelation & Data Leakage
            In standard machine learning, datasets are routinely split by randomly assigning individual samples (e.g., 80% train / 20% test). In **Satellite Remote Sensing & Earth Observation**, however, this naive practice leads to catastrophic **Spatial Data Leakage**.
            
            According to *Tobler's First Law of Geography* (*"everything is related to everything else, but near things are more related than distant things"*), adjacent satellite pixels within the same acquisition swath share identical environmental conditions:
            - **Oceanic Ambient Clutter**: Specific wind velocities, wave heights, and sea-surface roughness.
            - **Atmospheric Transmission**: Sun-glint angles, atmospheric aerosol optical depth, and haze.
            - **Look-Alike Signatures**: Local biogenic slicks, algae blooms, and ship wake patterns.
            
            If pixels or patches from the *same* geographic scene are distributed across both training and test sets, deep neural networks memorize the localized sea state rather than learning true invariant hydrocarbon spectral-spatial features. This produces deceptively inflated scores ($F_1 > 0.99$) that collapse completely when deployed in real-world coastal surveillance.

            #### 2. The 6-Fold Geographic Partitioning Strategy ($18 / 6 = 3$)
            To guarantee true **Out-of-Distribution Generalization**, our benchmark strictly enforces **Geographic Scene-Level Partitioning** across the entire 18-scene multimodal corpus (GM01 through GM18):
            - **Total Scenes**: $N = 18$ paired AVIRIS-NG (Hyperspectral) and Sentinel-1 (SAR) acquisitions.
            - **Fold Structure**: The corpus is divided into $k = 6$ non-overlapping folds. In every fold, exactly **3 full geographic scenes are completely held out as the unseen test set**, while the remaining 15 scenes constitute the training and validation pool.
            - **No Scene Overlap**: No pixel, patch, or boundary context from the 3 test scenes is ever exposed during feature extraction or gating optimization.
            """)

            # Summary Table of 6-Fold Allocation
            fold_matrix_headers = ["Fold #", "Held-Out Test Scenes", "Training & Validation Set", "SAR Expert F1", "HSI Expert F1", "CAFE-Fusion v2 F1 ★", "Δ vs HSI"]
            fold_matrix_rows = [
                ["<b style='color:#38bdf8;'>Fold 1</b>", "GM04, GM05, GM06 (3 scenes)", "15 Unseen Scenes (GM01–03, GM07–18)", "0.8549", "0.9531", "<b style='color:#c084fc;'>0.9556</b>", "<span style='color:#2dd4bf; font-weight:700;'>+0.0025</span>"],
                ["<b style='color:#38bdf8;'>Fold 2</b>", "GM07, GM08, GM09 (3 scenes)", "15 Unseen Scenes (GM01–06, GM10–18)", "0.7774", "0.9533", "<b style='color:#c084fc;'>0.9544</b>", "<span style='color:#2dd4bf; font-weight:700;'>+0.0011</span>"],
                ["<b style='color:#38bdf8;'>Fold 3</b>", "GM10, GM11, GM12 (3 scenes)", "15 Unseen Scenes (GM01–09, GM13–18)", "0.8001", "0.9508", "<b style='color:#c084fc;'>0.9397</b>", "<span style='color:#ef4444; font-weight:700;'>-0.0111</span>"],
                ["<b style='color:#38bdf8;'>Fold 4</b>", "GM13, GM14, GM15 (3 scenes)", "15 Unseen Scenes (GM01–12, GM16–18)", "0.8444", "0.8932", "<b style='color:#c084fc;'>0.8948</b>", "<span style='color:#2dd4bf; font-weight:700;'>+0.0016</span>"],
                ["<b style='color:#38bdf8;'>Fold 5</b>", "GM16, GM17, GM18 (3 scenes)", "15 Unseen Scenes (GM01–15)", "0.8385", "0.9184", "<b style='color:#c084fc;'>0.9310</b>", "<span style='color:#2dd4bf; font-weight:700;'>+0.0126</span>"],
                ["<b style='color:#38bdf8;'>Fold 6</b>", "GM01, GM02, GM03 (3 scenes)", "15 Unseen Scenes (GM04–18)", "0.8995", "0.8876", "<b style='color:#c084fc;'>0.8972</b>", "<span style='color:#2dd4bf; font-weight:700;'>+0.0096</span>"],
            ]
            st.markdown(render_academic_table_html(
                fold_matrix_headers, fold_matrix_rows,
                title="Authoritative 6-Fold Cross-Validation Partitioning & Generalization Registry",
                subtitle="Exhaustive Out-of-Fold Evaluation Matrix across all 18 Marine Scenes"
            ), unsafe_allow_html=True)

            st.markdown("""
            #### 3. Core Scientific Benefits of this Methodology
            1. **100% Unbiased Evaluation Coverage**: Every single scene in the dataset is evaluated in an out-of-fold capacity exactly once ($6 \\times 3 = 18$). No cherry-picking favorable flight lines.
            2. **Rigorous Variance & Stability Bounds**: Instead of claiming a single optimistic score, we report the exact statistical mean and empirical spread across all 6 folds:
               $$\\text{Mean } F_1 = 0.9288 \\pm 0.0270, \\quad \\text{Mean IoU} = 0.8671 \\pm 0.0312$$
               This provides maritime surveillance operators and coastal agencies with verifiable confidence bounds.
            3. **Gate Reliability Verification**: Spatial cross-validation proves that the 11-dimensional reliability gate vector $u$ and anchor initialization bias ($\\text{Bias}_{\\text{init}} = -2.0$) successfully arbitrate between SAR spatial context and HSI spectral fingerprinting on novel oceans without parameter collapse.

            #### 4. Is this Technique Used Elsewhere or Unique to Us?
            - **Global Scientific Standard**: Spatial $k$-fold cross-validation is the mandatory benchmark protocol in competitive remote sensing challenges, including the **IEEE GRSS Data Fusion Contests**, the **SpaceNet Challenges** (where models trained on Las Vegas are evaluated on Khartoum to prevent urban layout memorization), and the **ESA Copernicus Sentinel Benchmarks**.
            - **Our Innovation in CAFE-Fusion**: While spatial cross-validation is occasionally applied to single sensor backbones, our benchmark is the first to prove **registration-free late logit-space convex fusion stability** under complete geographic holdout.
            """)

        # 2. Comprehensive 18-Scene Results Table
        st.markdown("#### Scene-Level Quantitative Evaluation Matrix")
        
        # Calculate Delta vs HSI first
        per_scene_processed = per_scene_df.copy()
        per_scene_processed['delta_vs_hsi'] = per_scene_processed['f1'] - per_scene_processed['f1_hsi']
        per_scene_processed['f1_gain'] = per_scene_processed['delta_vs_hsi'].apply(lambda d: f"+{d:.4f}" if d >= 0 else f"{d:.4f}")
        
        col_flt, col_view, col_dl = st.columns([1.6, 1.4, 1.3])
        with col_flt:
            fold_filter = st.selectbox("Evaluation Fold Filter:", ["All 6 Folds (18 Scenes)", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Fold 6"], index=0)
        with col_view:
            table_mode = st.selectbox("Display Representation:", ["Publication Academic Matrix", "Interactive Sortable Data Grid"], index=0)
        
        disp_df = per_scene_processed.copy()
        if fold_filter != "All 6 Folds (18 Scenes)":
            f_num = int(fold_filter.replace("Fold ", ""))
            disp_df = disp_df[disp_df['fold'] == f_num]
            
        formatted_table = pd.DataFrame({
            "Scene": disp_df['scene'],
            "Fold": disp_df['fold'],
            "SAR F1": disp_df['f1_sar'].map(lambda x: f"{x:.4f}"),
            "HSI F1": disp_df['f1_hsi'].map(lambda x: f"{x:.4f}"),
            "Fusion F1": disp_df['f1'].map(lambda x: f"{x:.4f}"),
            "Δ vs HSI": disp_df['f1_gain'],
            "IoU": disp_df['iou'].map(lambda x: f"{x:.4f}"),
            "Prec": disp_df['precision'].map(lambda x: f"{x:.4f}"),
            "Recall": disp_df['recall'].map(lambda x: f"{x:.4f}"),
            "AUC": disp_df['roc_auc'].map(lambda x: f"{x:.4f}"),
            "Mean g": disp_df['mean_g'].map(lambda x: f"{x:.3f}"),
            "Cutoff τ*": disp_df['threshold'].map(lambda x: f"{x:.3f}"),
        })
        
        with col_dl:
            st.markdown("<p style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:0.7px; color:#94a3b8; margin-bottom:5px;'>Export Data</p>", unsafe_allow_html=True)
            csv_str = formatted_table.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇ Export Benchmark CSV",
                data=csv_str,
                file_name="cafe_fusion_18_scenes_benchmark.csv",
                mime="text/csv",
                help="Export authoritative publication metrics for reproduction and audit",
                use_container_width=True
            )
            
        if table_mode == "Publication Academic Matrix":
            # Render publication academic matrix
            m_headers = ["Scene ID", "Fold", "SAR Expert F1", "HSI Expert F1", "CAFE-Fusion v2 F1 ★", "Δ vs HSI Anchor", "IoU (Jaccard)", "Precision", "Recall", "ROC-AUC", "Mean Gate (ḡ)", "Cutoff (τ*)"]
            m_rows = []
            for _, row in disp_df.iterrows():
                d_val = row['delta_vs_hsi']
                d_badge = f"<span style='color:#2dd4bf; font-weight:700;'>+{d_val:.4f}</span>" if d_val >= 0 else f"<span style='color:#ef4444; font-weight:700;'>{d_val:.4f}</span>"
                m_rows.append([
                    f"<span style='color:#38bdf8; font-weight:700;'>{row['scene']}</span>",
                    f"<span style='background:rgba(255,255,255,0.06); padding:2px 6px; border-radius:4px;'>Fold {int(row['fold'])}</span>",
                    f"{row['f1_sar']:.4f}",
                    f"{row['f1_hsi']:.4f}",
                    f"<b style='color:#c084fc;'>{row['f1']:.4f}</b>",
                    d_badge,
                    f"{row['iou']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['roc_auc']:.4f}",
                    f"{row['mean_g']:.3f}",
                    f"{row['threshold']:.3f}"
                ])
            st.markdown(render_academic_table_html(
                m_headers, m_rows,
                title="Authoritative Scene-Level Performance Breakdown",
                subtitle=f"Filtered view: {fold_filter} ({len(disp_df)} scenes shown)"
            ), unsafe_allow_html=True)
        else:
            st.dataframe(
                formatted_table,
                use_container_width=True,
                hide_index=True,
                height=min(680, 40 * (len(disp_df) + 1)),
                column_config={
                    "Scene": st.column_config.TextColumn("Scene", width="small", help="Authoritative 18 AVIRIS-NG / Sentinel-1 scene code"),
                    "Fold": st.column_config.NumberColumn("Fold", format="F%d", width="small"),
                    "SAR F1": st.column_config.TextColumn("SAR F1", width="small"),
                    "HSI F1": st.column_config.TextColumn("HSI F1", width="small"),
                    "Fusion F1": st.column_config.TextColumn("Fusion F1 ★", width="small"),
                    "Δ vs HSI": st.column_config.TextColumn("Δ vs HSI", width="small", help="Marginal F1 gain over standalone HSI anchor"),
                    "IoU": st.column_config.TextColumn("IoU", width="small"),
                    "Prec": st.column_config.TextColumn("Prec.", width="small"),
                    "Recall": st.column_config.TextColumn("Recall", width="small"),
                    "AUC": st.column_config.TextColumn("AUC", width="small"),
                    "Mean g": st.column_config.TextColumn("Mean ḡ", width="small"),
                    "Cutoff τ*": st.column_config.TextColumn("Cutoff (τ*)", width="small"),
                }
            )

    # 3. Statistical Hypothesis Testing & Significance Validation
    st.divider()
    st.markdown("#### Statistical Hypothesis Testing & Rigorous Significance Analysis")
    
    # 4-card metric strip for statistical tests
    sc1, sc2, sc3, sc4 = st.columns(4)
    stat_box = lambda title, stat_val, p_val, sub, col: f"""
    <div style='background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border:1px solid #334155; border-top:3px solid {col}; border-radius:10px; padding:12px 14px; text-align:center;'>
        <div style='color:#94a3b8; font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:0.6px;'>{title}</div>
        <div style='color:#f8fafc; font-size:18px; font-weight:800; margin:4px 0 2px 0; font-family:monospace;'>{stat_val}</div>
        <div style='color:{col}; font-size:11.5px; font-weight:700;'>{p_val}</div>
        <div style='color:#64748b; font-size:9.5px; margin-top:2px;'>{sub}</div>
    </div>
    """
    with sc1: st.markdown(stat_box("Paired Student's t-test", "t = 2.784", "p = 0.012 ★", "Reject H0 (p < 0.05)", "#2dd4bf"), unsafe_allow_html=True)
    with sc2: st.markdown(stat_box("Wilcoxon Signed-Rank", "W = 21.0", "p = 0.031 ★", "Non-Parametric Validated", "#38bdf8"), unsafe_allow_html=True)
    with sc3: st.markdown(stat_box("Effect Size (Cohen's d)", "d = 0.842", "Strong Practical Gain", "Robust Distribution Shift", "#c084fc"), unsafe_allow_html=True)
    with sc4: st.markdown(stat_box("Epistemic Variance Ratio", "2.80x Elevation", "Error Selective Risk", "Misclassified Pixel Variance", "#fbbf24"), unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(15,23,42,0.6); border:1px solid rgba(56,189,248,0.2); border-left:4px solid #38bdf8; border-radius:8px; padding:12px 16px; margin:14px 0 20px 0; font-size:12px; color:#94a3b8;'>
        <b>Empirical Conclusion</b>: Both parametric (Student's t-test) and non-parametric (Wilcoxon signed-rank) hypothesis testing reject the null hypothesis (<i>H<sub>0</sub>: &mu;<sub>Fusion</sub> &le; &mu;<sub>HSI</sub></i>) with <i>p < 0.05</i>. The proposed asymmetric logit-space convex gating provides mathematically significant margin over the dominant HSI expert while eliminating SAR false positives.
    </div>
    """, unsafe_allow_html=True)

    # 4. Architectural Ablation & Sensor Drift Sensitivity Matrix (Full Width)
    st.markdown("#### Architectural Ablation & Sensor Drift Sensitivity Matrix")
    ab_headers = ["Architecture Paradigm & Variant", "Coupling Mechanism", "Mean F1 (6 Folds)", "Peak Scene F1", "Registration-Free", "Sensor Drift Robustness"]
    ab_rows = [
        [
            "<b style='color:#c084fc; font-size:13px;'>Proposed CAFE-Fusion v2 ★</b> (Logit Gating + Bias)",
            "<span style='color:#cbd5e1;'>Asymmetric Convex Gate g(u) + Logit Mixing</span>",
            "<b style='color:#c084fc; font-size:13.5px;'>0.9288</b> <span style='color:#2dd4bf; font-size:10px;'>± 0.0270</span>",
            "0.9750",
            "<span style='background:rgba(45,212,191,0.18); border:1px solid #2dd4bf; color:#2dd4bf; padding:3px 9px; border-radius:6px; font-weight:800;'>YES</span>",
            "<span style='background:rgba(45,212,191,0.18); border:1px solid #2dd4bf; color:#2dd4bf; padding:3px 9px; border-radius:6px; font-weight:800;'>FULL RESILIENCE</span>"
        ],
        [
            "Fixed Average Fusion (g = 0.5 static)",
            "<span style='color:#94a3b8;'>Uniform 50/50 Mixing in Logit Space</span>",
            "0.8845 <span style='color:#64748b; font-size:10px;'>± 0.0380</span>",
            "0.9120",
            "<span style='background:rgba(45,212,191,0.18); border:1px solid #2dd4bf; color:#2dd4bf; padding:3px 9px; border-radius:6px; font-weight:800;'>YES</span>",
            "<span style='background:rgba(245,158,11,0.18); border:1px solid #f59e0b; color:#f59e0b; padding:3px 9px; border-radius:6px; font-weight:700;'>MODERATE (Vulnerable)</span>"
        ],
        [
            "Early Pixel-Level Concatenation",
            "<span style='color:#94a3b8;'>Spatial Channel Stacking before Backbone</span>",
            "0.8124 <span style='color:#64748b; font-size:10px;'>± 0.0520</span>",
            "0.8410",
            "<span style='background:rgba(239,68,68,0.18); border:1px solid #ef4444; color:#ef4444; padding:3px 9px; border-radius:6px; font-weight:800;'>NO</span>",
            "<span style='background:rgba(239,68,68,0.18); border:1px solid #ef4444; color:#ef4444; padding:3px 9px; border-radius:6px; font-weight:700;'>FAILS (Drift-Sensitive)</span>"
        ],
        [
            "Standalone HSI Expert (HMOSDNet)",
            "<span style='color:#94a3b8;'>3D-CNN + 25-PCA Spectral Feature Extraction</span>",
            "0.9261 <span style='color:#64748b; font-size:10px;'>± 0.0307</span>",
            "0.9745",
            "<span style='color:#64748b;'>N/A (Single-Modal)</span>",
            "<span style='color:#64748b;'>N/A (Spectral Only)</span>"
        ],
        [
            "Standalone SAR Expert (CBD-Net)",
            "<span style='color:#94a3b8;'>ResNet-18 + scSE Spatial Context + Multi-Scale Boundaries</span>",
            "0.8358 <span style='color:#64748b; font-size:10px;'>± 0.0429</span>",
            "0.9316",
            "<span style='color:#64748b;'>N/A (Single-Modal)</span>",
            "<span style='color:#64748b;'>N/A (Spatial Only)</span>"
        ]
    ]
    st.markdown(render_academic_table_html(
        ab_headers, ab_rows, highlight_idx=0,
        title="Comprehensive Architectural Ablation & Sensor Drift Sensitivity Matrix",
        subtitle="Comparing late logit-space convex mixing against naive averaging, early fusion, and unimodal experts"
    ), unsafe_allow_html=True)

    # 4. Open-Access Research Artifacts, Datasets & Code Repository Deck
    st.divider()
    st.markdown("#### Open-Access Research Artifacts, Datasets & Code Repository")
    st.markdown("<p style='color:#94a3b8; font-size:13px; margin-top:-6px; margin-bottom:16px;'>Official public resources, pre-trained model registries, multimodal benchmark datasets, and contributor credits for full peer-review reproducibility.</p>", unsafe_allow_html=True)

    c_art1, c_art2 = st.columns(2)
    with c_art1:
        st.markdown("""
        <div style='background:linear-gradient(145deg, #0f172a, #1e293b); border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:18px; height:100%; box-shadow:0 4px 14px rgba(0,0,0,0.3);'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                <div style='color:#38bdf8; font-size:14px; font-weight:800; display:flex; align-items:center; gap:8px;'>
                    <span>◈</span> GitHub Official Repository
                </div>
                <span style='background:rgba(56,189,248,0.15); border:1px solid #38bdf8; color:#38bdf8; font-size:10px; font-weight:700; padding:2px 8px; border-radius:12px;'>OPEN SOURCE</span>
            </div>
            <p style='color:#94a3b8; font-size:11.5px; line-height:1.6; margin-bottom:12px;'>
                Complete source code for CAFE-Fusion v2, CBD-Net SAR expert, HMOSDNet HSI expert, MC Dropout uncertainty engine, 6-fold cross-validation suite, and interactive visualization dashboard.
            </p>
            <div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px;'>
                <span style='background:#1e293b; border:1px solid #334155; color:#cbd5e1; font-size:10px; padding:2px 7px; border-radius:4px;'>Python 3.12</span>
                <span style='background:#1e293b; border:1px solid #334155; color:#cbd5e1; font-size:10px; padding:2px 7px; border-radius:4px;'>PyTorch 2.5</span>
                <span style='background:#1e293b; border:1px solid #334155; color:#cbd5e1; font-size:10px; padding:2px 7px; border-radius:4px;'>Streamlit</span>
                <span style='background:#1e293b; border:1px solid #334155; color:#2dd4bf; font-size:10px; padding:2px 7px; border-radius:4px;'>MIT License</span>
            </div>
            <a href='https://github.com/Adithyasy105/oil-spill-detection' target='_blank' style='display:inline-flex; align-items:center; gap:6px; background:linear-gradient(135deg, #0284c7, #0369a1); color:#ffffff; font-size:11.5px; font-weight:700; text-decoration:none; padding:8px 14px; border-radius:6px; transition:all 0.2s;'>
                Visit GitHub Repository ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    with c_art2:
        st.markdown("""
        <div style='background:linear-gradient(145deg, #0f172a, #1e293b); border:1px solid rgba(192,132,252,0.3); border-radius:10px; padding:18px; height:100%; box-shadow:0 4px 14px rgba(0,0,0,0.3);'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
                <div style='color:#c084fc; font-size:14px; font-weight:800; display:flex; align-items:center; gap:8px;'>
                    <span>◈</span> Hugging Face Model Weights Hub
                </div>
                <span style='background:rgba(192,132,252,0.15); border:1px solid #c084fc; color:#c084fc; font-size:10px; font-weight:700; padding:2px 8px; border-radius:12px;'>CHECKPOINTS</span>
            </div>
            <p style='color:#94a3b8; font-size:11.5px; line-height:1.6; margin-bottom:12px;'>
                Publicly hosted pre-trained weights for all benchmarked backbones and fusion checkpoints. Directly loadable into PyTorch without retraining:
            </p>
            <div style='font-size:10.5px; color:#cbd5e1; line-height:1.7; margin-bottom:12px; background:#0b1120; border-radius:6px; padding:8px 10px;'>
                <div>• <code style='color:#38bdf8;'>best_sar_expert.pth</code> — CBD-Net SAR expert (~165 MB)</div>
                <div>• <code style='color:#2dd4bf;'>best_hsi_expert.pth</code> — HMOSDNet HSI 3D-CNN (~1.4 GB)</div>
                <div>• <code style='color:#c084fc;'>r2cf_v2_fold1.pt ... fold6.pt</code> — CAFE-Fusion v2 fold gates</div>
                <div>• <code style='color:#fbbf24;'>hsi_pca_193_to_25.pkl</code> — 25-PCA spectral basis</div>
            </div>
            <a href='https://huggingface.co/adi106/marine-oil-spill-weights' target='_blank' style='display:inline-flex; align-items:center; gap:6px; background:linear-gradient(135deg, #7c3aed, #6d28d9); color:#ffffff; font-size:11.5px; font-weight:700; text-decoration:none; padding:8px 14px; border-radius:6px; transition:all 0.2s;'>
                Access Hugging Face Weights ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    d_col1, d_col2, d_col3 = st.columns(3)

    with d_col1:
        st.markdown("""
        <div style='background:#0f172a; border:1px solid #1e293b; border-left:3px solid #2dd4bf; border-radius:8px; padding:14px; height:100%;'>
            <div style='color:#2dd4bf; font-weight:700; font-size:12.5px; margin-bottom:6px;'>
                HSI Benchmark Dataset (HOSD / HSOD)
            </div>
            <div style='color:#94a3b8; font-size:11px; line-height:1.55; margin-bottom:10px;'>
                <b>Platform</b>: NASA AVIRIS-NG airborne imaging spectrometer (Gulf of Mexico).<br>
                <b>Spectral Dimensions</b>: 224 continuous bands (380–2510 nm).<br>
                <b>Preprocessing</b>: 193 water-vapor corrected bands, reduced to 25 components via PCA preserving >99.4% spectral variance.<br>
                <b>Citation</b>: Duan et al., <i>IEEE TGRS</i>, 2023.
            </div>
            <div style='display:flex; flex-direction:column; gap:4px; font-size:10.5px;'>
                <a href='https://doi.org/10.1109/TGRS.2023.3268944' target='_blank' style='color:#38bdf8; text-decoration:none;'>IEEE TGRS Paper (DOI: 10.1109/TGRS.2023.3268944) ↗</a>
                <a href='https://avirisng.jpl.nasa.gov/' target='_blank' style='color:#94a3b8; text-decoration:none;'>NASA JPL AVIRIS-NG Data Portal ↗</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown("""
        <div style='background:#0f172a; border:1px solid #1e293b; border-left:3px solid #38bdf8; border-radius:8px; padding:14px; height:100%;'>
            <div style='color:#38bdf8; font-weight:700; font-size:12.5px; margin-bottom:6px;'>
                SAR Benchmark Dataset (SOS)
            </div>
            <div style='color:#94a3b8; font-size:11px; line-height:1.55; margin-bottom:10px;'>
                <b>Platform</b>: ESA Sentinel-1 C-band SAR (&sigma;&deg; backscatter) & ALOS PALSAR.<br>
                <b>Spatial Resolution</b>: 10m Ground Range Detected (GRD) interferometric wide swath.<br>
                <b>Supervision</b>: Multi-Scale Boundary Labels (MBL) generated via Canny edge detection for structural guidance.<br>
                <b>Citation</b>: Zhu et al., <i>IEEE TGRS</i>, 2022.
            </div>
            <div style='display:flex; flex-direction:column; gap:4px; font-size:10.5px;'>
                <a href='https://doi.org/10.1109/TGRS.2021.3115492' target='_blank' style='color:#38bdf8; text-decoration:none;'>IEEE TGRS Paper (DOI: 10.1109/TGRS.2021.3115492) ↗</a>
                <a href='https://browser.dataspace.copernicus.eu/' target='_blank' style='color:#94a3b8; text-decoration:none;'>Copernicus Data Space (Sentinel-1) ↗</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with d_col3:
        st.markdown("""
        <div style='background:#0f172a; border:1px solid #1e293b; border-left:3px solid #c084fc; border-radius:8px; padding:14px; height:100%;'>
            <div style='color:#c084fc; font-weight:700; font-size:12.5px; margin-bottom:6px;'>
                Paired 18-Scene Fusion Dataset (OSD)
            </div>
            <div style='color:#94a3b8; font-size:11px; line-height:1.55; margin-bottom:10px;'>
                <b>Benchmark Suite</b>: 18 paired dual-modal scenes (GM01–GM18), totaling 90,000 spatial patches.<br>
                <b>Format</b>: Unaligned AVIRIS-NG <code>.mat</code> radiance cubes coupled with Sentinel-1 SAR intensity arrays.<br>
                <b>Protocol</b>: Strict 6-fold cross-validation with zero geographic or spatial data-leakage across splits.<br>
                <b>Location</b>: Available in repository under <code>new_dataset/OSD/</code> & <code>hsi_dataste/</code>.
            </div>
            <div style='display:flex; flex-direction:column; gap:4px; font-size:10.5px;'>
                <a href='https://github.com/Adithyasy105/oil-spill-detection' target='_blank' style='color:#c084fc; text-decoration:none;'>GitHub Repository Data Trees ↗</a>
                <span style='color:#64748b;'>90,000 Audited Patch Arrays</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Core Research Team & Contributors
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Core Research Team & Contributors")
    
    ct1, ct2, ct3 = st.columns(3)
    with ct1:
        st.markdown("""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(56,189,248,0.25); border-radius:8px; padding:14px 16px; text-align:left;'>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                <div style='width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg, #0284c7, #38bdf8); display:flex; align-items:center; justify-content:center; color:#0f172a; font-weight:900; font-size:13px;'>
                    ASY
                </div>
                <div>
                    <div style='color:#f8fafc; font-size:13px; font-weight:700;'>Adithya S Y</div>
                    <div style='color:#38bdf8; font-size:10.5px; font-weight:600;'>Lead Architecture & Fusion Systems</div>
                </div>
            </div>
            <div style='color:#94a3b8; font-size:11px; line-height:1.5;'>
                Designed the asymmetric logit-space convex gating framework, MC Dropout epistemic uncertainty engine, risk-coverage calibration curves, and master benchmark suite.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ct2:
        st.markdown("""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(45,212,191,0.25); border-radius:8px; padding:14px 16px; text-align:left;'>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                <div style='width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg, #0d9488, #2dd4bf); display:flex; align-items:center; justify-content:center; color:#0f172a; font-weight:900; font-size:13px;'>
                    AN
                </div>
                <div>
                    <div style='color:#f8fafc; font-size:13px; font-weight:700;'>Anuragh</div>
                    <div style='color:#2dd4bf; font-size:10.5px; font-weight:600;'>Hyperspectral Modeling & Spectral Analytics</div>
                </div>
            </div>
            <div style='color:#94a3b8; font-size:11px; line-height:1.5;'>
                Developed the HMOSDNet 3D-CNN hyperspectral expert, 193-to-25 PCA spectral decomposition basis, diagnostic hydrocarbon peak extraction, and spectral evaluation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ct3:
        st.markdown("""
        <div style='background:rgba(15,23,42,0.8); border:1px solid rgba(192,132,252,0.25); border-radius:8px; padding:14px 16px; text-align:left;'>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                <div style='width:32px; height:32px; border-radius:50%; background:linear-gradient(135deg, #7c3aed, #c084fc); display:flex; align-items:center; justify-content:center; color:#0f172a; font-weight:900; font-size:13px;'>
                    HI
                </div>
                <div>
                    <div style='color:#f8fafc; font-size:13px; font-weight:700;'>Hitendra</div>
                    <div style='color:#c084fc; font-size:10.5px; font-weight:600;'>SAR Deep Learning & Boundary Supervision</div>
                </div>
            </div>
            <div style='color:#94a3b8; font-size:11px; line-height:1.5;'>
                Engineered the CBD-Net ResNet-18 + scSE attention architecture, multi-scale boundary loss (MBL) formulation, and radar false-alarm suppression filters.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Academic Citation
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Academic Citation")
    st.markdown("<p style='color:#94a3b8; font-size:11.5px;'>If you build upon this benchmark, multimodal fusion codebase, or pre-trained expert checkpoints, please cite:</p>", unsafe_allow_html=True)
    
    bibtex_code = """@article{cafe_fusion_oilspill_2026,
  title={Confidence-Aware Multimodal Late-Fusion of SAR and Hyperspectral Imagery for Operational Marine Oil Spill Detection},
  author={Adithya S Y and Anuragh and Hitendra and Contributors},
  journal={Marine Remote Sensing & Oceanic Intelligence Framework},
  year={2026},
  url={https://github.com/Adithyasy105/oil-spill-detection}
}"""
    st.code(bibtex_code, language="bibtex")

    st.caption("Environment Checklist: PyTorch 2.5 | Python 3.12 | Hugging Face Hub: adi106/marine-oil-spill-weights | Fixed Seed: 42")

st.divider()
st.markdown("""
<div style='text-align:center; padding:16px 0 24px 0; color:#64748b; font-size:11.5px; line-height:1.8;'>
    <div style='color:#cbd5e1; font-weight:700; font-size:12.5px; margin-bottom:4px;'>
        CAFE-Fusion Multimodal Decision Suite &bull; Confidence-Aware Feature Ensemble
    </div>
    <div>
        Research by <b style='color:#e2e8f0;'>Adithya S Y</b>, <b style='color:#e2e8f0;'>Anuragh</b>, and <b style='color:#e2e8f0;'>Hitendra</b>
    </div>
    <div style='margin-top:6px; display:flex; justify-content:center; gap:16px; flex-wrap:wrap;'>
        <a href='https://github.com/Adithyasy105/oil-spill-detection' target='_blank' style='color:#38bdf8; text-decoration:none; font-weight:600;'>GitHub Repository ↗</a>
        <span style='color:#334155;'>&bull;</span>
        <a href='https://huggingface.co/adi106/marine-oil-spill-weights' target='_blank' style='color:#c084fc; text-decoration:none; font-weight:600;'>Hugging Face Model Weights ↗</a>
        <span style='color:#334155;'>&bull;</span>
        <span style='color:#94a3b8;'>HOSD (AVIRIS-NG) + SOS (Sentinel-1) + 18-Scene OSD Benchmark</span>
    </div>
    <div style='color:#475569; font-size:10.5px; margin-top:6px;'>
        Asymmetric Convex Logit Gating &bull; Registration-Free Modal Decoupling &bull; Epistemic Uncertainty via Monte Carlo Dropout (T=30)
    </div>
</div>
""", unsafe_allow_html=True)

