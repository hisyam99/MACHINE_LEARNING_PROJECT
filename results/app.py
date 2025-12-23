import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from PIL import Image
from tensorflow.keras import layers, models
from transformers import AutoImageProcessor, ViTForImageClassification

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Ultimate COVID-19 AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Apple Liquid Glass Design with Dark Mode
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ==================== APPLE DESIGN SYSTEM ==================== */
    :root {
        --glass-bg: rgba(255, 255, 255, 0.72);
        --glass-border: rgba(255, 255, 255, 0.18);
        --shadow-light: 0 8px 32px rgba(0, 0, 0, 0.08);
        --shadow-medium: 0 12px 48px rgba(0, 0, 0, 0.12);
        --text-primary: #1d1d1f;
        --text-secondary: #6e6e73;
        --bg-primary: #f5f5f7;
        --bg-secondary: #e8e8ed;
        --accent-blue: #007aff;
        --accent-green: #34c759;
        --accent-red: #ff3b30;
        --accent-orange: #ff9500;
    }
    
    /* ==================== DARK MODE SUPPORT ==================== */
    [data-theme="dark"] {
        --glass-bg: rgba(30, 30, 30, 0.72);
        --glass-border: rgba(255, 255, 255, 0.12);
        --shadow-light: 0 8px 32px rgba(0, 0, 0, 0.5);
        --shadow-medium: 0 12px 48px rgba(0, 0, 0, 0.7);
        --text-primary: #f5f5f7;
        --text-secondary: #98989d;
        --bg-primary: #1c1c1e;
        --bg-secondary: #2c2c2e;
        --accent-blue: #0a84ff;
        --accent-green: #30d158;
        --accent-red: #ff453a;
        --accent-orange: #ff9f0a;
    }
    
    /* Auto-detect dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --glass-bg: rgba(30, 30, 30, 0.72);
            --glass-border: rgba(255, 255, 255, 0.12);
            --shadow-light: 0 8px 32px rgba(0, 0, 0, 0.5);
            --shadow-medium: 0 12px 48px rgba(0, 0, 0, 0.7);
            --text-primary: #f5f5f7;
            --text-secondary: #98989d;
            --bg-primary: #1c1c1e;
            --bg-secondary: #2c2c2e;
            --accent-blue: #0a84ff;
            --accent-green: #30d158;
            --accent-red: #ff453a;
            --accent-orange: #ff9f0a;
        }
    }
    
    /* ==================== ADAPTIVE BACKGROUND ==================== */
    .main {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        position: relative;
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 30%, rgba(0, 122, 255, 0.08), transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(52, 199, 89, 0.08), transparent 40%);
        z-index: 0;
        pointer-events: none;
    }
    
    @media (prefers-color-scheme: dark) {
        .main::before {
            background: 
                radial-gradient(circle at 20% 30%, rgba(10, 132, 255, 0.15), transparent 40%),
                radial-gradient(circle at 80% 70%, rgba(48, 209, 88, 0.15), transparent 40%);
        }
    }
    
    /* ==================== LIQUID GLASS CONTAINERS ==================== */
    .block-container {
        background: var(--glass-bg);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-radius: 24px;
        border: 1px solid var(--glass-border);
        padding: 2.5rem !important;
        box-shadow: var(--shadow-light);
        position: relative;
        z-index: 1;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* ==================== TYPOGRAPHY ==================== */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.75rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        line-height: 1.1;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-bottom: 0.875rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    p, div, span, label {
        font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: rgba(249, 249, 249, 0.92);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-right: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: none;
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            background: rgba(28, 28, 30, 0.92);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-primary) !important;
        font-weight: 400;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255, 255, 255, 0.6);
        padding: 12px 16px;
        border-radius: 12px;
        margin: 6px 0;
        transition: all 0.2s ease;
        border: 1px solid rgba(0, 0, 0, 0.04);
        cursor: pointer;
        display: block;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] .stRadio > label {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] .stRadio > label:hover {
            background: rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton > button {
        background: var(--accent-blue);
        color: white !important;
        border: none;
        border-radius: 12px;
        height: 52px;
        font-weight: 500;
        font-size: 1rem;
        letter-spacing: -0.2px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 122, 255, 0.3);
        background: #0051d5;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 8px rgba(0, 122, 255, 0.2);
    }
    
    /* Light mode button styling */
    @media (prefers-color-scheme: light) {
        .stButton > button {
            background: #007aff;
            box-shadow: 0 4px 16px rgba(0, 122, 255, 0.25);
        }
        
        .stButton > button:hover {
            background: #0051d5;
            box-shadow: 0 8px 24px rgba(0, 122, 255, 0.35);
        }
    }
    
    /* Dark mode button styling */
    @media (prefers-color-scheme: dark) {
        .stButton > button {
            background: #0a84ff;
            box-shadow: 0 4px 16px rgba(10, 132, 255, 0.3);
        }
        
        .stButton > button:hover {
            background: #409cff;
            box-shadow: 0 8px 24px rgba(10, 132, 255, 0.4);
        }
    }
    
    /* ==================== METRIC CARDS ==================== */
    .metric-card {
        background: var(--glass-bg);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        padding: 28px;
        border-radius: 20px;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-light);
        text-align: center;
        margin: 12px 0;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-medium);
    }
    
    .metric-value {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin: 12px 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem !important;
        color: var(--text-secondary) !important;
        font-weight: 500;
        margin-bottom: 8px;
        letter-spacing: -0.1px;
    }
    
    /* ==================== PREDICTION BOXES ==================== */
    .prediction-box {
        padding: 16px 24px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin-top: 12px;
        font-size: 1.25rem;
        letter-spacing: -0.3px;
        box-shadow: var(--shadow-light);
        transition: all 0.2s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }
    
    .pred-covid {
        background: var(--accent-red);
        color: white;
    }
    
    .pred-non {
        background: var(--accent-orange);
        color: white;
    }
    
    .pred-normal {
        background: var(--accent-green);
        color: white;
    }
    
    /* ==================== FILE UPLOADER ==================== */
    [data-testid="stFileUploader"] {
        background: var(--glass-bg);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border: 2px dashed rgba(0, 122, 255, 0.3);
        border-radius: 16px;
        padding: 32px;
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-blue);
        box-shadow: 0 8px 24px rgba(0, 122, 255, 0.1);
    }
    
    /* ==================== DATAFRAME ==================== */
    [data-testid="stDataFrame"] {
        background: white;
        backdrop-filter: blur(40px);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-light);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stDataFrame"] {
            background: rgba(44, 44, 46, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div {
        background: var(--accent-blue);
        border-radius: 8px;
        height: 8px;
        transition: all 0.3s ease;
    }
    
    .stProgress > div {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
    }
    
    @media (prefers-color-scheme: dark) {
        .stProgress > div {
            background: rgba(255, 255, 255, 0.1);
        }
    }
    
    /* Progress text */
    .stProgress + div {
        color: var(--text-primary) !important;
    }
    
    /* ==================== SELECTBOX & RADIO ==================== */
    [data-testid="stSelectbox"], .stRadio {
        background: var(--glass-bg);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid var(--glass-border);
        transition: all 0.2s ease;
    }
    
    [data-testid="stSelectbox"]:hover, .stRadio:hover {
        box-shadow: var(--shadow-light);
    }
    
    /* Ensure select and input text is visible in both modes */
    select, input, textarea {
        color: var(--text-primary) !important;
    }
    
    [data-testid="stSelectbox"] select {
        background: rgba(255, 255, 255, 0.9) !important;
        color: var(--text-primary) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stSelectbox"] select {
            background: rgba(44, 44, 46, 0.9) !important;
            color: var(--text-primary) !important;
        }
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(0, 0, 0, 0.04);
        border-radius: 14px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.2s ease;
        padding: 10px 20px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        color: var(--text-primary) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* ==================== IMAGES ==================== */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: var(--shadow-light);
        transition: all 0.2s ease;
    }
    
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
        box-shadow: var(--shadow-medium);
    }
    
    /* ==================== DIVIDER ==================== */
    hr {
        border: none;
        height: 1px;
        background: rgba(0, 0, 0, 0.1);
        margin: 24px 0;
    }
    
    @media (prefers-color-scheme: dark) {
        hr {
            background: rgba(255, 255, 255, 0.12);
        }
    }
    
    /* ==================== CHARTS ==================== */
    [data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: var(--shadow-light);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    @media (prefers-color-scheme: dark) {
        [data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] {
            background: rgba(44, 44, 46, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
    }
    
    /* ==================== INFO/WARNING BOXES ==================== */
    .stAlert {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-radius: 12px;
        border-left: 3px solid;
        box-shadow: var(--shadow-light);
    }
    
    /* Make sure alert text is visible in light mode */
    .stAlert [data-testid="stMarkdownContainer"] p,
    .stAlert div {
        color: var(--text-primary) !important;
    }
    
    /* Info alert styling for both modes */
    [data-baseweb="notification"] {
        background: var(--glass-bg) !important;
        border: 1px solid var(--glass-border) !important;
    }
    
    [data-baseweb="notification"] div {
        color: var(--text-primary) !important;
    }
    
    /* ==================== RESPONSIVE DESIGN ==================== */
    @media (max-width: 768px) {
        .block-container {
            padding: 1.5rem !important;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        .metric-value {
            font-size: 2rem !important;
        }
        
        .prediction-box {
            font-size: 1.1rem;
            padding: 14px 20px;
        }
        
        .stButton > button {
            height: 48px;
            font-size: 0.95rem;
        }
    }
    
    @media (max-width: 480px) {
        .block-container {
            padding: 1rem !important;
        }
        
        h1 {
            font-size: 1.75rem !important;
        }
        
        .metric-card {
            padding: 20px;
        }
        
        .metric-value {
            font-size: 1.75rem !important;
        }
    }
    
    /* ==================== LOADING ANIMATION ==================== */
    .stSpinner > div {
        border-top-color: var(--accent-blue) !important;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.04);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }
    
    @media (prefers-color-scheme: dark) {
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    }
    
    /* ==================== ADDITIONAL POLISH ==================== */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    .element-container {
        transition: all 0.2s ease;
    }
    
    /* ==================== LIGHT MODE FIXES ==================== */
    @media (prefers-color-scheme: light) {
        /* Ensure all text is dark in light mode */
        [data-testid="stMarkdownContainer"] {
            color: var(--text-primary) !important;
        }
        
        /* Metric labels and values */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: var(--accent-blue) !important;
        }
        
        /* Toast notifications */
        .stToast {
            background: white !important;
            color: #1d1d1f !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Code blocks */
        code {
            background: rgba(0, 0, 0, 0.05) !important;
            color: #1d1d1f !important;
        }
        
        /* Ensure captions are visible */
        .stImage caption,
        [data-testid="stCaptionContainer"] {
            color: var(--text-secondary) !important;
        }
    }
    
    /* ==================== DARK MODE ENHANCEMENTS ==================== */
    @media (prefers-color-scheme: dark) {
        /* Ensure all text is light in dark mode */
        [data-testid="stMarkdownContainer"] {
            color: var(--text-primary) !important;
        }
        
        /* Metric labels and values */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }
        
        /* Toast notifications */
        .stToast {
            background: rgba(44, 44, 46, 0.95) !important;
            color: #f5f5f7 !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Code blocks */
        code {
            background: rgba(255, 255, 255, 0.1) !important;
            color: #f5f5f7 !important;
        }
    }
    
    /* ==================== STREAMLIT THEME DETECTION ==================== */
    /* Streamlit's dark theme detection */
    [data-testid="stAppViewContainer"][data-theme="dark"],
    [data-testid="stHeader"][data-theme="dark"] {
        color-scheme: dark;
    }
    
    /* Light mode specific fixes */
    @media (prefers-color-scheme: light) {
        [data-testid="stAppViewContainer"] {
            background: var(--bg-primary);
        }
        
        .stMarkdown, .stText, p, span, div {
            color: var(--text-primary) !important;
        }
        
        /* Ensure proper contrast in light mode */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
        }
        
        /* Light mode glass effect should be more subtle */
        .block-container {
            background: rgba(255, 255, 255, 0.8);
        }
        
        /* Ensure inputs are visible */
        input, textarea, select {
            background: white !important;
            color: #1d1d1f !important;
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
    }
    
    /* Override Streamlit's default backgrounds in dark mode */
    @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] {
            background: var(--bg-primary);
        }
        
        .stMarkdown, .stText {
            color: var(--text-primary);
        }
        
        /* Dark mode inputs */
        input, textarea, select {
            background: rgba(44, 44, 46, 0.9) !important;
            color: #f5f5f7 !important;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
    }
    </style>
    
    <script>
    // Auto-detect and apply theme
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    
    // Listen for theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
    });
    </script>
    """, unsafe_allow_html=True)

# ==========================================
# 2. KONSTANTA & CONFIG
# ==========================================
CLASSES = {0: "COVID-19", 1: "Non-COVID", 2: "Normal"}
TARGET_SIZE = (224, 224)

# Fix path untuk Streamlit Cloud
import os
# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(APP_DIR, "artifacts")

# ==========================================
# 3. KELAS KUSTOM KERAS (Wajib untuk Load Model)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class LoRADense(layers.Layer):
    def __init__(self, units, rank=8, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        dim = input_shape[-1]
        self.kernel_frozen = self.add_weight(name="kernel_frozen", shape=(dim, self.units), initializer="glorot_uniform", trainable=False)
        self.lora_A = self.add_weight(name="lora_A", shape=(dim, self.rank), initializer="he_uniform", trainable=True)
        self.lora_B = self.add_weight(name="lora_B", shape=(self.rank, self.units), initializer="zeros", trainable=True)
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel_frozen) + tf.matmul(tf.matmul(inputs, self.lora_A), self.lora_B) + self.bias
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "rank": self.rank})
        return config

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size=16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
        return tf.reshape(patches, (batch_size, -1, patches.shape[-1]))
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=196, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        return self.projection(patch) + self.position_embedding(tf.range(start=0, limit=self.num_patches, delta=1))
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

# ==========================================
# 4. FUNGSI PREPROCESSING & FEATURE EXTRACTION
# ==========================================
def apply_clahe(img01):
    u8 = (img01 * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255.0

def heuristic_lung_crop(img01, padding=20):
    H, W = img01.shape
    u8 = (img01 * 255).astype(np.uint8)
    _, m = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, lab = cv2.connectedComponents(m)
    if num <= 1: return cv2.resize(img01, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    areas = sorted([(k, int((lab == k).sum())) for k in range(1, num)], key=lambda x: x[1], reverse=True)[:2]
    mask = np.zeros_like(m, dtype=np.uint8)
    for k, _ in areas: mask[lab == k] = 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return cv2.resize(img01, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    x0, x1 = max(0, xs.min() - padding), min(W - 1, xs.max() + padding)
    y0, y1 = max(0, ys.min() - padding), min(H - 1, ys.max() + padding)
    return cv2.resize(img01[y0:y1+1, x0:x1+1], TARGET_SIZE, interpolation=cv2.INTER_AREA)

def preprocess_image(image_file):
    """
    Mengembalikan:
    1. img_vis: Gambar 224x224 untuk visualisasi (float 0-1)
    2. img_rgb_batch: Batch (1, 224, 224, 3) untuk DenseNet/HF ViT
    3. img_gray_batch: Batch (1, 224, 224, 1) untuk CNN/Keras ViT
    4. features_hog: Feature vector untuk model klasik
    """
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Image Processing Pipeline
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = apply_clahe(img)
    img_vis = heuristic_lung_crop(img)
    
    # Prepare Batches
    img_gray_batch = np.expand_dims(img_vis[..., np.newaxis], axis=0)
    img_rgb_batch = np.repeat(img_gray_batch, 3, axis=-1)
    
    # Feature Extraction HOG (Classic)
    img_uint8 = (img_vis * 255).astype(np.uint8)
    features_hog = hog(
        img_uint8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        orientations=9,
        feature_vector=True
    ).reshape(1, -1)
    
    return img_vis, img_rgb_batch, img_gray_batch, features_hog

# ==========================================
# 5. LOADING MODELS (CACHED)
# ==========================================
@st.cache_resource
def load_deep_models():
    models_dict = {}
    
    # 1. Keras Models
    try:
        models_dict["Pure Custom CNN"] = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_PATH, "best_custom_pure_noaug.h5")
        )
        models_dict["Custom CNN (No Aug)"] = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_PATH, "best_custom_lora_noaug.h5"), 
            custom_objects={"LoRADense": LoRADense}
        )
        models_dict["Custom CNN (Augmented)"] = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_PATH, "best_custom_lora_aug.h5"), 
            custom_objects={"LoRADense": LoRADense}
        )
        models_dict["ViT Keras"] = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_PATH, "best_vit_model.h5"), 
            custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder}
        )
        models_dict["DenseNet121 LoRA"] = tf.keras.models.load_model(
            os.path.join(ARTIFACTS_PATH, "best_lora_densenet.h5"), 
            custom_objects={"LoRADense": LoRADense}
        )
    except Exception as e:
        st.error(f"Error loading Keras models: {e}")

    # 2. HF PyTorch Model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=3, ignore_mismatched_sizes=True
        )
        hf_model.load_state_dict(torch.load(os.path.join(ARTIFACTS_PATH, "hf_vit_pretrained_best.pt"), map_location=device))
        hf_model.to(device).eval()
        hf_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        models_dict["HF ViT Pretrained"] = (hf_model, hf_proc, device)
    except Exception as e:
        st.error(f"Error loading HF Model: {e}")
        
    return models_dict

@st.cache_resource
def load_classic_pipeline():
    """
    Try to load classic ML models with sklearn compatibility handling.
    Returns None, None, None if loading fails (sklearn version mismatch).
    """
    try:
        import warnings
        
        # Suppress all warnings during loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Try to load scaler and selector
            scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "feature_scaler_classic.joblib"))
            selector = joblib.load(os.path.join(ARTIFACTS_PATH, "feature_selector_classic.joblib"))
            
            cl_models = {}
            failed_models = []
            
            for name in ["svm_rbf", "random_forest", "knn"]:
                path = os.path.join(ARTIFACTS_PATH, "classic_models", f"{name}.joblib")
                if os.path.exists(path):
                    try:
                        # Attempt to load with full warning suppression
                        model = joblib.load(path)
                        cl_models[name] = model
                    except Exception as e:
                        failed_models.append(name)
                        continue
            
            # If all models failed to load, return None
            if not cl_models:
                return None, None, {}
                
            # If some models failed, note it but continue
            if failed_models:
                print(f"Note: Could not load {', '.join(failed_models)} due to sklearn version mismatch")
            
            return scaler, selector, cl_models
            
    except Exception as e:
        # Silently fail and return None - Deep Learning models will still work
        return None, None, {}

@st.cache_resource
def load_histories():
    hist_files = {
        "Pure Custom CNN": "history_custom_pure_noaug.pkl",
        "Custom CNN (No Aug)": "history_custom_lora_noaug.pkl",
        "Custom CNN (Augmented)": "history_custom_lora_aug.pkl",
        "DenseNet121 LoRA": "history_lora_densenet.pkl",
        "ViT Keras": "history_vit.pkl",
        "HF ViT Pretrained": "history_hf_vit_pretrained.pkl"
    }
    histories = {}
    for name, fname in hist_files.items():
        path = os.path.join(ARTIFACTS_PATH, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                histories[name] = pickle.load(f)
    return histories

# Header Banner (removed - keeping it minimal)

# Check if artifacts folder exists
if not os.path.exists(ARTIFACTS_PATH):
    st.error(f"""
    🚨 **Artifacts folder not found!**
    
    Expected path: `{ARTIFACTS_PATH}`
    
    **Solution:**
    1. Make sure `artifacts/` folder is in your GitHub repository
    2. The folder should contain all model files (.h5, .pt, .pkl, .joblib)
    3. Check if the folder is in the same directory as `app.py`
    
    **Folder structure should be:**
    ```
    results/
    ├── app.py
    ├── artifacts/
    │   ├── *.h5 files
    │   ├── *.pt files
    │   └── *.joblib files
    └── requirements.txt
    ```
    """)
    st.stop()

# Load resources once
with st.spinner("Loading AI models..."):
    deep_models = load_deep_models()
    scaler, selector, classic_models = load_classic_pipeline()
    histories = load_histories()

# Display info about available models
if classic_models is None or len(classic_models) == 0:
    st.info("ℹ️ Classic ML models are unavailable due to sklearn version incompatibility. All Deep Learning models (6 models) are ready to use.", icon="ℹ️")

# ==========================================
# 6. CORE LOGIC
# ==========================================
def get_prediction(model_key, model_obj, img_inputs):
    img_vis, img_rgb, img_gray, feats_hog = img_inputs
    probs = None
    
    # 1. Classic Models
    if model_key in ["svm_rbf", "random_forest", "knn"]:
        # Transform Features
        feats_sel = selector.transform(feats_hog)
        feats_scaled = scaler.transform(feats_sel)
        
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(feats_scaled)[0]
        else:
            # Fallback for models without proba (though SVM/RF usually have it)
            pred = model_obj.predict(feats_scaled)[0]
            probs = np.zeros(3)
            probs[pred] = 1.0

    # 2. HF ViT (PyTorch)
    elif model_key == "HF ViT Pretrained":
        hf_model, hf_proc, device = model_obj
        inputs = hf_proc(images=(img_rgb[0] * 255).astype(np.uint8), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = hf_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # 3. Keras RGB Models
    elif model_key == "DenseNet121 LoRA":
        probs = model_obj.predict(img_rgb, verbose=0)[0]
        
    # 4. Keras Gray Models
    else:
        probs = model_obj.predict(img_gray, verbose=0)[0]
        
    return probs

# ==========================================
# 7. UI LAYOUT
# ==========================================

# --- SIDEBAR ---
st.sidebar.markdown("""
    <div style="text-align: center; padding: 24px 0 20px 0;">
        <img src="https://img.icons8.com/clouds/200/lung.png" width="80" style="
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.1));
        "/>
        <h2 style="
            margin-top: 12px;
            font-size: 1.5rem;
            color: var(--text-primary);
            font-weight: 600;
            letter-spacing: -0.5px;
        ">COVID-19 AI</h2>
        <p style="
            color: var(--text-secondary);
            font-size: 0.8rem;
            margin-top: 4px;
            font-weight: 500;
        ">Medical Detection System</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='margin: 20px 0; border: none; height: 1px; background: var(--glass-border);'>", unsafe_allow_html=True)
st.sidebar.markdown("### Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["🖼️ Detection & Analysis", "📈 Training Metrics"], label_visibility="collapsed")

# --- PAGE 1: DETECTION & ANALYSIS ---
if app_mode == "🖼️ Detection & Analysis":
    st.markdown("""
        <div style="text-align: center; margin-bottom: 48px;">
            <h1 style="
                font-size: 2.75rem;
                margin-bottom: 12px;
                color: #1d1d1f;
                font-weight: 700;
                letter-spacing: -0.5px;
            ">COVID-19 Detection</h1>
            <p style="
                font-size: 1.1rem;
                color: #6e6e73;
                font-weight: 400;
                margin-bottom: 24px;
                line-height: 1.5;
            ">Upload a chest X-ray and analyze with multiple AI models</p>
            <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;">
                <span style="
                    background: rgba(0, 122, 255, 0.1);
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #007aff;
                ">9 AI Models</span>
                <span style="
                    background: rgba(52, 199, 89, 0.1);
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #34c759;
                ">Real-time Analysis</span>
                <span style="
                    background: rgba(255, 149, 0, 0.1);
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #ff9500;
                ">High Accuracy</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Area
    col_input, col_vis = st.columns([1, 1.5])
    
    with col_input:
        st.markdown("""
            <div style="
                padding: 20px;
                background: var(--glass-bg);
                backdrop-filter: blur(40px) saturate(180%);
                -webkit-backdrop-filter: blur(40px) saturate(180%);
                border-radius: 16px;
                border: 1px solid var(--glass-border);
                margin-bottom: 24px;
                box-shadow: var(--shadow-light);
            ">
                <h3 style="
                    margin: 0 0 8px 0;
                    text-align: center;
                    color: var(--text-primary);
                    font-size: 1.25rem;
                    font-weight: 600;
                ">Upload X-Ray</h3>
                <p style="
                    margin: 0;
                    text-align: center;
                    font-size: 0.875rem;
                    color: var(--text-secondary);
                ">JPG, PNG, or JPEG format</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose file", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        # Mode Selection
        st.markdown("""
            <div style="
                padding: 16px;
                background: var(--glass-bg);
                backdrop-filter: blur(40px) saturate(180%);
                -webkit-backdrop-filter: blur(40px) saturate(180%);
                border-radius: 12px;
                border: 1px solid var(--glass-border);
                margin: 24px 0 16px 0;
                box-shadow: var(--shadow-light);
            ">
                <h4 style="
                    margin: 0;
                    text-align: center;
                    color: var(--text-primary);
                    font-size: 1.1rem;
                    font-weight: 600;
                ">Analysis Mode</h4>
            </div>
        """, unsafe_allow_html=True)
        
        run_mode = st.radio(
            "Select mode:",
            ["Single Model", "All Deep Learning", "⚡ All Models"],
            label_visibility="collapsed"
        )
        
        selected_model = None
        if run_mode == "Single Model":
            st.markdown("<br>", unsafe_allow_html=True)
            all_options = list(deep_models.keys()) + (list(classic_models.keys()) if classic_models else [])
            selected_model = st.selectbox("Select Model:", all_options)

    # Process Upload
    if not uploaded_file:
        with col_vis:
            st.markdown("""
                <div style="
                    text-align: center;
                    padding: 60px 30px;
                    background: var(--glass-bg);
                    backdrop-filter: blur(40px) saturate(180%);
                    -webkit-backdrop-filter: blur(40px) saturate(180%);
                    border-radius: 20px;
                    border: 2px dashed rgba(0, 122, 255, 0.3);
                    margin-top: 40px;
                    box-shadow: var(--shadow-light);
                ">
                    <div style="
                        font-size: 4rem;
                        margin-bottom: 20px;
                    ">🏥</div>
                    <h3 style="
                        margin: 0 0 12px 0;
                        color: var(--text-primary);
                        font-size: 1.5rem;
                        font-weight: 600;
                    ">Upload X-Ray to Begin</h3>
                    <p style="
                        margin: 0;
                        font-size: 1rem;
                        color: var(--text-secondary);
                        line-height: 1.6;
                    ">
                        Upload a chest X-ray image for AI analysis.<br>
                        Get instant results from multiple models.
                    </p>
                    <div style="
                        margin-top: 24px;
                        display: flex;
                        justify-content: center;
                        gap: 12px;
                        flex-wrap: wrap;
                    ">
                        <span style="
                            background: rgba(0, 122, 255, 0.1);
                            padding: 6px 14px;
                            border-radius: 16px;
                            font-size: 0.8rem;
                            color: #007aff;
                            font-weight: 500;
                        ">JPG</span>
                        <span style="
                            background: rgba(52, 199, 89, 0.1);
                            padding: 6px 14px;
                            border-radius: 16px;
                            font-size: 0.8rem;
                            color: #34c759;
                            font-weight: 500;
                        ">PNG</span>
                        <span style="
                            background: rgba(255, 149, 0, 0.1);
                            padding: 6px 14px;
                            border-radius: 16px;
                            font-size: 0.8rem;
                            color: #ff9500;
                            font-weight: 500;
                        ">JPEG</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    elif uploaded_file:
        inputs = preprocess_image(uploaded_file)
        if inputs:
            img_vis, _, _, _ = inputs
            with col_vis:
                st.markdown("""
                    <div style="
                        text-align: center;
                        padding: 16px;
                        background: var(--glass-bg);
                        backdrop-filter: blur(40px) saturate(180%);
                        -webkit-backdrop-filter: blur(40px) saturate(180%);
                        border-radius: 16px;
                        border: 1px solid var(--glass-border);
                        margin-bottom: 20px;
                        box-shadow: var(--shadow-light);
                    ">
                        <h4 style="
                            margin: 0 0 8px 0;
                            color: var(--text-primary);
                            font-size: 1.1rem;
                            font-weight: 600;
                        ">Processed Image</h4>
                        <p style="
                            margin: 0;
                            font-size: 0.85rem;
                            color: var(--text-secondary);
                        ">Enhanced with CLAHE + Lung Cropping</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.image(img_vis, caption="", use_column_width=True, clamp=True, channels='GRAY')
                
                st.markdown("""
                    <div style="
                        text-align: center;
                        margin-top: 16px;
                        padding: 12px;
                        background: rgba(52, 199, 89, 0.1);
                        backdrop-filter: blur(40px);
                        border-radius: 12px;
                        border: 1px solid rgba(52, 199, 89, 0.2);
                    ">
                        <p style="
                            margin: 0;
                            font-size: 0.875rem;
                            color: #34c759;
                            font-weight: 500;
                        ">✓ Ready for analysis</p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Run Analysis", use_container_width=True, key="run_analysis_btn", type="primary"):
                st.divider()
                st.subheader("📊 Hasil Analisis")
                
                results_list = []
                
                # Logic Runner
                models_to_run = {}
                
                if run_mode == "Single Model":
                    if selected_model in deep_models:
                        models_to_run[selected_model] = deep_models[selected_model]
                    elif classic_models and selected_model in classic_models:
                        models_to_run[selected_model] = classic_models[selected_model]
                        
                elif run_mode == "All Deep Learning":
                    models_to_run = deep_models
                    
                elif run_mode == "⚡ All Models":
                    models_to_run = {**deep_models, **(classic_models if classic_models else {})}

                # Execution Loop
                st.markdown("""
                    <div style="
                        text-align: center;
                        padding: 20px;
                        background: var(--glass-bg);
                        backdrop-filter: blur(40px) saturate(180%);
                        -webkit-backdrop-filter: blur(40px) saturate(180%);
                        border-radius: 16px;
                        border: 1px solid var(--glass-border);
                        margin-bottom: 20px;
                        box-shadow: var(--shadow-light);
                    ">
                        <div style="font-size: 2.5rem; margin-bottom: 12px;">⚡</div>
                        <h3 style="margin: 0 0 8px 0; color: var(--text-primary); font-weight: 600;">Processing...</h3>
                        <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">
                            Running AI analysis
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                prog_bar = st.progress(0, text="Starting...")
                status_text = st.empty()
                
                for i, (name, model) in enumerate(models_to_run.items()):
                    status_text.markdown(f"""
                        <div style="
                            text-align: center;
                            padding: 10px;
                            background: var(--glass-bg);
                            backdrop-filter: blur(40px);
                            border-radius: 10px;
                            margin: 8px 0;
                        ">
                            <p style="margin: 0; color: var(--text-primary); font-weight: 500; font-size: 0.9rem;">
                                Running {name} ({i + 1}/{len(models_to_run)})
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    probs = get_prediction(name, model, inputs)
                    pred_idx = np.argmax(probs)
                    conf = probs[pred_idx]
                    
                    results_list.append({
                        "Model": name,
                        "Prediction": CLASSES[pred_idx],
                        "Confidence": conf,
                        "COVID": probs[0],
                        "Non-COVID": probs[1],
                        "Normal": probs[2]
                    })
                    
                    progress_pct = (i + 1) / len(models_to_run)
                    prog_bar.progress(progress_pct, text=f"{int(progress_pct * 100)}% Complete")
                
                status_text.empty()
                prog_bar.empty()
                
                # --- VISUALIZATION OF RESULTS ---
                df_res = pd.DataFrame(results_list).sort_values("Confidence", ascending=False)
                
                # Success message
                st.markdown("""
                    <div style="
                        text-align: center;
                        padding: 20px;
                        background: rgba(52, 199, 89, 0.1);
                        backdrop-filter: blur(40px);
                        border-radius: 16px;
                        border: 1px solid rgba(52, 199, 89, 0.2);
                        margin-bottom: 32px;
                    ">
                        <h3 style="margin: 0; color: #34c759; font-weight: 600;">Analysis Complete</h3>
                        <p style="margin: 8px 0 0 0; color: var(--text-secondary); font-size: 0.9rem;">All models have processed the image</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 1. Best Result Highlight
                best_model = df_res.iloc[0]
                label_color = "pred-covid" if best_model['Prediction'] == "COVID-19" else "pred-non" if best_model['Prediction'] == "Non-COVID" else "pred-normal"
                
                # Get icon based on prediction
                pred_icons = {
                    "COVID-19": "🦠",
                    "Non-COVID": "⚠️",
                    "Normal": "✓"
                }
                icon = pred_icons.get(best_model['Prediction'], "?")
                
                col_best1, col_best2 = st.columns([1, 2])
                with col_best1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Best Prediction</div>
                        <div style="
                            font-size: 3rem;
                            margin: 16px 0;
                        ">{icon}</div>
                        <div class="prediction-box {label_color}">{best_model['Prediction']}</div>
                        <div class="metric-value">{best_model['Confidence']:.1%}</div>
                        <div style="
                            margin-top: 12px;
                            padding: 8px;
                            background: rgba(0,0,0,0.04);
                            border-radius: 8px;
                            font-size: 0.85rem;
                            color: var(--text-secondary);
                        ">
                            {best_model['Model']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_best2:
                    st.markdown("""
                        <h3 style="
                            text-align: center;
                            margin-bottom: 20px;
                            font-size: 1.25rem;
                            color: var(--text-primary);
                            font-weight: 600;
                        ">Probability Distribution</h3>
                    """, unsafe_allow_html=True)
                    
                    # Stacked bar chart for detailed probs
                    df_chart = df_res.set_index("Model")[["COVID", "Non-COVID", "Normal"]]
                    st.bar_chart(df_chart, height=280, use_container_width=True)
                
                # 2. Statistics Cards
                st.markdown("<br>", unsafe_allow_html=True)
                stat_cols = st.columns(4)
                
                with stat_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Models</div>
                        <div class="metric-value">{len(df_res)}</div>
                        <small style="color: var(--text-secondary); font-size: 0.75rem;">Executed</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_cols[1]:
                    avg_conf = df_res['Confidence'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Confidence</div>
                        <div class="metric-value">{avg_conf:.0%}</div>
                        <small style="color: var(--text-secondary); font-size: 0.75rem;">Mean Score</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_cols[2]:
                    consensus = df_res['Prediction'].mode()[0] if len(df_res) > 0 else "N/A"
                    consensus_count = (df_res['Prediction'] == consensus).sum()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Consensus</div>
                        <div class="metric-value">{consensus_count}/{len(df_res)}</div>
                        <small style="color: var(--text-secondary); font-size: 0.75rem;">{consensus}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with stat_cols[3]:
                    std_conf = df_res['Confidence'].std()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Std Deviation</div>
                        <div class="metric-value">{std_conf:.1%}</div>
                        <small style="color: var(--text-secondary); font-size: 0.75rem;">Spread</small>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. Detailed Table
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <h3 style="
                        text-align: center;
                        margin: 32px 0 20px 0;
                        font-size: 1.5rem;
                        color: var(--text-primary);
                        font-weight: 600;
                    ">Detailed Results</h3>
                """, unsafe_allow_html=True)
                
                st.dataframe(
                    df_res.style.background_gradient(subset=["Confidence"], cmap="Blues", vmin=0, vmax=1)
                          .background_gradient(subset=["COVID"], cmap="Reds", vmin=0, vmax=1)
                          .background_gradient(subset=["Non-COVID"], cmap="Oranges", vmin=0, vmax=1)
                          .background_gradient(subset=["Normal"], cmap="Greens", vmin=0, vmax=1)
                          .format({"Confidence": "{:.1%}", "COVID": "{:.1%}", "Non-COVID": "{:.1%}", "Normal": "{:.1%}"}),
                    use_container_width=True,
                    height=350
                )

# --- PAGE 2: TRAINING METRICS ---
elif app_mode == "📈 Training Metrics":
    st.markdown("""
        <div style="text-align: center; margin-bottom: 48px;">
            <h1 style="
                font-size: 2.75rem;
                margin-bottom: 12px;
                color: var(--text-primary);
                font-weight: 700;
                letter-spacing: -0.5px;
            ">Training Metrics</h1>
            <p style="
                font-size: 1.1rem;
                color: var(--text-secondary);
                font-weight: 400;
            ">Deep dive into model training performance and evaluation</p>
            <div style="display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-top: 24px;">
                <span style="
                    background: rgba(0, 122, 255, 0.1);
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #007aff;
                ">Training Curves</span>
                <span style="
                    background: rgba(52, 199, 89, 0.1);
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #34c759;
                ">Loss Analysis</span>
                <span style="
                    background: rgba(255, 149, 0, 0.1);
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.875rem;
                    font-weight: 500;
                    color: #ff9500;
                ">Evaluation Reports</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Training Curves", "Evaluation Reports"])
    
    with tab1:
        st.markdown("""
            <div style="
                text-align: center;
                padding: 20px;
                background: var(--glass-bg);
                backdrop-filter: blur(40px) saturate(180%);
                -webkit-backdrop-filter: blur(40px) saturate(180%);
                border-radius: 16px;
                border: 1px solid var(--glass-border);
                margin-bottom: 28px;
                box-shadow: var(--shadow-light);
            ">
                <p style="
                    margin: 0;
                    font-size: 1rem;
                    color: var(--text-primary);
                ">Select a model to view <strong>Accuracy</strong> and <strong>Loss</strong> curves during training</p>
            </div>
        """, unsafe_allow_html=True)
        
        model_hist_choice = st.selectbox("Select Model:", list(histories.keys()))
        
        if model_hist_choice:
            hist = histories[model_hist_choice]
            
            # Extract data
            acc = hist.get('accuracy', hist.get('train_acc', []))
            val_acc = hist.get('val_accuracy', hist.get('val_acc', []))
            loss = hist.get('loss', hist.get('train_loss', []))
            val_loss = hist.get('val_loss', hist.get('val_loss', []))
            epochs = range(1, len(acc) + 1)
            
            # Summary Cards
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Epochs</div>
                    <div class="metric-value">{len(acc)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_cols[1]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Train Acc</div>
                    <div class="metric-value">{max(acc):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_cols[2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Best Val Acc</div>
                    <div class="metric-value">{max(val_acc):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with summary_cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Final Val Loss</div>
                    <div class="metric-value">{val_loss[-1]:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_g1, col_g2 = st.columns(2)
            
            # Configure clean matplotlib style with transparent backgrounds for dark mode support
            plt.style.use('default')
            
            # Plot Accuracy
            with col_g1:
                fig1, ax1 = plt.subplots(figsize=(7, 4.5), facecolor='none')
                ax1.set_facecolor('none')
                ax1.patch.set_alpha(0)
                fig1.patch.set_alpha(0)
                
                ax1.plot(epochs, acc, color='#0a84ff', linewidth=2.5, marker='o', 
                        markersize=6, label='Training', alpha=0.9)
                ax1.plot(epochs, val_acc, color='#30d158', linewidth=2.5, marker='s', 
                        markersize=6, label='Validation', alpha=0.9)
                
                ax1.set_title(f"Accuracy: {model_hist_choice}", 
                            fontsize=14, fontweight='600', pad=15)
                ax1.set_xlabel("Epochs", fontsize=11)
                ax1.set_ylabel("Accuracy", fontsize=11)
                ax1.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, framealpha=0.9)
                ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
                ax1.tick_params(labelsize=9)
                
                # Minimal spines
                for spine in ax1.spines.values():
                    spine.set_linewidth(1)
                    spine.set_alpha(0.3)
                
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()
                
            # Plot Loss
            with col_g2:
                fig2, ax2 = plt.subplots(figsize=(7, 4.5), facecolor='none')
                ax2.set_facecolor('none')
                ax2.patch.set_alpha(0)
                fig2.patch.set_alpha(0)
                
                ax2.plot(epochs, loss, color='#0a84ff', linewidth=2.5, marker='o', 
                        markersize=6, label='Training', alpha=0.9)
                ax2.plot(epochs, val_loss, color='#ff453a', linewidth=2.5, marker='s', 
                        markersize=6, label='Validation', alpha=0.9)
                
                ax2.set_title(f"Loss: {model_hist_choice}", 
                            fontsize=14, fontweight='600', pad=15)
                ax2.set_xlabel("Epochs", fontsize=11)
                ax2.set_ylabel("Loss", fontsize=11)
                ax2.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, framealpha=0.9)
                ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
                ax2.tick_params(labelsize=9)
                
                # Minimal spines
                for spine in ax2.spines.values():
                    spine.set_linewidth(1)
                    spine.set_alpha(0.3)
                
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            
            # Final metrics
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 24px;
                background: var(--glass-bg);
                backdrop-filter: blur(40px) saturate(180%);
                -webkit-backdrop-filter: blur(40px) saturate(180%);
                border-radius: 16px;
                border: 1px solid var(--glass-border);
                margin-top: 28px;
                box-shadow: var(--shadow-light);
            ">
                <h3 style="margin: 0 0 16px 0; color: var(--text-primary); font-weight: 600; font-size: 1.2rem;">Final Results</h3>
                <div style="
                    display: flex;
                    justify-content: center;
                    gap: 40px;
                    flex-wrap: wrap;
                    font-size: 1rem;
                ">
                    <div>
                        <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 4px;">Validation Accuracy</div>
                        <span style="
                            color: var(--accent-green);
                            font-size: 1.5rem;
                            font-weight: 600;
                        ">{val_acc[-1]:.1%}</span>
                    </div>
                    <div>
                        <div style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 4px;">Validation Loss</div>
                        <span style="
                            color: var(--accent-red);
                            font-size: 1.5rem;
                            font-weight: 600;
                        ">{val_loss[-1]:.3f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
            <div style="
                text-align: center;
                padding: 20px;
                background: var(--glass-bg);
                backdrop-filter: blur(40px) saturate(180%);
                -webkit-backdrop-filter: blur(40px) saturate(180%);
                border-radius: 16px;
                border: 1px solid var(--glass-border);
                margin-bottom: 28px;
                box-shadow: var(--shadow-light);
            ">
                <h3 style="margin: 0 0 8px 0; color: var(--text-primary); font-weight: 600;">Classification Reports</h3>
                <p style="
                    margin: 0;
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                ">Detailed evaluation metrics for each AI model</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Load CSVs from artifacts
        import glob
        report_files = glob.glob(os.path.join(ARTIFACTS_PATH, "classification_report*.csv"))
        
        if report_files:
            selected_report = st.selectbox("Select Report:", [os.path.basename(f) for f in report_files])
            df_rep = pd.read_csv(os.path.join(ARTIFACTS_PATH, selected_report), index_col=0)
            
            # Display styled dataframe
            st.markdown("""
                <h4 style="text-align: center; margin: 28px 0 20px 0; color: var(--text-primary); font-weight: 600;">
                    Classification Metrics
                </h4>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                df_rep.style
                    .background_gradient(subset=['precision'], cmap='Blues', vmin=0, vmax=1)
                    .background_gradient(subset=['recall'], cmap='Greens', vmin=0, vmax=1)
                    .background_gradient(subset=['f1-score'], cmap='Oranges', vmin=0, vmax=1)
                    .format("{:.3f}"),
                use_container_width=True,
                height=350
            )
            
            # Summary statistics if available
            if 'f1-score' in df_rep.columns:
                avg_f1 = df_rep['f1-score'].mean()
                st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 20px;
                    background: var(--glass-bg);
                    backdrop-filter: blur(40px) saturate(180%);
                    -webkit-backdrop-filter: blur(40px) saturate(180%);
                    border-radius: 16px;
                    border: 1px solid var(--glass-border);
                    margin-top: 24px;
                    box-shadow: var(--shadow-light);
                ">
                    <h4 style="margin: 0 0 12px 0; color: var(--text-primary); font-weight: 600;">Average F1-Score</h4>
                    <div style="font-size: 2.5rem; font-weight: 600; color: var(--accent-blue);">
                        {avg_f1:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="
                    text-align: center;
                    padding: 40px;
                    background: var(--glass-bg);
                    backdrop-filter: blur(40px) saturate(180%);
                    -webkit-backdrop-filter: blur(40px) saturate(180%);
                    border-radius: 16px;
                    border: 1px solid var(--glass-border);
                    margin-top: 28px;
                    box-shadow: var(--shadow-light);
                ">
                    <div style="font-size: 3.5rem; margin-bottom: 16px;">📄</div>
                    <h3 style="margin: 0 0 12px 0; color: var(--text-primary); font-weight: 600;">No Reports Found</h3>
                    <p style="margin: 0; color: var(--text-secondary);">
                        No classification reports found in artifacts folder.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="
        margin-top: 60px;
        padding: 32px 20px;
        text-align: center;
    ">
        <p style="
            margin: 0;
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-weight: 400;
        ">
            Powered by Streamlit • TensorFlow • PyTorch • Scikit-Learn
        </p>
        <p style="
            margin: 12px 0 0 0;
            font-size: 0.85rem;
            color: var(--text-secondary);
        ">
            © 2025 COVID-19 Detection System
        </p>
    </div>
""", unsafe_allow_html=True)

# Add minimal sidebar info
st.sidebar.markdown("<hr style='margin: 24px 0; border: none; height: 1px; background: var(--glass-border);'>", unsafe_allow_html=True)

# Count available models
total_models = len(deep_models) + (len(classic_models) if classic_models else 0)
models_text = f"{total_models} AI Models" if classic_models else f"{len(deep_models)} Deep Learning Models"

st.sidebar.markdown(f"""
    <div style="
        padding: 16px;
        background: var(--glass-bg);
        backdrop-filter: blur(40px) saturate(180%);
        -webkit-backdrop-filter: blur(40px) saturate(180%);
        border-radius: 12px;
        border: 1px solid var(--glass-border);
        margin-top: 16px;
        box-shadow: var(--shadow-light);
    ">
        <h4 style="
            margin: 0 0 12px 0;
            text-align: center;
            color: var(--text-primary);
            font-size: 1rem;
            font-weight: 600;
        ">System Info</h4>
        <div style="font-size: 0.85rem; line-height: 1.8; color: var(--text-primary);">
            <div style="margin: 6px 0; padding: 8px; background: rgba(128,128,128,0.08); border-radius: 8px;">
                <strong>Models:</strong> {models_text}
            </div>
            <div style="margin: 6px 0; padding: 8px; background: rgba(128,128,128,0.08); border-radius: 8px;">
                <strong>Input:</strong> X-Ray Images
            </div>
            <div style="margin: 6px 0; padding: 8px; background: rgba(128,128,128,0.08); border-radius: 8px;">
                <strong>Classes:</strong> 3 Categories
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
