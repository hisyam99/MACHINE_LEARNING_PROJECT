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

# Custom CSS untuk tampilan Modern & Clean
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    h2, h3 { color: #34495e; }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4b6cb7;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }
    .pred-covid { background-color: #e74c3c; }
    .pred-non { background-color: #f39c12; }
    .pred-normal { background-color: #27ae60; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. KONSTANTA & CONFIG
# ==========================================
CLASSES = {0: "COVID-19", 1: "Non-COVID", 2: "Normal"}
TARGET_SIZE = (224, 224)
ARTIFACTS_PATH = "./artifacts"

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
    try:
        scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "feature_scaler_classic.joblib"))
        selector = joblib.load(os.path.join(ARTIFACTS_PATH, "feature_selector_classic.joblib"))
        
        cl_models = {}
        for name in ["svm_rbf", "random_forest", "knn"]:
            path = os.path.join(ARTIFACTS_PATH, "classic_models", f"{name}.joblib")
            if os.path.exists(path):
                cl_models[name] = joblib.load(path)
        
        return scaler, selector, cl_models
    except Exception as e:
        st.warning(f"Classic pipeline artifacts not found: {e}")
        return None, None, None

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

# Load resources once
deep_models = load_deep_models()
scaler, selector, classic_models = load_classic_pipeline()
histories = load_histories()

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
st.sidebar.image("https://img.icons8.com/clouds/200/lung.png", width=100)
st.sidebar.title("Navigasi")
app_mode = st.sidebar.radio("Pilih Mode:", ["🖼️ Deteksi & Analisis", "📈 Training Metrics & Graphs"])

# --- PAGE 1: DETEKSI & ANALISIS ---
if app_mode == "🖼️ Deteksi & Analisis":
    st.title("🖼️ Deteksi COVID-19 Multi-Model")
    st.markdown("Upload X-Ray Anda dan jalankan analisis menggunakan **Machine Learning Klasik** dan **Deep Learning**.")
    
    # Input Area
    col_input, col_vis = st.columns([1, 1.5])
    
    with col_input:
        uploaded_file = st.file_uploader("📂 Upload X-Ray Image", type=["jpg", "png", "jpeg"])
        
        # Mode Seleksi
        run_mode = st.radio(
            "⚙️ Mode Eksekusi:",
            ["Single Model", "All Classic ML", "All Deep Learning", "⚡ RUN EVERYTHING"]
        )
        
        selected_model = None
        if run_mode == "Single Model":
            all_options = list(deep_models.keys()) + (list(classic_models.keys()) if classic_models else [])
            selected_model = st.selectbox("Pilih Model:", all_options)

    # Process Upload
    if uploaded_file:
        inputs = preprocess_image(uploaded_file)
        if inputs:
            img_vis, _, _, _ = inputs
            with col_vis:
                st.image(img_vis, caption="Processed Image (CLAHE + Lung Crop)", width=350, clamp=True, channels='GRAY')
                
            if st.button("🚀 JALANKAN ANALISIS", use_container_width=True):
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
                        
                elif run_mode == "All Classic ML":
                    if classic_models: models_to_run = classic_models
                    else: st.error("Model Klasik tidak ditemukan.")
                    
                elif run_mode == "All Deep Learning":
                    models_to_run = deep_models
                    
                elif run_mode == "⚡ RUN EVERYTHING":
                    models_to_run = {**deep_models, **(classic_models if classic_models else {})}

                # Execution Loop
                prog_bar = st.progress(0)
                for i, (name, model) in enumerate(models_to_run.items()):
                    probs = get_prediction(name, model, inputs)
                    pred_idx = np.argmax(probs)
                    conf = probs[pred_idx]
                    
                    results_list.append({
                        "Model": name,
                        "Prediksi": CLASSES[pred_idx],
                        "Confidence": conf,
                        "Prob COVID": probs[0],
                        "Prob Non-COVID": probs[1],
                        "Prob Normal": probs[2]
                    })
                    prog_bar.progress((i + 1) / len(models_to_run))
                
                # --- VISUALIZATION OF RESULTS ---
                df_res = pd.DataFrame(results_list).sort_values("Confidence", ascending=False)
                
                # 1. Best Result Highlight
                best_model = df_res.iloc[0]
                label_color = "pred-covid" if best_model['Prediksi'] == "COVID-19" else "pred-non" if best_model['Prediksi'] == "Non-COVID" else "pred-normal"
                
                col_best1, col_best2 = st.columns([1, 2])
                with col_best1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Konsensus Terbaik</div>
                        <div class="prediction-box {label_color}">{best_model['Prediksi']}</div>
                        <div class="metric-value">{best_model['Confidence']:.2%}</div>
                        <small>by {best_model['Model']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_best2:
                    st.write("#### 📈 Perbandingan Probabilitas Semua Model")
                    # Stacked bar chart for detailed probs
                    df_chart = df_res.set_index("Model")[["Prob COVID", "Prob Non-COVID", "Prob Normal"]]
                    st.bar_chart(df_chart, height=250)

                # 2. Detailed Table
                st.write("#### 📋 Tabel Detail")
                st.dataframe(
                    df_res.style.background_gradient(subset=["Confidence"], cmap="Greens")
                          .format({"Confidence": "{:.2%}", "Prob COVID": "{:.2%}", "Prob Non-COVID": "{:.2%}", "Prob Normal": "{:.2%}"}),
                    use_container_width=True
                )

# --- PAGE 2: TRAINING METRICS ---
elif app_mode == "📈 Training Metrics & Graphs":
    st.title("📈 Performance Metrics & Training History")
    
    tab1, tab2 = st.tabs(["📉 Training Curves (Loss/Acc)", "📑 Evaluation Reports"])
    
    with tab1:
        st.markdown("Pilih model untuk melihat grafik **Accuracy** dan **Loss** selama proses training.")
        model_hist_choice = st.selectbox("Pilih Model:", list(histories.keys()))
        
        if model_hist_choice:
            hist = histories[model_hist_choice]
            
            # Extract data
            acc = hist.get('accuracy', hist.get('train_acc', []))
            val_acc = hist.get('val_accuracy', hist.get('val_acc', []))
            loss = hist.get('loss', hist.get('train_loss', []))
            val_loss = hist.get('val_loss', hist.get('val_loss', []))
            epochs = range(1, len(acc) + 1)
            
            col_g1, col_g2 = st.columns(2)
            
            # Plot Accuracy
            with col_g1:
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.plot(epochs, acc, 'b-o', label='Training Acc')
                ax1.plot(epochs, val_acc, 'r-o', label='Validation Acc')
                ax1.set_title(f"Accuracy Curve: {model_hist_choice}")
                ax1.set_xlabel("Epochs")
                ax1.set_ylabel("Accuracy")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
            # Plot Loss
            with col_g2:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(epochs, loss, 'b-o', label='Training Loss')
                ax2.plot(epochs, val_loss, 'r-o', label='Validation Loss')
                ax2.set_title(f"Loss Curve: {model_hist_choice}")
                ax2.set_xlabel("Epochs")
                ax2.set_ylabel("Loss")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
            st.info(f"Final Validation Accuracy: **{val_acc[-1]:.2%}** | Final Validation Loss: **{val_loss[-1]:.4f}**")

    with tab2:
        st.markdown("### Laporan Klasifikasi (Test Set)")
        # Load CSVs from artifacts
        import glob
        report_files = glob.glob(os.path.join(ARTIFACTS_PATH, "classification_report*.csv"))
        
        if report_files:
            selected_report = st.selectbox("Pilih Laporan:", [os.path.basename(f) for f in report_files])
            df_rep = pd.read_csv(os.path.join(ARTIFACTS_PATH, selected_report), index_col=0)
            st.dataframe(df_rep.style.format("{:.4f}"))
        else:
            st.warning("Belum ada file classification report yang ditemukan di folder artifacts.")

# Footer
st.markdown("---")
st.markdown("<center><small>Developed with ❤️ using Streamlit, TensorFlow, PyTorch & Scikit-Learn</small></center>", unsafe_allow_html=True)
