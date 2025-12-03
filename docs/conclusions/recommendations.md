# Rekomendasi

## 📋 Ringkasan Eksekutif

Berdasarkan hasil eksperimen komprehensif yang meliputi machine learning klasik (HOG+SVM, Random Forest, kNN), deep learning from scratch (Custom CNN + LoRA), dan transfer learning (DenseNet121 + LoRA, Vision Transformer), berikut adalah rekomendasi praktis untuk berbagai skenario penggunaan.

## 🎯 Rekomendasi Berdasarkan Use Case

### 1. Untuk Akurasi Maksimal dan Clinical Decision Support

**Model yang Direkomendasikan:** **HuggingFace ViT Pretrained**

**Spesifikasi:**
- Model: `google/vit-base-patch16-224-in21k`
- Akurasi: **91.65%**
- Macro F1: **0.9017**
- F1 COVID-19: **0.9601** (sensitivitas sangat tinggi)

**Kelebihan:**
- ✅ Performa terbaik di semua metrik
- ✅ Keseimbangan kelas sangat baik (F1 >0.87 untuk semua kelas)
- ✅ Non-COVID akhirnya dapat ditangani dengan baik (F1 0.87)
- ✅ Cocok untuk clinical decision support yang memprioritaskan akurasi

**Trade-off:**
- ⚠️ Resource intensive (~350 MB model size)
- ⚠️ Memerlukan GPU untuk training (2-3 jam)
- ⚠️ Inference time ~50-100 ms per image
- ⚠️ Kompleksitas deployment lebih tinggi

**Rekomendasi Implementasi:**
```python
# Load model
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=3
)
# Fine-tune dengan LR 2e-5, class weights, early stopping
```

**Use Case Ideal:**
- Hospital systems dengan resource GPU
- Cloud-based diagnostic services
- Research dan clinical trials
- Situations where accuracy is paramount

---

### 2. Untuk Baseline Kuat Tanpa GPU

**Model yang Direkomendasikan:** **SVM RBF dengan HOG Features**

**Spesifikasi:**
- Akurasi: **86.27%** (terbaik kedua, terbaik untuk klasik)
- Macro F1: **0.843**
- Model size: ~95 MB
- Training: CPU sufficient (~10-15 menit)

**Kelebihan:**
- ✅ Tidak perlu GPU (CPU sufficient)
- ✅ Akurasi sangat baik (86.27%)
- ✅ Training cepat dibanding deep learning
- ✅ Feature engineering eksplisit (HOG) memberikan interpretabilitas
- ✅ Stable dan konsisten

**Trade-off:**
- ⚠️ Feature engineering manual diperlukan
- ⚠️ Model size relatif besar (~95 MB untuk menyimpan support vectors)
- ⚠️ Inference sedikit lebih lambat (~15 ms per image)

**Rekomendasi Implementasi:**
```python
# HOG extraction
from skimage.feature import hog
features = hog(image, orientations=9, pixels_per_cell=(16,16), 
               cells_per_block=(2,2))

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=4096).fit(X_train, y_train)

# SVM training
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=10, gamma='scale', 
          class_weight='balanced', probability=True)
```

**Use Case Ideal:**
- Systems tanpa GPU
- Rapid prototyping dan baseline establishment
- Resource-constrained environments
- Feature engineering research

---

### 3. Untuk Deployment Mobile/Edge Devices

**Model yang Direkomendasikan:** **Custom CNN + LoRA dengan Augmentation**

**Spesifikasi:**
- Akurasi: **81.35%**
- Macro F1: **0.7825**
- Model size: **~1.8 MB** (sangat lightweight)
- Training: ~30-45 menit pada GPU sedang

**Kelebihan:**
- ✅ **Sangat ringan** (~1.8 MB - cocok untuk mobile)
- ✅ Inference cepat (cocok untuk real-time)
- ✅ End-to-end learning (no manual feature engineering)
- ✅ Dapat di-quantize lebih lanjut (FP16/INT8)

**Trade-off:**
- ⚠️ Akurasi lebih rendah dibanding SVM dan ViT
- ⚠️ Perlu data augmentation yang tepat
- ⚠️ Struggles dengan Non-COVID class (F1 0.66)

**Rekomendasi Implementasi:**
```python
# Model architecture
model = Sequential([
    # 4 Conv blocks
    Conv2D(32, 3x3) -> BN -> ReLU -> MaxPool,
    Conv2D(64, 3x3) -> BN -> ReLU -> MaxPool,
    Conv2D(128, 3x3) -> BN -> ReLU -> MaxPool,
    Conv2D(256, 3x3) -> BN -> ReLU -> MaxPool,
    GlobalAveragePooling2D(),
    LoRADense(128, rank=4),
    LoRADense(3, rank=4, activation='softmax')
])

# Augmentation wajib
augmentation = RandomRotation(0.15) + RandomZoom(0.1) + RandomFlip()
```

**Use Case Ideal:**
- Mobile applications (Android/iOS)
- Edge devices (Raspberry Pi, NVIDIA Jetson)
- IoT medical devices
- Situations dengan storage dan compute terbatas

---

### 4. Untuk Keseimbangan Performa dan Efisiensi

**Model yang Direkomendasikan:** **DenseNet121 + LoRA**

**Spesifikasi:**
- Akurasi: **82.04%**
- Macro F1: **0.8003**
- Model size: ~29 MB
- Training: ~1-2 jam pada GPU

**Kelebihan:**
- ✅ Keseimbangan baik antara akurasi dan efisiensi
- ✅ Transfer learning yang proven
- ✅ Mudah diintegrasikan dengan Grad-CAM untuk interpretability
- ✅ Stable dan konsisten

**Trade-off:**
- ⚠️ Masih memerlukan GPU untuk training yang optimal
- ⚠️ Model size lebih besar dari Custom CNN

**Rekomendasi Implementasi:**
```python
# Base model (frozen)
base = DenseNet121(weights='imagenet', include_top=False)
base.trainable = False

# LoRA head
x = GlobalAveragePooling2D()(base.output)
x = LoRADense(256, rank=8, alpha=64)(x)
outputs = LoRADense(3, rank=8, alpha=64, activation='softmax')(x)

# Fine-tune dengan LR kecil
optimizer = Adam(1e-4)
```

**Use Case Ideal:**
- Hospital systems dengan GPU sedang
- Balanced deployment (tidak terlalu resource intensive)
- Systems yang memerlukan interpretability (Grad-CAM)
- Production systems dengan resource moderate

---

## 🔬 Rekomendasi untuk Pengembangan Lebih Lanjut

### 1. Ensemble Modeling

Berdasarkan analisis bahwa setiap model memiliki pola error yang berbeda, **ensemble** dari beberapa model top dapat meningkatkan performa lebih jauh:

**Recommended Ensemble:**
```
Ensemble = 0.5 × HF ViT + 0.3 × SVM + 0.2 × DenseNet121
```

**Expected Improvement:**
- Akurasi potensial: >92%
- Robustness meningkat
- Confidence calibration lebih baik

**Implementation Strategy:**
- Voting/averaging untuk prediksi final
- Weighted voting berdasarkan per-class confidence
- Cascade: SVM untuk quick screening, ViT untuk borderline cases

### 2. Model Compression dan Optimization

Untuk deployment yang lebih efisien:

**Quantization:**
```python
# TensorFlow Lite quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
# Expected: 4x size reduction, minimal accuracy loss
```

**Knowledge Distillation:**
- Teacher: HF ViT Pretrained (91.65%)
- Student: Custom CNN
- Expected student accuracy: 83-85%

**Pruning:**
- Structured pruning pada Custom CNN
- Target: 50% parameter reduction
- Expected accuracy drop: <2%

### 3. Interpretability dan Explainability

Untuk clinical acceptance, tambahkan layer interpretability:

**Grad-CAM untuk CNN:**
```python
from tf_keras_vis.gradcam import Gradcam
gradcam = Gradcam(model)
heatmap = gradcam(image, class_idx)
# Overlay heatmap on original X-ray
```

**Attention Visualization untuk ViT:**
```python
# Visualize attention weights
attention_maps = model.get_attention_weights()
# Show which patches model focuses on
```

**SHAP for SVM:**
```python
import shap
explainer = shap.KernelExplainer(svm.predict_proba, X_train)
shap_values = explainer.shap_values(X_test)
```

### 4. Data Enhancement

**Menggunakan Lung Segmentation Masks:**
- Dataset COVID-QU-Ex menyediakan lung masks
- Dapat digunakan untuk:
  - Automatic ROI cropping (lebih akurat)
  - Mask-guided attention
  - Infection segmentation

**Synthetic Data Augmentation:**
- Generative models (GANs, Diffusion) untuk sintesis X-ray COVID-19
- Expected impact: +3-5% accuracy pada minority classes

**Multi-task Learning:**
- Simultaneous classification + segmentation
- Share lower layers, separate heads
- Potential: Improved feature learning

### 5. Clinical Integration

**Confidence Thresholding:**
```python
if prediction_prob < 0.7:
    flag_for_manual_review()
elif prediction == 'COVID-19' and prob > 0.95:
    prioritize_for_immediate_action()
```

**Risk Stratification:**
- Low risk: Normal dengan confidence >0.9
- Medium risk: Non-COVID atau confidence 0.7-0.9
- High risk: COVID-19 atau confidence <0.7

**Reporting System:**
- Auto-generate report dengan confidence scores
- Include top-3 predictions dengan probabilities
- Highlight regions of interest (Grad-CAM/attention)

---

## 📊 Benchmarking dan Continuous Improvement

### Monitoring Metrics

Track performance over time:
```python
metrics_to_monitor = [
    'accuracy', 'macro_f1', 'per_class_f1',
    'false_positive_rate', 'false_negative_rate',
    'inference_latency', 'model_size'
]
```

### A/B Testing in Production

- Deploy multiple models in parallel
- Route traffic based on criteria
- Collect feedback from radiologists
- Update model weights based on real-world performance

### Continuous Learning

- Collect misclassified cases
- Fine-tune model quarterly
- Monitor for distribution shift
- Retrain dengan data baru

---

## 🎯 Kesimpulan Rekomendasi

| Scenario | Best Model | Akurasi | Trade-off |
|:---------|:-----------|:--------|:----------|
| **Max Accuracy** | HF ViT Pretrained | 91.65% | Resource intensive |
| **No GPU Available** | SVM RBF | 86.27% | Manual feature engineering |
| **Mobile/Edge** | Custom CNN + LoRA | 81.35% | Lower accuracy |
| **Balanced** | DenseNet121 + LoRA | 82.04% | Moderate resource |
| **Production** | Ensemble (ViT+SVM+DenseNet) | ~92%+ | High complexity |

**Final Recommendation:**

Untuk **production clinical system**, kami merekomendasikan **two-stage approach**:

1. **Stage 1 - Fast Screening:** Custom CNN atau SVM untuk rapid triaging
2. **Stage 2 - Accurate Diagnosis:** HF ViT Pretrained untuk kasus yang flagged atau borderline

Approach ini mengoptimalkan trade-off antara throughput, akurasi, dan resource usage.

## 🎯 Rekomendasi Berdasarkan Use Case

### Untuk Akurasi Maksimal

**Model:** HuggingFace ViT Pretrained

**Spesifikasi:**
- **Akurasi:** 91.65%
- **Macro F1:** 0.9017
- **F1 COVID-19:** 0.9601
- **Model Size:** ~350 MB
- **Training Time:** ~2-3 jam (GPU)

**Kapan digunakan:**
- Prioritas utama adalah akurasi
- Resources (GPU, memory) tersedia
- Tidak ada constraint ukuran model
- Use case: Research, high-accuracy screening

**Trade-off:**
- ✅ Akurasi tertinggi
- ✅ Keseimbangan kelas sangat baik
- ⚠️ Resource intensive
- ⚠️ Model size besar

### Untuk Keseimbangan Performa-Efisiensi

**Model:** DenseNet121 + LoRA

**Spesifikasi:**
- **Akurasi:** 82.04%
- **Macro F1:** 0.8003
- **Model Size:** ~30 MB
- **Training Time:** ~1-2 jam (GPU)

**Kapan digunakan:**
- Perlu keseimbangan antara akurasi dan efisiensi
- Resources terbatas tapi masih ada GPU
- Perlu model yang tidak terlalu besar
- Use case: Production deployment, balanced requirements

**Trade-off:**
- ✅ Keseimbangan baik
- ✅ Efisien parameter dengan LoRA
- ✅ Training lebih cepat
- ⚠️ Akurasi lebih rendah dari HF ViT

### Untuk Deployment Mobile/Edge

**Model:** Custom CNN + LoRA dengan Augmentation

**Spesifikasi:**
- **Akurasi:** 81.35%
- **Macro F1:** 0.7825
- **Model Size:** ~1.8 MB (sangat ringan)
- **Training Time:** ~30-45 menit (GPU recommended)

**Kapan digunakan:**
- Deployment di perangkat mobile/edge
- Constraint ukuran model sangat ketat
- Perlu inference cepat
- Use case: Mobile apps, IoT devices, edge computing

**Trade-off:**
- ✅ Sangat ringan (~1.8 MB)
- ✅ Inference cepat
- ✅ Cocok untuk mobile/edge
- ⚠️ Akurasi lebih rendah
- ⚠️ Perlu data augmentation

### Untuk Baseline Tanpa GPU

**Model:** SVM dengan HOG features

**Spesifikasi:**
- **Akurasi:** 86.27%
- **Macro F1:** 0.843
- **Model Size:** ~95 MB
- **Training Time:** ~10-15 menit (CPU)

**Kapan digunakan:**
- Tidak ada akses GPU
- Perlu baseline yang cepat
- Perlu interpretability
- Use case: Quick prototyping, CPU-only environments

**Trade-off:**
- ✅ Tidak perlu GPU
- ✅ Training cepat
- ✅ Stabil dan konsisten
- ⚠️ Ukuran model relatif besar
- ⚠️ Perlu feature engineering manual

## 📋 Rekomendasi Implementasi

### 1. Untuk Model From Scratch

**Selalu gunakan data augmentation:**
- Random Rotation (±15°)
- Random Zoom (0.9-1.1)
- Random Brightness/Contrast
- Horizontal Flip (50%)

**Alasan:**
- Meningkatkan akurasi signifikan (+9.61% untuk Custom CNN)
- Mengurangi overfitting
- Meningkatkan generalisasi

### 2. Untuk Transfer Learning

**Gunakan pretrained weights:**
- ImageNet pretrained untuk CNN
- HuggingFace pretrained untuk ViT
- Fine-tune dengan hati-hati

**Alasan:**
- Memberikan keunggulan signifikan
- Mengurangi kebutuhan data
- Meningkatkan generalisasi

### 3. Untuk Efisiensi Parameter

**Gunakan LoRA:**
- Rank: 4-8 untuk model kecil
- Rank: 8-16 untuk model besar
- Alpha: 32-64

**Alasan:**
- Mengurangi parameter yang di-train
- Mengurangi memory footprint
- Tidak mengorbankan performa signifikan

### 4. Untuk Dataset Terbatas

**Gunakan teknik berikut:**
- Data augmentation
- Transfer learning
- Class weights untuk handle imbalance
- Early stopping untuk prevent overfitting

**Alasan:**
- Dataset terbatas (5,826 citra) memerlukan teknik khusus
- Overfitting adalah masalah utama
- Class imbalance perlu di-handle

## 🎓 Best Practices

### Preprocessing

1. **Selalu gunakan CLAHE** untuk meningkatkan kontras
2. **Lung cropping** membantu fokus pada area relevan
3. **Normalisasi** penting untuk stabilitas numerik

### Training

1. **Gunakan class weights** untuk handle imbalance
2. **Early stopping** untuk prevent overfitting
3. **Model checkpoint** untuk save best model
4. **Monitor validation metrics** secara berkala

### Evaluasi

1. **Gunakan test set** yang tidak pernah dilihat selama training
2. **Evaluasi per kelas** untuk memahami performa detail
3. **Visualisasi confusion matrix** untuk analisis kesalahan
4. **Perhatikan false positives dan negatives**

## ⚠️ Peringatan

1. **Dataset ini untuk penelitian, bukan diagnostik medis langsung**
2. **Perlu validasi klinis** sebelum digunakan di production
3. **Perhatikan bias** dalam dataset
4. **Ethical considerations** penting untuk medical AI

## 📚 Referensi Implementasi

- [Installation Guide](../usage/installation.md)
- [Quick Start Guide](../usage/quickstart.md)
- [Notebooks Guide](../usage/notebooks.md)
- [Methodology](../methodology/preprocessing.md)

[🔮 Lihat future work →](future-work.md)

