# Rekomendasi

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

