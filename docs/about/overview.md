# Ringkasan Proyek

## 🎯 Tujuan Penelitian

Proyek ini bertujuan untuk:

1. **Membandingkan efektivitas** pendekatan klasik (HOG + SVM) vs Deep Learning (CNN, Transfer Learning, ViT + LoRA)
2. **Mengevaluasi trade-off** antara akurasi dan efisiensi komputasi pada dataset terbatas
3. **Mengembangkan model lightweight** yang cocok untuk deployment di perangkat mobile/edge
4. **Mengevaluasi dampak** data augmentation dan transfer learning pada performa model

## 📋 Klasifikasi Tiga Kelas

Proyek ini melakukan klasifikasi citra Chest X-Ray menjadi tiga kelas:

1. **COVID-19** - Infeksi COVID-19
2. **Non-COVID** - Pneumonia Viral atau Bacterial (bukan COVID-19)
3. **Normal** - Kondisi paru-paru normal

## 🔬 Pendekatan yang Dibandingkan

### 1. Machine Learning Klasik

- **Feature Engineering Manual:**
  - HOG (Histogram of Oriented Gradients)
  - Feature Selection dengan ANOVA F-test
  - StandardScaler untuk normalisasi
  
- **Model yang Diuji:**
  - SVM dengan RBF Kernel
  - Random Forest
  - k-Nearest Neighbors (kNN)

### 2. Deep Learning

- **Custom CNN dari Scratch:**
  - Arsitektur 4 blok konvolusi
  - LoRA (Low-Rank Adaptation) untuk efisiensi parameter
  - Dengan dan tanpa data augmentation

- **Transfer Learning:**
  - DenseNet121 pretrained + LoRA
  - Vision Transformer (ViT) dari HuggingFace
  - Vision Transformer implementasi manual

## 🎯 Hasil Utama

### Model Terbaik

**HuggingFace ViT Pretrained** mencapai:
- **Akurasi:** 91.65%
- **Macro F1:** 0.9017
- **F1 COVID-19:** 0.9601

### Temuan Penting

1. **Transfer Learning unggul:** DenseNet121 + LoRA (82.04%) dan HF ViT (91.65%) mengungguli model from scratch
2. **Data Augmentation penting:** Meningkatkan Custom CNN dari 71.74% menjadi 81.35% akurasi
3. **SVM masih relevan:** 86.27% akurasi sebagai baseline kuat tanpa GPU training
4. **Model lightweight:** Custom CNN + LoRA hanya ~1.8 MB dengan performa 81.35%

## 📊 Dataset

Proyek ini menggunakan dataset **COVID-QU-Ex** dari Qatar University:

- **Total Citra:** 5,826 gambar
- **Distribusi Kelas:**
  - COVID-19: 2,913 (50.0%)
  - Non-COVID: 1,457 (25.0%)
  - Normal: 1,456 (25.0%)
- **Pembagian Data:** 70% Train, 15% Validation, 15% Test

[📖 Pelajari lebih lanjut tentang dataset →](../dataset/introduction.md)

## 🛠️ Teknologi

- **Deep Learning:** TensorFlow/Keras
- **Classic ML:** Scikit-learn
- **Image Processing:** OpenCV
- **Transfer Learning:** HuggingFace Transformers
- **Efficiency:** LoRA (Low-Rank Adaptation)

## 📈 Struktur Eksperimen

Proyek ini dibagi menjadi tiga task utama:

1. **TASK 1:** Preprocessing + Feature Extraction + Classic ML Models
2. **TASK 2:** Custom CNN Architecture + LoRA Implementation
3. **TASK 3:** Pretrained DenseNet + Vision Transformer + Augmentation + LoRA

[🧪 Pelajari lebih lanjut tentang eksperimen →](../experiments/task1-classic-ml.md)

