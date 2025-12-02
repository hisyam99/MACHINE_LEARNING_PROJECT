# Perbandingan Model

## 📊 Tabel Perbandingan Lengkap

### Semua Metrik

| Model | Akurasi | Macro F1 | Weighted F1 | Precision | Recall | F1 COVID-19 | F1 Non-COVID | F1 Normal |
|:------|:-------:|:--------:|:-----------:|:---------:|:------:|:-----------:|:------------:|:---------:|
| **HF ViT Pretrained** | **91.65%** | **0.9017** | **0.9163** | 0.916 | 0.917 | **0.9601** | **0.8677** | **0.8773** |
| **SVM (RBF)** | 86.27% | 0.843 | - | 0.862 | 0.863 | - | - | - |
| **DenseNet121 + LoRA** | 82.04% | 0.8003 | 0.8187 | 0.819 | 0.820 | 0.8743 | 0.7241 | 0.8025 |
| **Custom CNN (+Aug)** | 81.35% | 0.7825 | 0.8093 | 0.809 | 0.813 | 0.8901 | 0.6601 | 0.7972 |
| **Custom CNN (No Aug)** | 71.74% | 0.6586 | 0.7024 | 0.702 | 0.717 | 0.8342 | 0.4788 | 0.6627 |
| **ViT (Keras)** | 68.54% | 0.6645 | 0.6876 | 0.688 | 0.685 | 0.7569 | 0.5921 | 0.6446 |
| **kNN (k=5)** | 77.57% | 0.739 | - | 0.776 | 0.776 | - | - | - |
| **Random Forest** | 76.09% | 0.719 | - | 0.761 | 0.761 | - | - | - |

## 🎯 Kategori Perbandingan

### 1. Akurasi

| Ranking | Model | Akurasi |
|:--------|:------|:-------:|
| 🥇 | **HF ViT Pretrained** | **91.65%** |
| 🥈 | **SVM (RBF)** | 86.27% |
| 🥉 | **DenseNet121 + LoRA** | 82.04% |
| 4 | Custom CNN (+Aug) | 81.35% |
| 5 | kNN (k=5) | 77.57% |
| 6 | Random Forest | 76.09% |
| 7 | Custom CNN (No Aug) | 71.74% |
| 8 | ViT (Keras) | 68.54% |

### 2. Macro F1-Score

| Ranking | Model | Macro F1 |
|:--------|:------|:--------:|
| 🥇 | **HF ViT Pretrained** | **0.9017** |
| 🥈 | **SVM (RBF)** | 0.843 |
| 🥉 | **DenseNet121 + LoRA** | 0.8003 |
| 4 | Custom CNN (+Aug) | 0.7825 |
| 5 | kNN (k=5) | 0.739 |
| 6 | Random Forest | 0.719 |
| 7 | ViT (Keras) | 0.6645 |
| 8 | Custom CNN (No Aug) | 0.6586 |

### 3. F1 COVID-19

| Ranking | Model | F1 COVID-19 |
|:--------|:------|:-----------:|
| 🥇 | **HF ViT Pretrained** | **0.9601** |
| 🥈 | **Custom CNN (+Aug)** | 0.8901 |
| 🥉 | **DenseNet121 + LoRA** | 0.8743 |
| 4 | Custom CNN (No Aug) | 0.8342 |
| 5 | ViT (Keras) | 0.7569 |

### 4. F1 Non-COVID

| Ranking | Model | F1 Non-COVID |
|:--------|:------|:------------:|
| 🥇 | **HF ViT Pretrained** | **0.8677** |
| 🥈 | **DenseNet121 + LoRA** | 0.7241 |
| 🥉 | **Custom CNN (+Aug)** | 0.6601 |
| 4 | ViT (Keras) | 0.5921 |
| 5 | Custom CNN (No Aug) | 0.4788 |

### 5. F1 Normal

| Ranking | Model | F1 Normal |
|:--------|:------|:---------:|
| 🥇 | **HF ViT Pretrained** | **0.8773** |
| 🥈 | **DenseNet121 + LoRA** | 0.8025 |
| 🥉 | **Custom CNN (+Aug)** | 0.7972 |
| 4 | Custom CNN (No Aug) | 0.6627 |
| 5 | ViT (Keras) | 0.6446 |

## 🔍 Analisis Per Kategori

### Classic ML vs Deep Learning

| Kategori | Best Model | Akurasi | Catatan |
|:---------|:-----------|:-------:|:--------|
| **Classic ML** | SVM (RBF) | 86.27% | Terbaik kedua overall |
| **Deep Learning** | HF ViT Pretrained | 91.65% | Terbaik overall |

**Kesimpulan:** Deep Learning dengan transfer learning mengungguli classic ML, namun SVM masih sangat kompetitif.

### From Scratch vs Transfer Learning

| Kategori | Best Model | Akurasi | Catatan |
|:---------|:-----------|:-------:|:--------|
| **From Scratch** | Custom CNN (+Aug) | 81.35% | Dengan augmentation |
| **Transfer Learning** | HF ViT Pretrained | 91.65% | +10.3% improvement |

**Kesimpulan:** Transfer learning memberikan keunggulan signifikan (+10.3%).

### Dengan vs Tanpa Augmentation

| Model | Tanpa Aug | Dengan Aug | Improvement |
|:------|:---------:|:----------:|:-----------:|
| **Custom CNN** | 71.74% | 81.35% | **+9.61%** |

**Kesimpulan:** Data augmentation sangat penting untuk model from scratch.

## 📈 Trade-off Analysis

### Akurasi vs Model Size

| Model | Akurasi | Model Size | Trade-off |
|:------|:-------:|:----------:|:---------:|
| **HF ViT Pretrained** | 91.65% | ~350 MB | Akurasi maksimal |
| **DenseNet121 + LoRA** | 82.04% | ~30 MB | Keseimbangan baik |
| **Custom CNN (+Aug)** | 81.35% | ~1.8 MB | Sangat ringan |
| **SVM (RBF)** | 86.27% | ~95 MB | Baseline kuat |

### Akurasi vs Training Time

| Model | Akurasi | Training Time | Catatan |
|:------|:-------:|:-------------:|:--------|
| **HF ViT Pretrained** | 91.65% | ~2-3 jam | GPU required |
| **DenseNet121 + LoRA** | 82.04% | ~1-2 jam | GPU required |
| **Custom CNN (+Aug)** | 81.35% | ~30-45 min | GPU recommended |
| **SVM (RBF)** | 86.27% | ~10-15 min | CPU sufficient |

## 💡 Rekomendasi Berdasarkan Use Case

### Untuk Akurasi Maksimal
- **Model:** HF ViT Pretrained
- **Akurasi:** 91.65%
- **Trade-off:** Resource intensive

### Untuk Keseimbangan Performa-Efisiensi
- **Model:** DenseNet121 + LoRA
- **Akurasi:** 82.04%
- **Trade-off:** Keseimbangan baik

### Untuk Deployment Mobile/Edge
- **Model:** Custom CNN + LoRA dengan Augmentation
- **Akurasi:** 81.35%
- **Trade-off:** Sangat ringan (~1.8 MB)

### Untuk Baseline Tanpa GPU
- **Model:** SVM dengan HOG features
- **Akurasi:** 86.27%
- **Trade-off:** Ukuran model relatif besar

[📊 Lihat analisis detail →](analysis.md)

