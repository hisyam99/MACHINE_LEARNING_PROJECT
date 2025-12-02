# Analisis Hasil

## 🔍 Analisis Mendalam

### 1. Performa Terbaik: HuggingFace ViT Pretrained

**Hasil:**
- **Akurasi:** 91.65% (mengalahkan semua model termasuk SVM)
- **Macro F1:** 0.9017
- **F1 COVID-19:** 0.9601 (sangat tinggi)
- **Keseimbangan kelas baik:** F1 untuk semua kelas di atas 0.87

**Faktor Keberhasilan:**
1. **Pretrained weights** dari ImageNet memberikan representasi yang kuat
2. **Vision Transformer** architecture cocok untuk medical imaging
3. **Fine-tuning** yang tepat pada dataset COVID-19
4. **Data augmentation** meningkatkan generalisasi

**Kesimpulan:** Transfer learning dengan Vision Transformer pretrained memberikan hasil terbaik.

### 2. Keunggulan Machine Learning Klasik (SVM)

**Hasil:**
- **Akurasi:** 86.27% (terbaik kedua setelah HF ViT)
- **Macro F1:** 0.843
- **Stabilitas:** Sangat stabil dan konsisten

**Faktor Keberhasilan:**
1. **HOG features** terbukti robust untuk ukuran dataset ini
2. **Feature selection** dengan ANOVA F-test efektif
3. **RBF kernel** cocok untuk non-linear patterns
4. **Class balancing** membantu handle imbalance

**Kesimpulan:** Fitur terstruktur dari HOG terbukti robust, SVM masih relevan sebagai baseline yang kuat tanpa memerlukan GPU training.

### 3. Dampak Data Augmentation pada Custom CNN

**Perbandingan:**

| Metrik | Tanpa Aug | Dengan Aug | Improvement |
|:-------|:---------:|:----------:|:-----------:|
| **Akurasi** | 71.74% | 81.35% | **+9.61%** |
| **Macro F1** | 0.6586 | 0.7825 | **+0.1239** |
| **F1 COVID-19** | 0.8342 | 0.8901 | +0.0559 |
| **F1 Non-COVID** | 0.4788 | 0.6601 | **+37.8%** |
| **F1 Normal** | 0.6627 | 0.7972 | +0.1345 |

**Analisis:**
- **F1 Non-COVID meningkat drastis:** 0.4788 → 0.6601 (peningkatan 37.8%)
- **Overfitting berkurang:** Training dan validation lebih seimbang
- **Generalisasi meningkat:** Model lebih robust terhadap variasi data

**Kesimpulan:** Data augmentation sangat penting untuk model kecil dari scratch, terutama untuk kelas yang sulit seperti Non-COVID.

### 4. Transfer Learning vs From Scratch

**Perbandingan:**

| Model | Type | Akurasi | Catatan |
|:------|:----|:-------:|:--------|
| **HF ViT Pretrained** | Transfer Learning | 91.65% | Best overall |
| **DenseNet121 + LoRA** | Transfer Learning | 82.04% | Keseimbangan baik |
| **Custom CNN (+Aug)** | From Scratch | 81.35% | Dengan augmentation |
| **ViT (Keras)** | From Scratch | 68.54% | Perlu dataset lebih besar |

**Analisis:**
- **Transfer Learning memberikan keunggulan:** +10.3% untuk HF ViT vs Custom CNN
- **Pretrained weights penting:** ViT from scratch (68.54%) vs HF ViT (91.65%)
- **LoRA membantu efisiensi:** DenseNet121 + LoRA mencapai 82.04% dengan efisiensi parameter

**Kesimpulan:** Transfer learning memberikan keunggulan signifikan, terutama dengan pretrained weights yang baik.

### 5. Analisis Per Kelas

#### F1 COVID-19

| Model | F1 COVID-19 | Ranking |
|:------|:-----------:|:-------:|
| **HF ViT Pretrained** | **0.9601** | 🥇 |
| **Custom CNN (+Aug)** | 0.8901 | 🥈 |
| **DenseNet121 + LoRA** | 0.8743 | 🥉 |

**Temuan:**
- Semua model deep learning mencapai >0.75
- HF ViT mencapai 0.9601 (sangat tinggi)
- Model cenderung lebih baik dalam mendeteksi COVID-19

#### F1 Non-COVID

| Model | F1 Non-COVID | Ranking |
|:------|:------------:|:-------:|
| **HF ViT Pretrained** | **0.8677** | 🥇 |
| **DenseNet121 + LoRA** | 0.7241 | 🥈 |
| **Custom CNN (+Aug)** | 0.6601 | 🥉 |

**Temuan:**
- Kelas paling sulit untuk dibedakan
- Hanya HF ViT dan DenseNet121 yang mencapai >0.70
- Custom CNN struggles tanpa augmentation (0.4788)

**Kesimpulan:** Non-COVID pneumonia adalah kelas paling challenging karena:
- Mirip dengan COVID-19 dalam beberapa aspek
- Variasi yang lebih besar dalam manifestasi
- Perlu fine-grained features untuk membedakan

#### F1 Normal

| Model | F1 Normal | Ranking |
|:------|:---------:|:-------:|
| **HF ViT Pretrained** | **0.8773** | 🥇 |
| **DenseNet121 + LoRA** | 0.8025 | 🥈 |
| **Custom CNN (+Aug)** | 0.7972 | 🥉 |

**Temuan:**
- Performa relatif baik untuk semua model (>0.64)
- HF ViT mencapai 0.8773
- Model dapat membedakan normal dengan baik

### 6. Efisiensi vs Akurasi

**Trade-off Analysis:**

| Model | Akurasi | Model Size | Training Time | Use Case |
|:------|:-------:|:----------:|:-------------:|:---------|
| **HF ViT Pretrained** | 91.65% | ~350 MB | ~2-3 jam | Akurasi maksimal |
| **DenseNet121 + LoRA** | 82.04% | ~30 MB | ~1-2 jam | Keseimbangan |
| **Custom CNN (+Aug)** | 81.35% | ~1.8 MB | ~30-45 min | Mobile/Edge |
| **SVM (RBF)** | 86.27% | ~95 MB | ~10-15 min | Baseline CPU |

**Kesimpulan:** Pilih berdasarkan use case:
- **Akurasi maksimal:** HF ViT
- **Keseimbangan:** DenseNet121 + LoRA
- **Mobile/Edge:** Custom CNN + LoRA
- **Baseline CPU:** SVM

### 7. Kesalahan Umum

#### False Positives

**Pola:**
- Normal sering diprediksi sebagai COVID-19
- Non-COVID sering diprediksi sebagai COVID-19
- Artefak klinis (kabel, selang) disalahartikan sebagai lesi paru

**Penyebab:**
- Model terlalu sensitif terhadap COVID-19
- Struktur mirip antara kelas
- Artefak dalam citra

#### False Negatives

**Pola:**
- COVID-19 jarang terlewat (kecuali untuk model yang sangat buruk)
- Non-COVID lebih sering terlewat
- Normal jarang terlewat

**Penyebab:**
- Non-COVID adalah kelas paling sulit
- Variasi yang lebih besar dalam manifestasi
- Perlu fine-grained features

## 💡 Key Insights

1. **Transfer Learning unggul:** HF ViT (91.65%) mengungguli semua model
2. **Data Augmentation penting:** +9.61% untuk Custom CNN
3. **SVM masih relevan:** 86.27% sebagai baseline kuat
4. **Non-COVID paling sulit:** F1 lebih rendah dibandingkan kelas lain
5. **Trade-off Akurasi vs Efisiensi:** Pilih berdasarkan use case

[📊 Lihat perbandingan lengkap →](comparison.md)

[🖼️ Lihat visualisasi →](visualizations.md)

