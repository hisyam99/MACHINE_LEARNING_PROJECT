# Kesimpulan - Ringkasan

## 🎯 Temuan Utama

### 1. Model Terbaik: HuggingFace ViT Pretrained

**HuggingFace ViT Pretrained** (`google/vit-base-patch16-224-in21k`) adalah model terbaik dengan:
- **Akurasi:** **91.65%** (🏆 mengungguli semua model termasuk SVM klasik)
- **Macro F1:** **0.9017** (🏆 keseimbangan terbaik)
- **F1 COVID-19:** **0.9601** (sangat tinggi - hanya 4% yang terlewat)
- **F1 Non-COVID:** **0.8677** (🏆 akhirnya kelas tersulit teratasi)
- **F1 Normal:** **0.8773** (keseimbangan sempurna)
- **Keseimbangan kelas sangat baik:** F1 untuk semua kelas di atas 0.87

**Faktor Keberhasilan:**

1. **Pre-training Skala Besar:**
   - Dilatih pada ImageNet-21k (14 juta images, 21,000 classes)
   - Representasi visual yang sangat general dan transferable
   - Fitur low-level hingga high-level sudah dipelajari

2. **Vision Transformer Architecture:**
   - Self-attention mechanism dapat menangkap hubungan long-range
   - Global context dari seluruh image sekaligus
   - Adaptive attention weights fokus pada area penting

3. **Fine-tuning yang Tepat:**
   - Learning rate sangat kecil (2e-5) mencegah catastrophic forgetting
   - Gradual unfreezing memberikan stabilitas
   - Class weights untuk handle imbalance

**Kesimpulan:** Transfer learning dengan Vision Transformer pretrained pada dataset skala besar (ImageNet-21k) memberikan hasil terbaik, menunjukkan bahwa representasi yang dipelajari dari natural images dapat di-transfer secara efektif ke domain medical imaging melalui fine-tuning yang hati-hati.

### 2. Transfer Learning Memberikan Keunggulan Signifikan

**Perbandingan:**
- **HF ViT Pretrained:** 91.65% akurasi
- **DenseNet121 + LoRA:** 82.04% akurasi
- **Custom CNN + Aug:** 81.35% akurasi (from scratch)
- **ViT Keras (from scratch):** 68.54% akurasi

**Kesimpulan:** Transfer learning memberikan keunggulan signifikan, terutama dengan pretrained weights yang baik.

### 3. Data Augmentation Sangat Penting

**Dampak pada Custom CNN:**
- **Tanpa Augmentation:** 71.74% akurasi, Macro F1 0.6586
- **Dengan Augmentation:** 81.35% akurasi, Macro F1 0.7825
- **Peningkatan:** +9.61% akurasi, +0.1239 Macro F1
- **F1 Non-COVID meningkat drastis:** 0.4788 → 0.6601 (peningkatan 37.8%)

**Kesimpulan:** Data augmentation sangat penting untuk model from scratch, terutama untuk kelas yang sulit seperti Non-COVID.

### 4. Machine Learning Klasik (SVM) Masih Relevan

**SVM dengan HOG features:**
- **Akurasi:** 86.27% (terbaik kedua setelah HF ViT)
- **Macro F1:** 0.843
- **Stabilitas:** Sangat stabil dan konsisten
- **Tidak perlu GPU:** Dapat dijalankan pada CPU

**Kesimpulan:** Machine Learning Klasik (SVM) masih relevan sebagai baseline yang kuat tanpa memerlukan GPU training.

### 5. Non-COVID Pneumonia adalah Kelas Paling Challenging

**Analisis per kelas:**
- **F1 COVID-19:** Semua model deep learning mencapai >0.75, dengan HF ViT mencapai 0.9601
- **F1 Non-COVID:** Kelas paling sulit, hanya HF ViT dan DenseNet121 yang mencapai >0.70
- **F1 Normal:** Performa relatif baik untuk semua model (>0.64)

**Kesimpulan:** Non-COVID pneumonia adalah kelas paling challenging untuk dibedakan karena:
- Mirip dengan COVID-19 dalam beberapa aspek
- Variasi yang lebih besar dalam manifestasi
- Perlu fine-grained features untuk membedakan

### 6. Trade-off Akurasi vs Efisiensi

**Perbandingan:**
- **HF ViT Pretrained:** Akurasi tertinggi (91.65%) namun memerlukan resources lebih besar
- **DenseNet121 + LoRA:** Keseimbangan baik (82.04%) dengan efisiensi parameter melalui LoRA
- **Custom CNN + LoRA:** Sangat ringan (~1.8 MB) dengan performa 81.35% (dengan augmentation)
- **SVM:** Baseline kuat (86.27%) dengan ukuran model relatif besar (~95 MB)

**Kesimpulan:** Pilih berdasarkan use case (akurasi maksimal vs deployment mobile/edge).

## 📊 Ringkasan Performa

| Model | Akurasi | Macro F1 | Ranking |
|:------|:-------:|:--------:|:-------:|
| **HF ViT Pretrained** | **91.65%** | **0.9017** | 🥇 |
| **SVM (RBF)** | 86.27% | 0.843 | 🥈 |
| **DenseNet121 + LoRA** | 82.04% | 0.8003 | 🥉 |
| **Custom CNN (+Aug)** | 81.35% | 0.7825 | 4 |
| **Custom CNN (No Aug)** | 71.74% | 0.6586 | 7 |

## 💡 Key Insights

1. **Transfer Learning unggul** pada dataset terbatas
2. **Data Augmentation penting** untuk model from scratch
3. **SVM masih relevan** sebagai baseline kuat
4. **Non-COVID paling sulit** untuk dibedakan
5. **Trade-off Akurasi vs Efisiensi** perlu dipertimbangkan

## 🎓 Pelajaran yang Dipetik

1. **Pretrained models** memberikan keunggulan signifikan pada dataset terbatas
2. **Data augmentation** sangat penting untuk model from scratch
3. **Classic ML** masih relevan dan kompetitif
4. **LoRA** membantu efisiensi parameter tanpa mengorbankan performa
5. **Fine-grained features** penting untuk membedakan kelas yang mirip

[📋 Lihat rekomendasi →](recommendations.md)

[🔮 Lihat future work →](future-work.md)

