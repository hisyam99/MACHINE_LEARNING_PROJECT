# Deteksi COVID-19 pada Citra Chest X-Ray

<div align="center">

<p style="font-size: 1.1em; font-weight: 600; margin-bottom: 1rem;">
Komparasi Machine Learning Klasik dan Deep Learning dengan LoRA
</p>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python" style="margin: 0 4px;"></a>
<a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.10+-orange.svg" alt="TensorFlow" style="margin: 0 4px;"></a>
<a href="https://creativecommons.org/licenses/by-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-green.svg" alt="License" style="margin: 0 4px;"></a>

</div>

---

## 🎯 Tentang Proyek

Proyek ini melakukan komparasi mendalam antara pendekatan **Machine Learning Klasik** (dengan feature engineering manual) dan **Deep Learning** (CNN, Transfer Learning, dan LoRA) untuk klasifikasi tiga kelas pada citra Chest X-Ray:

1. **COVID-19** - Infeksi COVID-19
2. **Non-COVID** - Pneumonia Viral atau Bacterial (bukan COVID-19)
3. **Normal** - Kondisi paru-paru normal

### ✨ Hasil Utama

| Metrik | Nilai |
|:------|:-----:|
| **Model Terbaik** | HuggingFace ViT Pretrained |
| **Akurasi** | **91.65%** |
| **Macro F1** | **0.9017** |
| **F1 COVID-19** | **0.9601** |

### 🏆 Pencapaian

- ✅ **Transfer Learning unggul:** DenseNet121 + LoRA (82.04%) dan HF ViT (91.65%) mengungguli model from scratch
- ✅ **Data Augmentation penting:** Meningkatkan Custom CNN dari 71.74% menjadi 81.35% akurasi
- ✅ **SVM masih relevan:** 86.27% akurasi sebagai baseline kuat tanpa GPU training
- ✅ **Model lightweight:** Custom CNN + LoRA hanya ~1.8 MB dengan performa 81.35%

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install tensorflow>=2.10
pip install opencv-python
pip install tqdm
pip install jupyter
```

### Menjalankan Eksperimen

1. **Setup Dataset**
   ```bash
   # Download dataset COVID-QU-Ex dari Kaggle
   # Extract ke folder datasets/
   ```

2. **TASK 1: Classic ML**
   ```bash
   jupyter notebook "TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb"
   ```

3. **TASK 2: Custom CNN + LoRA**
   ```bash
   jupyter notebook "TASK_2:CNN+LoRA.ipynb"
   ```

4. **TASK 3: Transfer Learning + ViT**
   ```bash
   jupyter notebook "TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb"
   ```

---

## 📊 Dataset

Proyek ini menggunakan dataset **COVID-QU-Ex** dari Qatar University.

| Statistik | Nilai |
|:----------|:-----:|
| **Total Citra** | 5,826 gambar |
| **COVID-19** | 2,913 (50.0%) |
| **Non-COVID** | 1,457 (25.0%) |
| **Normal** | 1,456 (25.0%) |

**Pembagian Data:**
- Train (70%): 4,078 citra
- Validation (15%): 874 citra
- Test (15%): 874 citra

[📖 Pelajari lebih lanjut tentang dataset →](dataset/introduction.md)

---

## 📈 Hasil Eksperimen

### Perbandingan Performa

| Model | Akurasi | Macro F1 | F1 COVID-19 | F1 Non-COVID | F1 Normal |
|:------|:-------:|:--------:|:-----------:|:------------:|:---------:|
| **HF ViT Pretrained** | **91.65%** | **0.9017** | **0.9601** | **0.8677** | **0.8773** |
| **SVM (RBF)** | 86.27% | 0.843 | - | - | - |
| **DenseNet121 + LoRA** | 82.04% | 0.8003 | 0.8743 | 0.7241 | 0.8025 |
| **Custom CNN (+Aug)** | 81.35% | 0.7825 | 0.8901 | 0.6601 | 0.7972 |
| **Custom CNN (No Aug)** | 71.74% | 0.6586 | 0.8342 | 0.4788 | 0.6627 |

[📊 Lihat analisis lengkap →](results/overview.md)

---

## 🛠️ Teknologi yang Digunakan

<div class="grid cards" markdown>

-   <span style="font-size: 2em;">⚡</span> <strong>TensorFlow/Keras</strong>

    ---

    Deep learning framework untuk implementasi CNN, Transfer Learning, dan Vision Transformer

-   <span style="font-size: 2em;">🧠</span> <strong>Scikit-learn</strong>

    ---

    Machine learning klasik: SVM, Random Forest, KNN dengan HOG feature extraction

-   <span style="font-size: 2em;">🖼️</span> <strong>OpenCV</strong>

    ---

    Preprocessing citra: CLAHE, lung cropping, dan normalisasi

-   <span style="font-size: 2em;">🤗</span> <strong>HuggingFace</strong>

    ---

    Pretrained Vision Transformer untuk transfer learning

</div>

---

## 📚 Dokumentasi

Dokumentasi lengkap tersedia di situs ini, mencakup:

- 📖 [Tentang Proyek](about/overview.md) - Ringkasan dan tujuan penelitian
- 📊 [Dataset](dataset/introduction.md) - Informasi lengkap tentang COVID-QU-Ex
- 🔬 [Metodologi](methodology/preprocessing.md) - Preprocessing, model, dan teknik yang digunakan
- 🧪 [Eksperimen](experiments/task1-classic-ml.md) - Detail implementasi setiap task
- 📈 [Hasil](results/overview.md) - Analisis performa dan visualisasi
- 💡 [Kesimpulan](conclusions/summary.md) - Temuan dan rekomendasi

---

## 👥 Tim

| NIM | Nama |
|:---|:---|
| **202210370311060** | **Muhammad Hisyam Kamil** |
| **202210370311449** | **Elga Putri Tri Farma** |

**Institusi:** Universitas Muhammadiyah Malang (UMM)  
**Mata Kuliah:** Machine Learning  
**Semester:** 7

[👥 Pelajari lebih lanjut tentang tim →](about/team.md)

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik. Dataset COVID-QU-Ex memiliki lisensi **CC BY-SA 4.0**.

---

## 📧 Kontak

- **Muhammad Hisyam Kamil:** hisyamkamil99@webmail.umm.ac.id
- **Elga Putri Tri Farma:** elgafarma@webmail.umm.ac.id

---

<div align="center">

</div>

