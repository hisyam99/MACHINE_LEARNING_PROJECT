# Deteksi COVID-19 pada Citra Chest X-Ray
## Komparasi Machine Learning Klasik dan Deep Learning dengan LoRA

Repository ini berisi dokumentasi teknis, eksperimen, dan kode implementasi untuk proyek pembelajaran mesin (Machine Learning) yang bertujuan mendeteksi COVID-19, Pneumonia Non-COVID, dan kondisi Paru Normal menggunakan citra Chest X-Ray.

> 📖 **Dokumentasi Lengkap:** [Baca dokumentasi online](https://hisyam99.github.io/MACHINE_LEARNING/) | [Build lokal](#-membangun-dokumentasi-lokal)

---

## 👥 Anggota Kelompok

| NIM | Nama |
| :--- | :--- |
| **202210370311060** | **Muhammad Hisyam Kamil** |
| **202210370311449** | **Elga Putri Tri Farma** |

**Institusi:** Universitas Muhammadiyah Malang (UMM)  
**Mata Kuliah:** Machine Learning  
**Semester:** 7

---

## 📋 Daftar Isi

- [Ringkasan Proyek](#-ringkasan-proyek)
- [Dataset](#-dataset)
- [Struktur Proyek](#-struktur-proyek)
- [Metodologi](#-metodologi)
- [Hasil Eksperimen](#-hasil-eksperimen)
- [Cara Menggunakan](#-cara-menggunakan)
- [Output & Visualisasi](#-output--visualisasi)
- [Kesimpulan](#-kesimpulan)
- [Referensi](#-referensi)
- [Dokumentasi](#-dokumentasi)

---

## 🎯 Ringkasan Proyek

Proyek ini melakukan komparasi mendalam antara pendekatan **Machine Learning Klasik** (dengan feature engineering manual) dan **Deep Learning** (CNN, Transfer Learning, dan LoRA) untuk klasifikasi tiga kelas pada citra Chest X-Ray:

1. **COVID-19** - Infeksi COVID-19
2. **Non-COVID** - Pneumonia Viral atau Bacterial (bukan COVID-19)
3. **Normal** - Kondisi paru-paru normal

### Tujuan Penelitian

- Membandingkan efektivitas pendekatan klasik (HOG + SVM) vs Deep Learning (CNN, Transfer Learning, ViT + LoRA)
- Mengevaluasi trade-off antara akurasi dan efisiensi komputasi pada dataset terbatas
- Mengembangkan model lightweight yang cocok untuk deployment di perangkat mobile/edge
- Mengevaluasi dampak data augmentation dan transfer learning pada performa model

### Hasil Utama

- **Model Terbaik:** HuggingFace ViT Pretrained mencapai **91.65% akurasi** dan **0.9017 Macro F1**
- **Transfer Learning unggul:** DenseNet121 + LoRA (82.04%) dan HF ViT (91.65%) mengungguli model from scratch
- **Data Augmentation penting:** Meningkatkan Custom CNN dari 71.74% menjadi 81.35% akurasi
- **SVM masih relevan:** 86.27% akurasi sebagai baseline kuat tanpa GPU training

---

## 📊 Dataset

Proyek ini menggunakan dataset **COVID-QU-Ex** dari Qatar University. Untuk informasi lengkap tentang dataset, lihat [README.md di folder datasets](./datasets/README.md).

### Ringkasan Dataset yang Digunakan

- **Total Citra:** 5,826 gambar
- **Distribusi Kelas:**
  - COVID-19: 2,913 (50.0%)
  - Non-COVID: 1,457 (25.0%)
  - Normal: 1,456 (25.0%)
- **Pembagian Data (Stratified Split):**
  - Train (70%): 4,078 citra
  - Validation (15%): 874 citra
  - Test (15%): 874 citra

---

## 📁 Struktur Proyek

```
MACHINE_LEARNING/
│
├── datasets/                          # Folder dataset (lihat README.md di dalamnya)
│   └── README.md                      # Dokumentasi dataset COVID-QU-Ex
│
├── output_images/                     # Hasil visualisasi dan grafik
│   ├── acc_loss_*.png                 # Grafik akurasi & loss per model
│   ├── conf_matrix_*.png              # Confusion matrix per model
│   ├── *_5_predict_true_false.png     # Visualisasi prediksi benar/salah
│   ├── benchmark_*.png                # Grafik perbandingan antar model
│   └── per_class_*.png                # Analisis per kelas
│
├── TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb
│   └── Preprocessing, HOG Feature Extraction, Classic ML Models (SVM, RF, KNN)
│
├── TASK_2:CNN+LoRA.ipynb
│   └── Custom CNN Architecture + LoRA Implementation
│
├── TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb
│   └── Transfer Learning (DenseNet121), Vision Transformer, Data Augmentation
│
├── POSTER_MACHINE_LEARNING.png        # Poster penelitian
├── Laporan_Machine_Learning_v3.pdf    # Laporan lengkap penelitian
│
└── README.md                          # File ini
```

---

## 🛠️ Metodologi

### 1. Preprocessing Data

Pipeline preprocessing yang diterapkan pada semua citra:

1. **Grayscale Conversion** - Konversi citra ke skala abu-abu
2. **Resize** - Standarisasi ukuran menjadi `224×224` piksel
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Clip Limit: 2.0
   - Tile Grid Size: 8×8
   - Meningkatkan kontras lokal untuk menonjolkan struktur paru
4. **Heuristic Lung Cropping** - Otomatis crop area paru-paru menggunakan thresholding & connected components
5. **Normalisasi** - Skala piksel ke rentang `[0, 1]`

### 2. TASK 1: Machine Learning Klasik

#### Feature Engineering
1. **HOG (Histogram of Oriented Gradients)**
   - Menangkap tekstur dan pola tepi
   - Menghasilkan vektor fitur 6,084 dimensi
2. **Feature Selection (SelectKBest)**
   - Metode: ANOVA F-test
   - Memilih 4,096 fitur terbaik
3. **Scaling (StandardScaler)**
   - Standarisasi fitur untuk stabilitas distribusi

#### Model yang Diuji
- **SVM (RBF Kernel)**
  - C=10, Class Weight: Balanced
  - Best performer dengan akurasi 86.27%
- **Random Forest**
  - 300 trees
  - Akurasi: 76.09%
- **k-Nearest Neighbors (kNN)**
  - k=5
  - Akurasi: 77.57%

### 3. TASK 2: Custom CNN + LoRA

#### Arsitektur
- **4 Blok Konvolusi:**
  - Conv2D → BatchNorm → ReLU → MaxPool
  - Filter: 32, 64, 128, 256
- **LoRA (Low-Rank Adaptation)**
  - Mengganti Dense Layer standar dengan `LoRADense`
  - Efisiensi parameter tinggi
- **Spesifikasi Model:**
  - Parameter: ~405,731
  - Ukuran: ~1.8 MB (sangat lightweight)
  - Input: 224×224×1 (grayscale)

#### Training Configuration
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy dengan Class Weights
- **Callbacks:** Early Stopping, Model Checkpoint
- **Augmentation:** Tidak digunakan (baseline)

### 4. TASK 3: Transfer Learning + Vision Transformer

#### Model yang Diuji

**A. DenseNet121 + LoRA**
- Pretrained pada ImageNet
- Fine-tuning dengan LoRA pada layer akhir
- Data augmentation diterapkan

**B. Vision Transformer (ViT)**
- **Manual Implementation:** ViT dari scratch
- **HuggingFace Pretrained:** ViT-Base/16 pretrained
- Patch size: 16×16
- Embedding dimension: 768

#### Data Augmentation
- Random Rotation (±15°)
- Random Zoom (0.9-1.1)
- Random Brightness/Contrast
- Horizontal Flip (50% probability)

---

## 📈 Hasil Eksperimen

### Perbandingan Performa pada Test Set

#### Machine Learning Klasik

| Model | Akurasi | Macro F1 | Catatan |
| :--- | :---: | :---: | :--- |
| **SVM (RBF)** | **86.27%** | **0.843** | Best overall, stabil, presisi tinggi |
| **kNN (k=5)** | 77.57% | 0.739 | Rentan high dimensionality |
| **Random Forest** | 76.09% | 0.719 | Struggles dengan non-linear kompleks |

#### Deep Learning Models

| Model | Akurasi | Macro F1 | Weighted F1 | F1 COVID-19 | F1 Non-COVID | F1 Normal |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HF ViT Pretrained** | **91.65%** | **0.9017** | **0.9163** | **0.9601** | **0.8677** | **0.8773** |
| **DenseNet121 + LoRA** | 82.04% | 0.8003 | 0.8187 | 0.8743 | 0.7241 | 0.8025 |
| **Custom CNN (+Aug, LoRA Head)** | 81.35% | 0.7825 | 0.8093 | 0.8901 | 0.6601 | 0.7972 |
| **Custom CNN (No Aug, LoRA Head)** | 71.74% | 0.6586 | 0.7024 | 0.8342 | 0.4788 | 0.6627 |
| **ViT (Keras)** | 68.54% | 0.6645 | 0.6876 | 0.7569 | 0.5921 | 0.6446 |

> **Catatan:** Hasil lengkap untuk semua model dapat dilihat pada file notebook dan visualisasi di folder `output_images/`.

### Analisis Utama

#### 1. Performa Terbaik: HuggingFace ViT Pretrained
- **Akurasi tertinggi:** 91.65% (mengalahkan semua model termasuk SVM)
- **Macro F1 tertinggi:** 0.9017
- **F1 COVID-19 sangat tinggi:** 0.9601 (deteksi COVID-19 sangat akurat)
- **Keseimbangan kelas baik:** F1 untuk semua kelas di atas 0.87
- **Kesimpulan:** Transfer learning dengan Vision Transformer pretrained memberikan hasil terbaik

#### 2. Keunggulan Machine Learning Klasik (SVM)
- Fitur terstruktur dari HOG terbukti robust untuk ukuran dataset ini
- Akurasi 86.27% dengan stabilitas tinggi (terbaik kedua setelah HF ViT)
- Ukuran model relatif besar (~95 MB) namun performa optimal
- **Masih relevan** sebagai baseline yang kuat tanpa memerlukan GPU training

#### 3. Dampak Data Augmentation pada Custom CNN
- **Tanpa Augmentation:** Akurasi 71.74%, Macro F1 0.6586
- **Dengan Augmentation:** Akurasi 81.35%, Macro F1 0.7825
- **Peningkatan signifikan:** +9.61% akurasi, +0.1239 Macro F1
- **F1 Non-COVID meningkat drastis:** 0.4788 → 0.6601 (peningkatan 37.8%)
- **Kesimpulan:** Data augmentation sangat penting untuk model kecil dari scratch

#### 4. Transfer Learning vs From Scratch
- **DenseNet121 + LoRA:** 82.04% akurasi (transfer learning)
- **Custom CNN + Aug:** 81.35% akurasi (from scratch)
- **ViT Keras (from scratch):** 68.54% akurasi (terendah)
- **Kesimpulan:** Transfer learning memberikan keunggulan signifikan, terutama dengan pretrained weights yang baik

#### 5. Analisis Per Kelas
- **F1 COVID-19:** Semua model deep learning mencapai >0.75, dengan HF ViT mencapai 0.9601
- **F1 Non-COVID:** Kelas paling sulit, hanya HF ViT dan DenseNet121 yang mencapai >0.70
- **F1 Normal:** Performa relatif baik untuk semua model (>0.64)
- **Kesimpulan:** Non-COVID pneumonia adalah kelas paling challenging untuk dibedakan

#### 6. Efisiensi vs Akurasi
- **HF ViT Pretrained:** Akurasi tertinggi (91.65%) namun memerlukan resources lebih besar
- **DenseNet121 + LoRA:** Keseimbangan baik (82.04%) dengan efisiensi parameter melalui LoRA
- **Custom CNN + LoRA:** Sangat lightweight (~1.8 MB) dengan performa 81.35% (dengan augmentation)
- **Trade-off:** Pilih berdasarkan use case (akurasi maksimal vs deployment mobile/edge)

#### 7. Kesalahan Umum
- Mayoritas model kesulitan membedakan **Non-COVID vs COVID-19** (terlihat dari F1 Non-COVID yang lebih rendah)
- Artefak klinis (kabel, selang) sering disalahartikan sebagai lesi paru
- Model from scratch (tanpa transfer learning) memerlukan data augmentation untuk performa optimal

---

## 🚀 Cara Menggunakan

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
   - Download dataset COVID-QU-Ex dari Kaggle
   - Extract ke folder `datasets/`
   - Struktur folder harus sesuai dengan yang diharapkan notebook

2. **TASK 1: Classic ML**
   ```bash
   jupyter notebook "TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb"
   ```
   - Menjalankan preprocessing, HOG extraction, dan training model klasik

3. **TASK 2: Custom CNN + LoRA**
   ```bash
   jupyter notebook "TASK_2:CNN+LoRA.ipynb"
   ```
   - Training Custom CNN dengan LoRA

4. **TASK 3: Transfer Learning + ViT**
   ```bash
   jupyter notebook "TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb"
   ```
   - Training dengan pretrained models dan Vision Transformer

### Output

- Model weights disimpan di folder `artifacts/`
- Visualisasi disimpan di folder `output_images/`
- Metrics dan classification reports tersedia di notebook

---

## 📊 Output & Visualisasi

Folder `output_images/` berisi berbagai visualisasi hasil eksperimen:

### Grafik Training
- `acc_loss_*.png` - Grafik akurasi dan loss selama training
- `benchmark_*.png` - Perbandingan performa antar model

### Evaluasi Model
- `conf_matrix_*.png` - Confusion matrix untuk setiap model
- `per_class_*.png` - Analisis detail per kelas (F1, Error Rate, Recall)

### Visualisasi Prediksi
- `*_5_predict_true_false.png` - Contoh prediksi benar dan salah dengan confidence score

### Model yang Divisualisasikan
1. Custom CNN + LoRA (No Augmentation)
2. Custom CNN + LoRA (With Augmentation)
3. DenseNet121 + LoRA
4. Vision Transformer (Manual)
5. Vision Transformer (HuggingFace Pretrained)

---

## 💡 Kesimpulan

### Temuan Utama

1. **HuggingFace ViT Pretrained adalah model terbaik** dengan akurasi 91.65% dan Macro F1 0.9017, mengungguli semua model termasuk SVM klasik
2. **Transfer Learning memberikan keunggulan signifikan:**
   - HF ViT Pretrained: 91.65% akurasi
   - DenseNet121 + LoRA: 82.04% akurasi
   - Keduanya mengungguli model from scratch
3. **Data Augmentation sangat penting** untuk model from scratch:
   - Custom CNN tanpa augmentation: 71.74% akurasi
   - Custom CNN dengan augmentation: 81.35% akurasi (+9.61%)
4. **Machine Learning Klasik (SVM) masih relevan** sebagai baseline yang kuat dengan akurasi 86.27% tanpa memerlukan GPU training
5. **Non-COVID Pneumonia adalah kelas paling challenging:**
   - F1 Non-COVID lebih rendah dibandingkan kelas lain di sebagian besar model
   - Hanya HF ViT dan DenseNet121 yang mencapai F1 >0.70 untuk kelas ini
6. **Trade-off Akurasi vs Efisiensi:**
   - HF ViT: Akurasi tertinggi (91.65%) namun memerlukan resources lebih besar
   - DenseNet121 + LoRA: Keseimbangan baik (82.04%) dengan efisiensi parameter
   - Custom CNN + LoRA: Sangat ringan (~1.8 MB) dengan performa 81.35% (dengan augmentation)

### Rekomendasi

- **Untuk Akurasi Maksimal:** Gunakan **HuggingFace ViT Pretrained** (91.65% akurasi)
- **Untuk Keseimbangan Performa-Efisiensi:** Gunakan **DenseNet121 + LoRA** (82.04% akurasi)
- **Untuk Deployment Mobile/Edge:** Gunakan **Custom CNN + LoRA dengan Augmentation** (81.35% akurasi, ~1.8 MB)
- **Untuk Baseline Tanpa GPU:** Gunakan **SVM dengan HOG features** (86.27% akurasi)
- **Untuk Model From Scratch:** **Selalu gunakan data augmentation** untuk performa optimal

### Future Work

- Eksperimen dengan dataset lebih besar untuk validasi generalisasi
- Fine-tuning hyperparameter LoRA untuk optimasi lebih lanjut
- Ensemble methods menggabungkan multiple models
- Attention mechanisms untuk mengurangi false positive pada kelas Non-COVID
- Eksperimen dengan arsitektur transformer lainnya (Swin Transformer, ConvNeXt)
- Optimasi model untuk deployment edge devices (quantization, pruning)

---

## 📚 Referensi

### Dataset
- COVID-QU-Ex Dataset: [Kaggle](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
- Lihat [datasets/README.md](./datasets/README.md) untuk informasi lengkap dan citation

### Papers & Methods
1. Tahir, A. M., et al. "COVID-19 Infection Localization and Severity Grading from Chest X-ray Images", Computers in Biology and Medicine, vol. 139, p. 105002, 2021.
2. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
3. Vision Transformer (Dosovitskiy et al., 2020)
4. DenseNet: Densely Connected Convolutional Networks (Huang et al., 2017)

### Tools & Libraries
- TensorFlow/Keras
- Scikit-learn
- OpenCV
- HuggingFace Transformers

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik. Dataset COVID-QU-Ex memiliki lisensi CC BY-SA 4.0.

---

## 📧 Kontak

- **Muhammad Hisyam Kamil:** hisyamkamil99@webmail.umm.ac.id
- **Elga Putri Tri Farma:** elgafarma@webmail.umm.ac.id

---

## 📚 Dokumentasi

Dokumentasi lengkap proyek ini tersedia dalam format **Material for MkDocs** yang dapat diakses secara online atau dibangun secara lokal.

### 🌐 Dokumentasi Online

Dokumentasi tersedia di: **https://hisyam99.github.io/MACHINE_LEARNING_PROJECT/**

### 🏗️ Membangun Dokumentasi Lokal

#### Prerequisites

```bash
pip install -r requirements-docs.txt
```

#### Build & Serve

```bash
# Build dokumentasi
mkdocs build

# Serve lokal (dengan auto-reload)
mkdocs serve

# Buka browser di http://127.0.0.1:8000
```

#### Deploy ke GitHub Pages

Dokumentasi akan otomatis di-deploy ke GitHub Pages ketika:
- Push ke branch `main` atau `master`
- File di folder `docs/` atau `mkdocs.yml` berubah
- GitHub Actions workflow berhasil dijalankan

Untuk deploy manual:
```bash
mkdocs gh-deploy
```

### 📁 Struktur Dokumentasi

```
docs/
├── index.md                    # Halaman utama
├── about/                      # Tentang proyek
│   ├── overview.md
│   ├── team.md
│   └── objectives.md
├── dataset/                    # Dokumentasi dataset
│   ├── introduction.md
│   ├── statistics.md
│   ├── preprocessing.md
│   └── citation.md
├── methodology/               # Metodologi
│   ├── preprocessing.md
│   ├── classic-ml.md
│   ├── custom-cnn.md
│   ├── transfer-learning.md
│   └── lora.md
├── experiments/               # Detail eksperimen
│   ├── task1-classic-ml.md
│   ├── task2-cnn-lora.md
│   └── task3-pretrained.md
├── results/                    # Hasil eksperimen
│   ├── overview.md
│   ├── comparison.md
│   ├── analysis.md
│   └── visualizations.md
├── usage/                      # Panduan penggunaan
│   ├── installation.md
│   ├── quickstart.md
│   └── notebooks.md
├── conclusions/                # Kesimpulan
│   ├── summary.md
│   ├── recommendations.md
│   └── future-work.md
└── references.md              # Referensi
```

### 🎨 Fitur Dokumentasi

- ✅ **Material Design** - UI modern dan responsif
- ✅ **Dark Mode** - Tema gelap untuk kenyamanan mata
- ✅ **Search** - Pencarian cepat di seluruh dokumentasi
- ✅ **Navigation** - Navigasi yang mudah dan intuitif
- ✅ **Code Highlighting** - Syntax highlighting untuk code blocks
- ✅ **Responsive** - Mobile-friendly
- ✅ **Auto-deploy** - Otomatis deploy ke GitHub Pages via GitHub Actions

---

**Last Updated:** November 2025

