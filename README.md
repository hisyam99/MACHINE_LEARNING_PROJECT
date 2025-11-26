# Deteksi COVID-19 pada Citra Chest X-Ray
## Komparasi Machine Learning Klasik dan Deep Learning dengan LoRA

Repository ini berisi dokumentasi teknis, eksperimen, dan kode implementasi untuk proyek pembelajaran mesin (Machine Learning) yang bertujuan mendeteksi COVID-19, Pneumonia Non-COVID, dan kondisi Paru Normal menggunakan citra Chest X-Ray.

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

---

## 🎯 Ringkasan Proyek

Proyek ini melakukan komparasi mendalam antara pendekatan **Machine Learning Klasik** (dengan feature engineering manual) dan **Deep Learning** (CNN, Transfer Learning, dan LoRA) untuk klasifikasi tiga kelas pada citra Chest X-Ray:

1. **COVID-19** - Infeksi COVID-19
2. **Non-COVID** - Pneumonia Viral atau Bacterial (bukan COVID-19)
3. **Normal** - Kondisi paru-paru normal

### Tujuan Penelitian

- Membandingkan efektivitas pendekatan klasik (HOG + SVM) vs Deep Learning (CNN + LoRA)
- Mengevaluasi trade-off antara akurasi dan efisiensi komputasi pada dataset terbatas
- Mengembangkan model lightweight yang cocok untuk deployment di perangkat mobile/edge

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

| Model | Akurasi | Macro F1 | Precision | Recall | Catatan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SVM (RBF)** | **86.27%** | **0.843** | 0.860 | 0.832 | Best overall, stabil |
| **kNN (k=5)** | 77.57% | 0.739 | 0.784 | 0.723 | Rentan high dimensionality |
| **Random Forest** | 76.09% | 0.719 | 0.807 | 0.697 | Struggles dengan non-linear kompleks |
| **Custom CNN + LoRA (No Aug)** | 71.05% | - | - | - | Recall COVID-19: 95% (high sensitivity) |
| **Custom CNN + LoRA (With Aug)** | - | - | - | - | Improved generalization |
| **DenseNet121 + LoRA** | - | - | - | - | Transfer learning baseline |
| **ViT Manual** | - | - | - | - | Vision Transformer from scratch |
| **ViT Pretrained (HF)** | - | - | - | - | HuggingFace pretrained |

> **Catatan:** Hasil lengkap untuk semua model dapat dilihat pada file notebook dan visualisasi di folder `output_images/`.

### Analisis Utama

#### 1. Keunggulan Machine Learning Klasik (SVM)
- Fitur terstruktur dari HOG terbukti lebih robust untuk ukuran dataset ini
- Akurasi tertinggi (86.27%) dengan stabilitas tinggi
- Ukuran model relatif besar (~95 MB) namun performa optimal

#### 2. Sensitivitas vs Spesifisitas (CNN + LoRA)
- **Sensitivitas Tinggi:** Recall COVID-19 mencapai 95%
- **False Positive Tinggi:** Banyak kasus Normal/Non-COVID diprediksi sebagai COVID-19
- **Cocok untuk Screening:** Model sangat waspada, cocok untuk tahap screening awal

#### 3. Efisiensi vs Akurasi
- **SVM:** Akurasi tinggi (86%) namun ukuran besar (95 MB)
- **CNN + LoRA:** Akurasi lebih rendah (71%) namun sangat lightweight (1.8 MB)
- **Trade-off:** Pilih berdasarkan use case (akurasi vs deployment)

#### 4. Kesalahan Umum
- Mayoritas model kesulitan membedakan **Non-COVID vs COVID-19**
- Artefak klinis (kabel, selang) sering disalahartikan sebagai lesi paru
- Fine-grained features sulit ditangkap oleh model kecil

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

1. **Machine Learning Klasik (SVM) tetap unggul** untuk dataset terbatas ini dengan akurasi 86.27%
2. **Custom CNN + LoRA menunjukkan potensi besar** sebagai model lightweight (1.8 MB) dengan sensitivitas tinggi (95% recall COVID-19)
3. **Trade-off Akurasi vs Efisiensi:**
   - SVM: Akurasi tinggi, ukuran besar
   - CNN+LoRA: Akurasi lebih rendah, sangat ringan
4. **False Positive masih menjadi tantangan** untuk model CNN kecil dalam membedakan Non-COVID Pneumonia dari Normal

### Rekomendasi

- **Untuk Akurasi Maksimal:** Gunakan SVM dengan HOG features
- **Untuk Deployment Mobile/Edge:** Gunakan Custom CNN + LoRA dengan optimasi lebih lanjut
- **Untuk Generalisasi Lebih Baik:** Pertimbangkan Transfer Learning dengan data augmentation
- **Future Work:** 
  - Eksperimen dengan dataset lebih besar
  - Fine-tuning hyperparameter LoRA
  - Ensemble methods
  - Attention mechanisms untuk mengurangi false positive

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

**Last Updated:** November 2025

