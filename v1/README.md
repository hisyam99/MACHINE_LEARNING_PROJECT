# Deteksi COVID-19 pada Citra Chest X-Ray
### Studi Komparasi Pendekatan Machine Learning Klasik vs Deep Learning (Custom CNN + LoRA)

Repository ini berisi dokumentasi teknis, eksperimen, dan kode implementasi untuk proyek pembelajaran mesin (Machine Learning) yang bertujuan mendeteksi COVID-19, Pneumonia Non-COVID, dan kondisi Paru Normal menggunakan citra Chest X-Ray.

## 👥 Anggota Kelompok
| NIM | Nama |
| :--- | :--- |
| **202210370311060** | **Muhammad Hisyam Kamil** |
| **202210370311449** | **Elga Putri Tri Farma** |

---

## 📊 Ringkasan Dataset
Proyek ini menggunakan dataset **COVID-QU-Ex**.
- **Total Citra:** 5.826 gambar.
- **Distribusi Kelas:**
  - COVID-19: 2.913 (50.0%)
  - Non-COVID: 1.457 (25.0%)
  - Normal: 1.456 (25.0%)
- **Pembagian Data (Stratified Split):**
  - Train (70%): 4.078 citra
  - Validation (15%): 874 citra
  - Test (15%): 874 citra

---

## 🛠️ Metodologi Pipeline

### 1. Preprocessing Data
Langkah awal untuk menyeragamkan kualitas citra sebelum masuk ke model:
- **Grayscale & Resize:** Konversi ke hitam-putih dan ubah ukuran menjadi `224x224` piksel.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Meningkatkan kontras (clip limit 2.0, tile 8x8) untuk menonjolkan struktur paru seperti garis interstisial.
- **Normalisasi:** Skala piksel diubah ke rentang `[0, 1]`.

### 2. Feature Engineering (Pendekatan Klasik)
Kami menerapkan ekstraksi fitur manual sebelum klasifikasi:
1.  **HOG (Histogram of Oriented Gradients):** Menangkap tekstur dan pola tepi (vektor 6.084 fitur).
2.  **Feature Selection (SelectKBest):** Memilih 4.096 fitur terbaik berdasarkan skor ANOVA.
3.  **Scaling (StandardScaler):** Menstandarisasi fitur agar distribusi data stabil.

### 3. Model Machine Learning Klasik
Tiga algoritma diuji menggunakan fitur hasil HOG:
- **SVM (RBF Kernel):** *Best Performer*. (C=10, Class Weight Balanced).
- **Random Forest:** (300 Trees).
- **k-Nearest Neighbors (kNN):** (k=5).

### 4. Deep Learning (Custom CNN + LoRA)
Eksperimen arsitektur ringan (*lightweight*) dibangun dari nol (*from scratch*):
- **Arsitektur:** 4 Blok Konvolusi (Conv2D -> BatchNorm -> ReLU -> MaxPool).
- **LoRA (Low-Rank Adaptation):** Mengganti Dense Layer standar dengan `LoRADense` untuk efisiensi parameter.
- **Ukuran Model:** Sangat kecil (~1.8 MB) dengan hanya ~405k parameter.
- **Optimizer:** Adam dengan Class Weights untuk menangani ketidakseimbangan data.

---

## 📈 Hasil Eksperimen & Benchmark

Perbandingan performa pada **Test Set**:

| Model | Akurasi | F1-Score (Macro) | Catatan |
| :--- | :---: | :---: | :--- |
| **SVM (RBF)** | **86.27%** | **0.843** | Performa terbaik, stabil, dan presisi tinggi. |
| **kNN (k=5)** | 77.57% | 0.739 | Rentan terhadap kepadatan fitur (high dimensionality). |
| **Random Forest** | 76.09% | 0.719 | Kesulitan menangkap hubungan non-linier kompleks. |
| **Custom CNN + LoRA** | 71.05% | - | Recall COVID-19 sangat tinggi (95%), namun banyak False Positive pada kelas Normal/Non-COVID. |

### Analisis Utama
1.  **Keunggulan SVM:** Fitur terstruktur dari HOG terbukti lebih *robust* untuk ukuran dataset ini dibandingkan fitur yang dipelajari dari nol oleh CNN kecil.
2.  **Sensitivitas CNN:** Meskipun akurasi total lebih rendah, model CNN+LoRA memiliki sensitivitas (Recall) 95% terhadap COVID-19, membuatnya "terlalu waspada" (sering menganggap Normal sebagai COVID).
3.  **Kesalahan Umum:** Mayoritas model kesulitan membedakan **Non-COVID vs COVID-19** akibat adanya artefak klinis (kabel, selang) yang menyerupai lesi paru.
