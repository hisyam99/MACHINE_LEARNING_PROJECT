# Dataset COVID-QU-Ex

## 📋 Deskripsi

**COVID-QU-Ex Dataset** adalah dataset citra Chest X-Ray (CXR) yang dikompilasi oleh peneliti dari Qatar University. Dataset ini merupakan salah satu dataset terbesar untuk deteksi dan segmentasi COVID-19 dari citra X-Ray.

## 📊 Statistik Dataset Lengkap

- **Total Citra:** 33,920 gambar Chest X-Ray
- **Distribusi Kelas (Full Dataset):**
  - **COVID-19:** 11,956 citra
  - **Non-COVID Infections** (Viral atau Bacterial Pneumonia): 11,263 citra
  - **Normal:** 10,701 citra
- **Ground-truth Masks:** Dataset ini menyediakan lung segmentation masks untuk seluruh dataset

## 📦 Dataset yang Digunakan dalam Proyek

Untuk proyek ini, kami menggunakan **Infection Segmentation Data** subset dari COVID-QU-Ex:

| Kelas | Jumlah | Persentase |
|:------|:------:|:----------:|
| **COVID-19** | 2,913 | 50.0% |
| **Non-COVID** | 1,457 | 25.0% |
| **Normal** | 1,456 | 25.0% |
| **Total** | **5,826** | **100%** |

### Pembagian Data

| Split | Jumlah | Persentase |
|:------|:------:|:----------:|
| **Train** | 4,078 | 70% |
| **Validation** | 874 | 15% |
| **Test** | 874 | 15% |

## 🎯 Keunikan Dataset

Menurut pengetahuan peneliti, ini adalah **studi pertama** yang memanfaatkan baik **lung segmentation** maupun **infection segmentation** untuk mendeteksi, melokalisasi, dan mengkuantifikasi infeksi COVID-19 dari citra X-Ray.

Dataset ini dapat membantu dokter untuk:
- Mendiagnosis tingkat keparahan pneumonia COVID-19 dengan lebih baik
- Memantau perkembangan penyakit secara mudah

## 📥 Cara Mendapatkan Dataset

### Download dari Kaggle

1. **Akses Dataset:**
   - URL: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
   - Atau cari "COVID-QU-Ex Dataset" di Kaggle

2. **Menggunakan Kaggle API:**
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Setup credentials (dapatkan dari Kaggle Account Settings)
   # Letakkan kaggle.json di ~/.kaggle/
   
   # Download dataset
   kaggle datasets download -d anasmohammedtahir/covidqu
   
   # Extract
   unzip covidqu.zip -d datasets/
   ```

3. **Struktur Folder Setelah Extract:**
   ```
   datasets/
   ├── Infection Segmentation Data/
   │   └── Infection Segmentation Data/
   │       ├── Train/
   │       │   ├── COVID-19/
   │       │   ├── Non-COVID/
   │       │   └── Normal/
   │       ├── Val/
   │       │   ├── COVID-19/
   │       │   ├── Non-COVID/
   │       │   └── Normal/
   │       └── Test/
   │           ├── COVID-19/
   │           ├── Non-COVID/
   │           └── Normal/
   └── Lung Segmentation Data/
       └── Lung Segmentation Data/
           └── ...
   ```

## 📊 Metadata Dataset

- **Usability Score:** 6.88/10
- **License:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0)
- **File Count:** 85.3k files
- **Dataset Size:** ~1.21 GB

### Statistik Penggunaan (per November 2025)
- **Total Views:** 47.5K+
- **Total Downloads:** 10.4K+
- **Engagement Rate:** 0.21845 (downloads per view)

## ⚠️ Catatan Penting

1. **Lisensi:** Dataset ini berlisensi CC BY-SA 4.0. Pastikan untuk mematuhi ketentuan lisensi saat menggunakan dataset ini.

2. **Penggunaan Medis:** Dataset ini dibuat untuk tujuan penelitian dan pendidikan. **TIDAK** untuk digunakan sebagai alat diagnostik medis langsung tanpa validasi klinis yang memadai.

3. **Etika Penelitian:** Pastikan penelitian yang menggunakan dataset ini mematuhi etika penelitian medis dan mendapatkan persetujuan yang diperlukan.

4. **Preprocessing:** Dataset mungkin memerlukan preprocessing tambahan tergantung pada kebutuhan penelitian Anda.

[📖 Lihat preprocessing yang digunakan →](../methodology/preprocessing.md)

[📚 Lihat citation dan referensi →](citation.md)

