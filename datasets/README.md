# COVID-QU-Ex Dataset

## 📋 Deskripsi Dataset

**COVID-QU-Ex Dataset** adalah dataset citra Chest X-Ray (CXR) yang dikompilasi oleh peneliti dari Qatar University. Dataset ini merupakan salah satu dataset terbesar untuk deteksi dan segmentasi COVID-19 dari citra X-Ray.

### Statistik Dataset

- **Total Citra:** 33,920 gambar Chest X-Ray
- **Distribusi Kelas:**
  - **COVID-19:** 11,956 citra
  - **Non-COVID Infections** (Viral atau Bacterial Pneumonia): 11,263 citra
  - **Normal:** 10,701 citra
- **Ground-truth Masks:** Dataset ini menyediakan lung segmentation masks untuk seluruh dataset, menjadikannya dataset mask paru terbesar yang pernah dibuat.

### Keunikan Dataset

Menurut pengetahuan peneliti, ini adalah **studi pertama** yang memanfaatkan baik **lung segmentation** maupun **infection segmentation** untuk mendeteksi, melokalisasi, dan mengkuantifikasi infeksi COVID-19 dari citra X-Ray. Oleh karena itu, dataset ini dapat membantu dokter untuk:
- Mendiagnosis tingkat keparahan pneumonia COVID-19 dengan lebih baik
- Memantau perkembangan penyakit secara mudah

---

## 📊 Struktur Dataset

Dataset dibagi menjadi dua set eksperimen, masing-masing dibagi menjadi train, validation, dan test sets:

### 1. Lung Segmentation Data
- **Seluruh dataset COVID-QU-Ex** (33,920 citra CXR dengan ground-truth lung masks yang sesuai)

### 2. COVID-19 Infection Segmentation Data
- **Subset dari COVID-QU-Ex dataset:**
  - 1,456 citra Normal dengan lung mask yang sesuai
  - 1,457 citra Non-COVID-19 dengan lung mask yang sesuai
  - 2,913 citra COVID-19 dengan:
    - Lung mask dari COVID-QU-Ex dataset
    - Infection masks dari QaTaCov19 dataset

**Total untuk Infection Segmentation:** 5,826 citra (digunakan dalam proyek ini)

---

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

---

## 📝 Citation

**PENTING:** Jika Anda menggunakan dataset COVID-QU-Ex dalam penelitian Anda, mohon untuk mengutip publikasi/dataset berikut:

### Paper Utama

**[1]** A. M. Tahir, M. E. H. Chowdhury, A. Khandakar, Y. Qiblawey, U. Khurshid, S. Kiranyaz, N. Ibtehaz, M. S. Rahman, S. Al-Madeed, S. Mahmud, M. Ezeddin, K. Hameed, and T. Hamid, **"COVID-19 Infection Localization and Severity Grading from Chest X-ray Images"**, *Computers in Biology and Medicine*, vol. 139, p. 105002, 2021.  
DOI: https://doi.org/10.1016/j.compbiomed.2021.105002

### Dataset Citation

**[2]** Anas M. Tahir, Muhammad E. H. Chowdhury, Yazan Qiblawey, Amith Khandakar, Tawsifur Rahman, Serkan Kiranyaz, Uzair Khurshid, Nabil Ibtehaz, Sakib Mahmud, and Maymouna Ezeddin, **"COVID-QU-Ex"**, Kaggle, 2021.  
DOI: https://doi.org/10.34740/kaggle/dsv/3122958

### Paper Terkait

**[3]** T. Rahman, A. Khandakar, Y. Qiblawey A. Tahir S. Kiranyaz, S. Abul Kashem, M. Islam, S. Al Maadeed, S. Zughaier, M. Khan, M. Chowdhury, **"Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-rays Images"**, *Computers in Biology and Medicine*, p. 104319, 2021.  
DOI: https://doi.org/10.1016/j.compbiomed.2021.104319

**[4]** A. Degerli, M. Ahishali, M. Yamac, S. Kiranyaz, M. E. H. Chowdhury, K. Hameed, T. Hamid, R. Mazhar, and M. Gabbouj, **"Covid-19 infection map generation and detection from chest X-ray images"**, *Health Inf Sci Syst* 9, 15 (2021).  
DOI: https://doi.org/10.1007/s13755-021-00146-8

**[5]** M. E. H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. A. Emadi, M. B. I. Reaz, M. T. Islam, **"Can AI Help in Screening Viral and COVID-19 Pneumonia?"**, *IEEE Access*, vol. 8, pp. 132665-132676, 2020.  
DOI: https://doi.org/10.1109/ACCESS.2020.3010287

### Format BibTeX

```bibtex
@article{tahir2021covid,
  title={COVID-19 Infection Localization and Severity Grading from Chest X-ray Images},
  author={Tahir, A. M. and Chowdhury, M. E. H. and Khandakar, A. and Qiblawey, Y. and Khurshid, U. and Kiranyaz, S. and Ibtehaz, N. and Rahman, M. S. and Al-Madeed, S. and Mahmud, S. and Ezeddin, M. and Hameed, K. and Hamid, T.},
  journal={Computers in Biology and Medicine},
  volume={139},
  pages={105002},
  year={2021},
  doi={10.1016/j.compbiomed.2021.105002}
}

@dataset{covidqu2021,
  title={COVID-QU-Ex Dataset},
  author={Tahir, Anas M. and Chowdhury, Muhammad E. H. and Qiblawey, Yazan and Khandakar, Amith and Rahman, Tawsifur and Kiranyaz, Serkan and Khurshid, Uzair and Ibtehaz, Nabil and Mahmud, Sakib and Ezeddin, Maymouna},
  publisher={Kaggle},
  year={2021},
  doi={10.34740/kaggle/dsv/3122958}
}
```

---

## 📚 Sumber Data

Citra X-Ray dalam COVID-QU-Ex dikumpulkan dari repository dan studi berikut:

### COVID-19 Samples
1. **QaTa-COV19 Database**  
   https://www.kaggle.com/aysendegerli/qatacov19-dataset  
   Diakses: 14 Maret 2021

2. **Covid-19-image-repository**  
   https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png  
   Diakses: 14 Maret 2021

3. **Eurorad**  
   https://www.eurorad.org/  
   Diakses: 14 Maret 2021

4. **Covid-chestxray-dataset**  
   https://github.com/ieee8023/covid-chestxray-dataset  
   Diakses: 14 Maret 2021

5. **COVID-19 DATABASE (SIRM)**  
   https://www.sirm.org/category/senza-categoria/covid-19/  
   Diakses: 14 Maret 2021

6. **COVID-19 Radiography Database (Kaggle)**  
   https://www.kaggle.com/tawsifurrahman/covid19-radiography-database  
   Diakses: 14 Maret 2021

7. **COVID-CXNet (GitHub)**  
   https://github.com/armiro/COVID-CXNet  
   Diakses: 14 Maret 2021

### Non-COVID & Normal Samples
8. **RSNA Pneumonia Detection Challenge**  
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data  
   Diakses: 14 Maret 2021

9. **Chest X-Ray Images (Pneumonia)**  
   https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia  
   Diakses: 14 Maret 2021

10. **PadChest: Medical Imaging Databank of the Valencia Region**  
    https://bimcv.cipf.es/bimcv-projects/padchest/  
    Diakses: 14 Maret 2021

---

## 📊 Metadata Dataset

- **Usability Score:** 6.88/10
- **License:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0)
- **Update Frequency:** Not specified
- **Tags:** Universities and Colleges, Image, Data Visualization, Deep Learning, Healthcare, Coronavirus
- **File Count:** 85.3k files
- **Dataset Size:** ~1.21 GB

### Statistik Penggunaan (per November 2025)
- **Total Views:** 47.5K+
- **Total Downloads:** 10.4K+
- **Engagement Rate:** 0.21845 (downloads per view)

---

## ⚠️ Catatan Penting

1. **Lisensi:** Dataset ini berlisensi CC BY-SA 4.0. Pastikan untuk mematuhi ketentuan lisensi saat menggunakan dataset ini.

2. **Penggunaan Medis:** Dataset ini dibuat untuk tujuan penelitian dan pendidikan. **TIDAK** untuk digunakan sebagai alat diagnostik medis langsung tanpa validasi klinis yang memadai.

3. **Etika Penelitian:** Pastikan penelitian yang menggunakan dataset ini mematuhi etika penelitian medis dan mendapatkan persetujuan yang diperlukan.

4. **Preprocessing:** Dataset mungkin memerlukan preprocessing tambahan tergantung pada kebutuhan penelitian Anda. Lihat notebook di folder utama untuk contoh preprocessing yang digunakan dalam proyek ini.

---

## 🔗 Link Terkait

- **Kaggle Dataset:** https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
- **Qatar University:** https://www.qu.edu.qa/
- **Related Notebooks di Kaggle:**
  - DENSENET201_COVIDQU-Ex
  - Xception
  - Alzheimers (menggunakan dataset ini)

---

## 📧 Kontak Dataset

Untuk pertanyaan tentang dataset, silakan hubungi:
- **Anas M. Tahir** dan tim peneliti Qatar University
- Melalui platform Kaggle atau email institusi

---

**Last Updated:** November 2025  
**Dataset Version:** COVID-QU-Ex (2021)

