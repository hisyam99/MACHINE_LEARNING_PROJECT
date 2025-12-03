# Machine Learning Klasik

## 🎯 Pendekatan

Machine Learning Klasik menggunakan **feature engineering manual** untuk mengekstrak fitur dari citra, kemudian menggunakan algoritma klasik untuk klasifikasi.

## 🔬 Feature Engineering

### Pipeline Feature Engineering

Feature engineering adalah tahap krusial yang mengubah gambar yang sudah seragam menjadi representasi numerik yang kaya informasi, tetapi tetap ringkas dan stabil untuk dipakai oleh model. Pipeline terdiri dari tiga komponen utama: **Feature Extraction**, **Feature Selection**, dan **Feature Scaling**.

### 1. HOG (Histogram of Oriented Gradients) - Feature Extraction {#1-hog-histogram-of-oriented-gradients}

HOG (Histogram of Oriented Gradients) adalah metode ekstraksi fitur yang menangkap tekstur dan pola tepi dari citra dengan menganalisis distribusi gradien orientasi. Metode ini sangat efektif untuk mendeteksi pola struktur pada citra medis, khususnya pola infiltrasi dan opasitas pada paru-paru.

**Parameter yang Digunakan:**
- **Orientations:** 9 bins
- **Pixels per cell:** (16, 16) piksel
- **Cells per block:** (2, 2) sel
- **Block normalization:** L2-Hys
- **Feature vector:** True (output sebagai vektor 1D)

**Dimensi Output:** Vektor fitur **6,084 dimensi** per gambar

**Tabel 2.1: Konfigurasi HOG**

| Parameter | Nilai | Alasan Singkat |
|:----------|:------|:---------------|
| **orientations** | 9 | Granularitas arah tepi yang umum dipakai |
| **pixels_per_cell** | (16, 16) | Menangkap tekstur paru pada skala lokal |
| **cells_per_block** | (2, 2) | Normalisasi gradien yang stabil |
| **feature_vector** | True | Vektor 1D siap ke tahap seleksi & skala |
| **Dimensi HOG** | 6,084 | Setelah diratakan per gambar |

**Mengapa HOG Efektif untuk Citra X-Ray Paru?**

1. **Menangkap Pola Gradien:** Kelainan pada paru seperti ground-glass opacity, konsolidasi, dan infiltrat interstisial menghasilkan pola gradien intensitas yang khas. HOG mampu mengkuantifikasi pola-pola ini secara robust.

2. **Invariant terhadap Iluminasi Lokal:** Normalisasi blok pada HOG membuat fitur relatif stabil terhadap variasi pencahayaan global, yang sering terjadi pada citra X-ray dari sumber berbeda.

3. **Representasi Tekstur Multi-Scale:** Dengan membagi citra menjadi sel-sel kecil dan menggabungkannya dalam blok, HOG dapat menangkap tekstur pada berbagai skala spatial, dari detail halus hingga pola regional.

4. **Komputasi Efisien:** HOG dapat diekstrak dengan cepat tanpa memerlukan GPU, menjadikannya pilihan praktis untuk baseline klasik.

**Proses Ekstraksi:**

Seluruh matriks fitur HOG untuk semua gambar dalam dataset disimpan ke file `hog_features_classic.npz` (ukuran ~217 MB). Penyimpanan ini memungkinkan:
- Reprodusibilitas eksperimen tanpa perlu re-ekstraksi
- Percepatan iterasi hyperparameter tuning
- Konsistensi fitur antar split (train/val/test)

```python
from skimage.feature import hog

features = hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    visualize=False
)
```

### 2. Feature Selection (SelectKBest)

Setelah ekstraksi HOG menghasilkan 6,084 fitur, langkah berikutnya adalah **feature selection** untuk mengurangi dimensi dan memilih fitur yang paling informatif. Jumlah fitur yang besar dapat menyebabkan:
- **Curse of dimensionality:** Performa model menurun pada ruang fitur berdimensi tinggi
- **Overfitting:** Model menghafal noise alih-alih pola sebenarnya
- **Komputasi lambat:** Training dan inference memerlukan waktu lebih lama

**Metode:** SelectKBest dengan ANOVA F-test (f_classif)

**Prinsip Kerja:**
ANOVA F-test mengukur seberapa baik setiap fitur dapat membedakan antar kelas dengan menghitung rasio variance between-class terhadap variance within-class. Fitur dengan F-score tertinggi adalah fitur yang paling diskriminatif.

**Parameter:**
- **Method:** ANOVA F-test (f_classif)
- **K:** 4,096 fitur terbaik (dari 6,084 fitur)
- **Fit:** Hanya pada data train
- **Transform:** Train, validation, dan test

**Tabel 2.2: Uji Nilai k pada SelectKBest (Validation Macro-F1)**

| k | Macro-F1 (Val) |
|:--|:---------------|
| 256 | 0.6843 |
| 512 | 0.6944 |
| 1,024 | 0.7317 |
| 2,048 | 0.7462 |
| **4,096** | **0.7669** |

**Hasil Tuning Hyperparameter:**

Untuk menentukan nilai k yang optimal, kami menguji beberapa kandidat nilai k (256 hingga 4,096) dan menilai kinerjanya pada validation set menggunakan macro-F1. Kinerja meningkat konsisten sampai k=4,096 dan berhenti membaik setelahnya (bahkan cenderung menurun karena mulai memasukkan noise). Oleh karena itu, kami menetapkan **k=4,096** sebagai pilihan akhir.

**Implementasi:**

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Fit hanya pada train set
selector = SelectKBest(score_func=f_classif, k=4096)
X_train_selected = selector.fit_transform(X_train_hog, y_train)

# Transform validation dan test menggunakan selector yang sama
X_val_selected = selector.transform(X_val_hog)
X_test_selected = selector.transform(X_test_hog)
```

**Penyimpanan:**
Selektor yang sudah terlatih disimpan sebagai `feature_selector_classic.joblib` untuk memastikan konsistensi transformasi pada data baru saat inferensi.

**Keuntungan Feature Selection:**
1. **Reduksi Dimensi:** Dari 6,084 menjadi 4,096 fitur (~33% reduction)
2. **Peningkatan Performa:** Macro-F1 meningkat dari ~0.70 menjadi ~0.77
3. **Reduksi Overfitting:** Model lebih fokus pada fitur informatif
4. **Percepatan Training:** Waktu training berkurang ~25%

### 3. Scaling (StandardScaler)

Setelah fitur diseleksi, langkah terakhir dalam feature engineering adalah **feature scaling** menggunakan StandardScaler. Penskalaan ini ditempatkan **setelah seleksi** agar statistik mean dan deviasi standar dihitung persis pada subruang fitur yang betul-betul dipakai model.

**Mengapa Scaling Penting?**

1. **Stabilitas Numerik:** Fitur dengan skala berbeda dapat menyebabkan masalah numerik pada optimisasi
2. **Kesetaraan Pengaruh:** Mencegah fitur dengan nilai besar mendominasi fitur dengan nilai kecil
3. **Konvergensi Lebih Cepat:** Algoritma seperti SVM dan gradient-based methods konvergen lebih cepat pada data yang ter-standarisasi
4. **Performa Optimal:** Banyak algoritma ML dirancang dengan asumsi fitur ter-standarisasi

**Metode: StandardScaler**

StandardScaler mentransformasi setiap fitur sehingga memiliki:
- **Mean:** 0
- **Standard Deviation:** 1

Formula: `X_scaled = (X - μ) / σ`

Dimana:
- `μ` adalah mean dari fitur pada training set
- `σ` adalah standard deviation dari fitur pada training set

**Tabel 2.3: Standardisasi Fitur**

| Komponen | Kebijakan Fit/Transform | Catatan |
|:---------|:------------------------|:--------|
| **SelectKBest** | Fit di Train → Transform Val/Test | k=4096, f_classif |
| **StandardScaler** | Fit di Train → Transform Val/Test | Skala setelah seleksi |

**Implementasi:**

```python
from sklearn.preprocessing import StandardScaler

# Fit hanya pada train set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)

# Transform validation dan test menggunakan statistik dari train
X_val_scaled = scaler.transform(X_val_selected)
X_test_scaled = scaler.transform(X_test_selected)
```

**Penyimpanan:**
StandardScaler yang sudah fit disimpan sebagai `feature_scaler_classic.joblib` untuk digunakan pada inferensi.

**Urutan Pipeline yang Konsisten:**

Sangat penting untuk selalu menerapkan transformasi dalam urutan yang sama:

1. **HOG Extraction** → 6,084 fitur
2. **SelectKBest** → 4,096 fitur terbaik
3. **StandardScaler** → Fitur ter-standarisasi

Urutan yang konsisten ini kami gunakan hingga ke tahap inferensi satu gambar, memastikan bahwa model menerima input dalam format yang persis sama seperti saat training.

## 🤖 Model yang Diuji

Dengan fitur yang sudah siap (HOG → SelectKBest → StandardScaler), kami melatih tiga model klasik: **SVM dengan kernel RBF**, **Random Forest**, dan **k-Nearest Neighbors**. Semua model dilatih pada data latih yang telah melalui pipeline feature engineering lengkap, kemudian dievaluasi di validation dan test set.

**Tabel 3.1: Ringkasan Hiperparameter Model**

| Model | Hiperparameter Utama |
|:------|:---------------------|
| **SVM RBF** | C=10, gamma="scale", class_weight="balanced", probas=On |
| **RandomForest** | n_estimators=300, class_weight="balanced_subsample" |
| **kNN** | n_neighbors=5 |

### 1. SVM (Support Vector Machine) dengan Kernel RBF

SVM adalah algoritma pembelajaran mesin yang mencari hyperplane optimal untuk memisahkan kelas-kelas dalam ruang fitur. Pada proyek ini, kami menggunakan **kernel RBF (Radial Basis Function)** yang mampu menangani pola non-linear dengan memetakan data ke ruang berdimensi lebih tinggi.

**Kernel:** RBF (Radial Basis Function)

**Formula Kernel RBF:**
`K(x, x') = exp(-γ ||x - x'||²)`

**Hyperparameter:**
- **C:** 10 (regularization parameter, mengontrol trade-off antara margin maksimum dan kesalahan klasifikasi)
- **Gamma:** 'scale' (parameter kernel RBF, diset sebagai 1 / (n_features × variance))
- **Class Weight:** 'balanced' (untuk menangani imbalance kelas)
- **Probability:** True (untuk mendapatkan probabilitas prediksi)
- **Random State:** 42 (untuk reprodusibilitas)

**Hasil:**
- **Akurasi Test:** 86.27%
- **Macro F1:** 0.843
- **Precision (macro):** 0.860
- **Recall (macro):** 0.832

**Tabel 3.2: Kinerja Model - Validation & Test**

| Model | Split | Akurasi | Precision (macro) | Recall (macro) | F1 (macro) |
|:------|:------|:--------|:------------------|:---------------|:-----------|
| **SVM RBF** | Val | 0.8478 | 0.8313 | 0.8178 | 0.8234 |
| **SVM RBF** | Test | 0.8627 | 0.8601 | 0.8323 | 0.8432 |

**Kelebihan SVM RBF:**
- ✅ **Performa terbaik** di antara model klasik (86.27% akurasi)
- ✅ **Stabil dan konsisten** antara validation dan test
- ✅ **Presisi tinggi** terutama untuk kelas COVID-19
- ✅ **Robust terhadap high dimensionality** dengan 4,096 fitur
- ✅ **Output probabilitas** untuk analisis confidence

**Keterbatasan:**
- ⚠️ **Ukuran model besar:** ~95.7 MB (menyimpan support vectors)
- ⚠️ **Training time lebih lama:** ~10-15 menit pada CPU
- ⚠️ **Inference latency:** ~15 ms per sampel (lebih lambat dari RF)

**Implementasi:**

```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # Untuk confidence scores
    random_state=42
)

# Training
svm.fit(X_train_scaled, y_train)

# Prediction dengan probabilitas
y_pred = svm.predict(X_test_scaled)
y_proba = svm.predict_proba(X_test_scaled)
```

**Analisis Per Kelas (Test Set):**

Confusion Matrix pada SVM memperlihatkan karakteristik yang menarik:

- **Kelas COVID-19:** Recall sekitar 95% - model sangat sensitif dalam mendeteksi kasus positif. Dari perspektif klinis, ini sangat penting karena melewatkan kasus COVID-19 (false negative) berpotensi berbahaya.

- **Kelas Non-COVID:** Masih menjadi sumber kesalahan terbesar. Sebagian kasusnya didorong ke kelas COVID-19 (false positive). Dari peninjauan visual, kesalahan seperti ini kerap terjadi pada citra yang menyertakan kabel, tube, atau artefak klinis lain yang memperkaya struktur tepi sehingga "terbaca" sebagai pola yang mirip lesi.

- **Kelas Normal:** Relatif lebih stabil. Struktur paru yang bersih setelah CLAHE membuatnya lebih mudah dipisahkan dari dua kelas lainnya. Recall sekitar 83%, menunjukkan bahwa model dapat mengidentifikasi paru sehat dengan baik.

**Model disimpan ke:** `artifacts/classic_models/svm_rbf.joblib`

### 2. Random Forest

Random Forest adalah ensemble learning method yang membangun banyak decision trees selama training dan menggabungkan prediksi mereka melalui voting (untuk klasifikasi). Setiap tree dilatih pada subset data yang berbeda (bootstrap sampling) dan subset fitur yang berbeda, membuat model lebih robust dan mengurangi overfitting.

**Hyperparameter:**
- **N Estimators:** 300 (jumlah decision trees dalam forest)
- **Max Depth:** None (trees dikembangkan hingga semua leaves pure atau berisi min_samples_split sampel)
- **Class Weight:** 'balanced_subsample' (weight disesuaikan untuk setiap tree berdasarkan bootstrap sample)
- **Min Samples Split:** 2 (jumlah minimum sampel untuk split node)
- **Min Samples Leaf:** 1 (jumlah minimum sampel pada leaf node)
- **Random State:** 42 (untuk reprodusibilitas)

**Hasil:**
- **Akurasi Test:** 76.09%
- **Macro F1:** 0.7188
- **Precision (macro):** 0.8066
- **Recall (macro):** 0.6966

**Tabel 3.2: Kinerja Random Forest**

| Model | Split | Akurasi | Precision (macro) | Recall (macro) | F1 (macro) |
|:------|:------|:--------|:------------------|:---------------|:-----------|
| **RandomForest** | Val | 0.7551 | 0.8050 | 0.6829 | 0.7044 |
| **RandomForest** | Test | 0.7609 | 0.8066 | 0.6966 | 0.7188 |

**Kelebihan Random Forest:**
- ✅ **Robust terhadap overfitting** melalui ensemble averaging
- ✅ **Dapat menangani non-linearitas** tanpa feature engineering tambahan
- ✅ **Feature importance tersedia** untuk interpretabilitas
- ✅ **Inference cepat:** ~0.14 ms per sampel (tercepat)
- ✅ **Model ringan:** ~21.16 MB
- ✅ **Tidak perlu feature scaling** (tree-based method)

**Kekurangan:**
- ⚠️ **Struggles dengan hubungan non-linear kompleks** pada dataset ini
- ⚠️ **Performa lebih rendah** dibandingkan SVM (76.09% vs 86.27%)
- ⚠️ **Cenderung bias** terhadap kelas mayoritas meskipun sudah menggunakan class weight

**Implementasi:**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1  # Paralelisasi untuk percepatan training
)

# Training
rf.fit(X_train_scaled, y_train)

# Prediction
y_pred = rf.predict(X_test_scaled)

# Feature importance untuk interpretabilitas
feature_importances = rf.feature_importances_
```

**Analisis:**

Random Forest mencapai akurasi 76% dengan macro-F1 0.719. Model cenderung stabil tetapi tidak sekuat SVM karena keterbatasan dalam menangkap hubungan non-linear antar-fitur yang kompleks pada ruang fitur HOG 4,096 dimensi.

**Feature Importances:**
Random Forest memberikan insight tentang fitur mana yang paling berkontribusi dalam klasifikasi. Dari analisis, fitur-fitur HOG yang dominan berasal dari area paru bawah dan tengah, yang merupakan lokasi umum manifestasi pneumonia.

**Model disimpan ke:** `artifacts/classic_models/random_forest.joblib`

### 3. k-Nearest Neighbors (kNN)

kNN adalah algoritma instance-based learning yang mengklasifikasikan sampel baru berdasarkan "voting" dari k tetangga terdekat dalam ruang fitur. Algoritma ini tidak membangun model eksplisit selama training (lazy learning), melainkan menyimpan seluruh training data dan melakukan komputasi saat inferensi.

**Hyperparameter:**
- **K (n_neighbors):** 5 (jumlah tetangga yang dipertimbangkan)
- **Weights:** 'distance' (tetangga lebih dekat memiliki bobot lebih besar: w = 1 / distance)
- **Metric:** 'euclidean' (jarak Euclidean dalam ruang fitur 4,096 dimensi)
- **Algorithm:** 'auto' (memilih algoritma optimal: ball_tree, kd_tree, atau brute)

**Hasil:**
- **Akurasi Test:** 77.57%
- **Macro F1:** 0.7389
- **Precision (macro):** 0.7841
- **Recall (macro):** 0.7225

**Tabel 3.2: Kinerja kNN**

| Model | Split | Akurasi | Precision (macro) | Recall (macro) | F1 (macro) |
|:------|:------|:--------|:------------------|:---------------|:-----------|
| **kNN (k=5)** | Val | 0.7769 | 0.7971 | 0.7195 | 0.7345 |
| **kNN (k=5)** | Test | 0.7757 | 0.7841 | 0.7225 | 0.7389 |

**Kelebihan kNN:**
- ✅ **Simple dan interpretable:** Keputusan dapat ditraceIback ke tetangga konkret
- ✅ **Tidak memerlukan training eksplisit:** Lazy learning
- ✅ **Non-parametric:** Tidak membuat asumsi tentang distribusi data
- ✅ **Dapat handle multi-class naturally:** Voting langsung untuk multiple classes

**Kekurangan:**
- ⚠️ **Rentan terhadap curse of dimensionality:** Performa menurun pada ruang 4,096 dimensi
- ⚠️ **Sensitif terhadap noise dan outliers:** Tetangga yang salah dapat mengubah prediksi
- ⚠️ **Computationally expensive pada inferensi:** ~127.47 ms per sampel (paling lambat)
- ⚠️ **Memory intensive:** Harus menyimpan seluruh training set (~127.47 MB)
- ⚠️ **Sensitif terhadap skala fitur:** Memerlukan feature scaling yang tepat

**Implementasi:**

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Tetangga lebih dekat, bobot lebih besar
    metric='euclidean',
    algorithm='auto',    # Otomatis pilih algoritma optimal
    n_jobs=-1           # Paralelisasi untuk percepatan
)

# "Training" (hanya menyimpan data)
knn.fit(X_train_scaled, y_train)

# Prediction
y_pred = knn.predict(X_test_scaled)

# Mendapatkan tetangga terdekat untuk analisis
distances, indices = knn.kneighbors(X_test_scaled)
```

**Analisis:**

Model kNN dengan k=5 memiliki akurasi 77.57% dengan macro-F1 0.739. Model ini sangat bergantung pada jarak fitur sehingga lebih rentan terhadap distribusi fitur yang padat pada ruang berdimensi tinggi (4,096 dimensi).

**Masalah Curse of Dimensionality:**
Pada ruang 4,096 dimensi, konsep "tetangga terdekat" menjadi kurang bermakna karena:
1. Jarak antar semua titik cenderung sama (concentration of distances)
2. Volume ruang meningkat eksponensial dengan dimensi
3. Sampel training menjadi sparse di ruang fitur

**Analisis Tetangga:**
Visualisasi t-SNE 2D memperlihatkan pengelompokan COVID-19 yang cukup jelas, namun cluster Non-COVID dan Normal sering berdekatan. Hal ini menandakan jarak Euclidean pada fitur HOG belum sepenuhnya merepresentasikan variasi klinis.

**Model disimpan ke:** `artifacts/classic_models/knn.joblib`

## 📊 Perbandingan Performa

| Model | Akurasi | Macro F1 | Catatan |
|:------|:-------:|:--------:|:--------|
| **SVM (RBF)** | **86.27%** | **0.843** | Best overall, stabil, presisi tinggi |
| **kNN (k=5)** | 77.57% | 0.739 | Rentan high dimensionality |
| **Random Forest** | 76.09% | 0.719 | Struggles dengan non-linear kompleks |

## 💡 Kesimpulan

1. **SVM adalah pilihan terbaik** untuk machine learning klasik pada dataset ini
2. **HOG features** terbukti efektif untuk menangkap karakteristik citra X-Ray
3. **Feature selection** penting untuk mengurangi dimensi dan meningkatkan performa
4. **Class balancing** penting untuk dataset dengan distribusi tidak seimbang

## 🔗 Referensi

- [HOG Feature Descriptor](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

