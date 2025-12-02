# Machine Learning Klasik

## 🎯 Pendekatan

Machine Learning Klasik menggunakan **feature engineering manual** untuk mengekstrak fitur dari citra, kemudian menggunakan algoritma klasik untuk klasifikasi.

## 🔬 Feature Engineering

### 1. HOG (Histogram of Oriented Gradients)

HOG menangkap tekstur dan pola tepi dari citra dengan menganalisis distribusi gradien orientasi.

**Parameter:**
- **Orientations:** 9
- **Pixels per cell:** (8, 8)
- **Cells per block:** (2, 2)
- **Block normalization:** L2-Hys

**Output:** Vektor fitur 6,084 dimensi

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

Menggunakan ANOVA F-test untuk memilih fitur terbaik.

**Parameter:**
- **Method:** ANOVA F-test
- **K:** 4,096 (dari 6,084 fitur)

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=4096)
X_selected = selector.fit_transform(X, y)
```

### 3. Scaling (StandardScaler)

Standarisasi fitur untuk stabilitas distribusi.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
```

## 🤖 Model yang Diuji

### 1. SVM (Support Vector Machine)

**Kernel:** RBF (Radial Basis Function)

**Hyperparameter:**
- **C:** 10
- **Gamma:** 'scale'
- **Class Weight:** 'balanced'

**Hasil:**
- **Akurasi:** 86.27%
- **Macro F1:** 0.843

**Kelebihan:**
- ✅ Performa terbaik di antara model klasik
- ✅ Stabil dan konsisten
- ✅ Presisi tinggi

```python
from sklearn.svm import SVC

svm = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    random_state=42
)
svm.fit(X_train, y_train)
```

### 2. Random Forest

**Hyperparameter:**
- **N Estimators:** 300
- **Max Depth:** None
- **Class Weight:** 'balanced'

**Hasil:**
- **Akurasi:** 76.09%
- **Macro F1:** 0.719

**Kelebihan:**
- ✅ Robust terhadap overfitting
- ✅ Dapat menangani non-linearitas
- ✅ Feature importance tersedia

**Kekurangan:**
- ⚠️ Struggles dengan non-linear kompleks
- ⚠️ Perlu banyak trees untuk performa optimal

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
```

### 3. k-Nearest Neighbors (kNN)

**Hyperparameter:**
- **K:** 5
- **Weights:** 'distance'
- **Metric:** 'euclidean'

**Hasil:**
- **Akurasi:** 77.57%
- **Macro F1:** 0.739

**Kelebihan:**
- ✅ Simple dan interpretable
- ✅ Tidak memerlukan training (lazy learning)

**Kekurangan:**
- ⚠️ Rentan terhadap high dimensionality
- ⚠️ Sensitif terhadap noise
- ⚠️ Computationally expensive untuk dataset besar

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)
knn.fit(X_train, y_train)
```

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

