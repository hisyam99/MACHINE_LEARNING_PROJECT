# TASK 1: Classic ML

## 📋 Deskripsi

Task ini mengimplementasikan pendekatan Machine Learning Klasik dengan feature engineering manual menggunakan HOG (Histogram of Oriented Gradients) dan berbagai algoritma klasik.

## 🔬 Implementasi

### 1. Preprocessing

Pipeline preprocessing yang sama dengan yang digunakan untuk deep learning:

1. Grayscale conversion
2. Resize ke 224×224
3. CLAHE
4. Heuristic lung cropping
5. Normalisasi

### 2. Feature Extraction

#### HOG Features

```python
from skimage.feature import hog

def extract_hog_features(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features
```

**Output:** 6,084 dimensi

#### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=4096)
X_selected = selector.fit_transform(X_hog, y)
```

**Output:** 4,096 fitur terbaik

#### Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
```

### 3. Model Training

#### SVM (RBF Kernel)

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

#### Random Forest

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

#### k-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)
knn.fit(X_train, y_train)
```

## 📊 Hasil

### Performa pada Test Set

| Model | Akurasi | Macro F1 | Precision | Recall |
|:------|:-------:|:--------:|:---------:|:------:|
| **SVM (RBF)** | **86.27%** | **0.843** | 0.862 | 0.863 |
| **kNN (k=5)** | 77.57% | 0.739 | 0.776 | 0.776 |
| **Random Forest** | 76.09% | 0.719 | 0.761 | 0.761 |

### Confusion Matrix (SVM)

| | Predicted COVID-19 | Predicted Non-COVID | Predicted Normal |
|:---|:-----------------:|:-------------------:|:----------------:|
| **Actual COVID-19** | 417 | 3 | 12 |
| **Actual Non-COVID** | 42 | 157 | 20 |
| **Actual Normal** | 21 | 17 | 180 |

### Analisis Per Kelas (SVM)

| Kelas | Precision | Recall | F1-Score |
|:------|:---------:|:------:|:--------:|
| **COVID-19** | 0.870 | 0.965 | 0.915 |
| **Non-COVID** | 0.887 | 0.718 | 0.794 |
| **Normal** | 0.849 | 0.826 | 0.837 |

## 💡 Analisis

### Kelebihan SVM

1. **Performa Terbaik:** 86.27% akurasi, terbaik di antara model klasik
2. **Stabil:** Konsisten dan reliable
3. **Presisi Tinggi:** Khususnya untuk kelas COVID-19 (0.870 precision)
4. **Tidak Perlu GPU:** Dapat dijalankan pada CPU

### Keterbatasan

1. **Ukuran Model:** Relatif besar (~95 MB)
2. **Training Time:** Lebih lama dibandingkan RF dan kNN
3. **Memory:** Memerlukan lebih banyak memory untuk dataset besar

## 🔍 Kesalahan Umum

### False Positives

- Normal sering diprediksi sebagai COVID-19 (21 kasus)
- Non-COVID sering diprediksi sebagai COVID-19 (42 kasus)

### False Negatives

- COVID-19 jarang terlewat (hanya 15 dari 432)
- Non-COVID lebih sering terlewat (62 dari 219)

## 📈 Visualisasi

Lihat visualisasi lengkap di:
- Confusion Matrix: `output_images/conf_matrix_svm.png`
- Classification Report: Tersedia di notebook

## 🔗 Referensi

- [Notebook: TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb](https://github.com/hisyam99/MACHINE_LEARNING_PROJECT/blob/main/TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb)
- [HOG Feature Descriptor](../methodology/classic-ml.md#1-hog-histogram-of-oriented-gradients)

