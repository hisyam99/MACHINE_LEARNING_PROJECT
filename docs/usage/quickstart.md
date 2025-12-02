# Quick Start

## 🚀 Memulai dengan Cepat

Panduan ini akan membantu Anda memulai proyek dalam beberapa menit.

## 📋 Checklist Awal

- [ ] Python 3.8+ terinstall
- [ ] Dependencies terinstall (lihat [Installation](installation.md))
- [ ] Dataset sudah didownload dan di-extract
- [ ] Jupyter Notebook terinstall

## 🎯 Workflow Dasar

### 1. Setup Environment

```bash
# Aktifkan virtual environment (jika menggunakan)
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### 2. Download Dataset

```bash
# Menggunakan Kaggle API
kaggle datasets download -d anasmohammedtahir/covidqu
unzip covidqu.zip -d datasets/
```

### 3. Jalankan Jupyter Notebook

```bash
jupyter notebook
```

### 4. Pilih Task

Berdasarkan kebutuhan Anda:

#### TASK 1: Classic ML (SVM, RF, KNN)

**File:** `TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb`

**Kapan digunakan:**
- Ingin baseline yang cepat tanpa GPU
- Ingin memahami feature engineering manual
- Ingin hasil yang stabil dan interpretable

**Waktu:** ~10-15 menit (CPU)

#### TASK 2: Custom CNN + LoRA

**File:** `TASK_2:CNN+LoRA.ipynb`

**Kapan digunakan:**
- Ingin model lightweight untuk mobile/edge
- Ingin memahami CNN dari scratch
- Ingin eksperimen dengan LoRA

**Waktu:** ~30-45 menit (GPU recommended)

#### TASK 3: Transfer Learning + ViT

**File:** `TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb`

**Kapan digunakan:**
- Ingin akurasi maksimal
- Ingin menggunakan pretrained models
- Ingin eksperimen dengan Vision Transformer

**Waktu:** ~1-3 jam (GPU required)

## 📝 Contoh Penggunaan

### Preprocessing Data

```python
import cv2
import numpy as np
from pathlib import Path

def load_and_preprocess(path):
    """Load and preprocess image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    # ... (CLAHE, cropping, dll)
    return img
```

### Training Model

```python
# Contoh: Custom CNN
from tensorflow import keras

model = build_custom_cnn()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint]
)
```

### Evaluasi Model

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_true, y_pred_classes))
print(confusion_matrix(y_true, y_pred_classes))
```

## 🎓 Tutorial Step-by-Step

### Tutorial 1: Classic ML (SVM)

1. Buka `TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb`
2. Jalankan semua cells secara berurutan
3. Hasil akan tersimpan di folder `artifacts/`
4. Visualisasi tersimpan di `output_images/`

### Tutorial 2: Custom CNN

1. Buka `TASK_2:CNN+LoRA.ipynb`
2. Pastikan dataset sudah di-setup dengan benar
3. Jalankan preprocessing cells
4. Build dan train model
5. Evaluasi dan visualisasi hasil

### Tutorial 3: Transfer Learning

1. Buka `TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb`
2. Install transformers jika belum: `pip install transformers`
3. Jalankan semua cells
4. Bandingkan hasil dengan model lain

## 🔧 Konfigurasi

### GPU Setup (Opsional)

```python
import tensorflow as tf

# Check GPU
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Set memory growth (jika perlu)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

### Path Configuration

```python
# Update paths sesuai struktur folder Anda
ROOT = Path("/path/to/dataset")
ARTIFACTS = Path("./artifacts")
OUTPUT_IMAGES = Path("./output_images")
```

## 📊 Output

Setelah menjalankan eksperimen, Anda akan mendapatkan:

1. **Model Weights:** Di folder `artifacts/`
2. **Visualisasi:** Di folder `output_images/`
3. **Metrics:** Classification reports di notebook
4. **Confusion Matrix:** PNG files di `output_images/`

## 🐛 Troubleshooting

### Memory Error

```python
# Kurangi batch size
batch_size = 16  # atau lebih kecil

# Gunakan data generator
def data_generator(X, y, batch_size):
    # Implementasi generator
    pass
```

### GPU Not Found

```bash
# Check CUDA installation
nvidia-smi

# Install CUDA dan cuDNN
# Lihat: https://www.tensorflow.org/install/gpu
```

### Dataset Not Found

```python
# Verify dataset path
from pathlib import Path
dataset_path = Path("datasets/Infection Segmentation Data/Infection Segmentation Data")
print("Dataset exists:", dataset_path.exists())
print("Train exists:", (dataset_path / "Train").exists())
```

## 📚 Next Steps

Setelah quick start:

1. [Notebooks Guide](notebooks.md) - Detail setiap notebook
2. [Methodology](../methodology/preprocessing.md) - Memahami metodologi
3. [Results](../results/overview.md) - Melihat hasil eksperimen

## 💡 Tips

1. **Mulai dengan TASK 1** untuk memahami workflow dasar
2. **Gunakan GPU** untuk TASK 2 dan 3 jika tersedia
3. **Simpan checkpoint** secara berkala
4. **Monitor training** dengan TensorBoard (opsional)
5. **Backup hasil** sebelum eksperimen besar

