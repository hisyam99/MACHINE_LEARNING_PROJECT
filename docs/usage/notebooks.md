# Panduan Notebook

## 📓 Daftar Notebook

Proyek ini terdiri dari 3 notebook utama:

1. **TASK_1:** Preprocessing + Feature Extraction + Classic ML Models
2. **TASK_2:** Custom CNN Architecture + LoRA Implementation
3. **TASK_3:** Pretrained DenseNet + Vision Transformer + Augmentation + LoRA

## 📋 TASK 1: Classic ML

### File
`TASK_1:PREPROCESS+FEATEXTRACT+CLASSICMODEL.ipynb`

### Deskripsi
Implementasi Machine Learning Klasik dengan feature engineering manual menggunakan HOG (Histogram of Oriented Gradients) dan berbagai algoritma klasik.

### Sections

1. **Setup & Konfigurasi**
   - Import libraries
   - Set random seeds
   - Define paths

2. **Preprocessing**
   - Load images
   - Apply CLAHE
   - Heuristic lung cropping
   - Normalisasi

3. **Feature Extraction**
   - HOG features extraction
   - Feature selection (SelectKBest)
   - Scaling (StandardScaler)

4. **Model Training**
   - SVM (RBF Kernel)
   - Random Forest
   - k-Nearest Neighbors

5. **Evaluasi**
   - Classification reports
   - Confusion matrices
   - Visualisasi

### Output

- **Models:** Saved di `artifacts/`
- **Features:** HOG features di `artifacts/`
- **Visualizations:** Di `output_images/`

### Waktu Eksekusi
~10-15 menit (CPU)

### Requirements
- CPU sufficient
- Tidak perlu GPU

## 📋 TASK 2: Custom CNN + LoRA

### File
`TASK_2:CNN+LoRA.ipynb`

### Deskripsi
Implementasi Custom CNN dari scratch dengan arsitektur 4 blok konvolusi dan LoRA (Low-Rank Adaptation) untuk efisiensi parameter.

### Sections

1. **Setup & Konfigurasi**
   - Import libraries
   - Set random seeds
   - Define paths dan parameters

2. **Preprocessing**
   - Load dan preprocess images
   - Data loading dengan generator
   - Class weights calculation

3. **Model Architecture**
   - Custom CNN definition
   - LoRA layer implementation
   - Model compilation

4. **Training**
   - Training configuration
   - Callbacks (EarlyStopping, ModelCheckpoint)
   - Training dengan dan tanpa augmentation

5. **Evaluasi**
   - Test set evaluation
   - Classification reports
   - Confusion matrices
   - Prediction visualizations

### Output

- **Models:** Saved di `artifacts/`
- **Training History:** Di notebook
- **Visualizations:** Di `output_images/`

### Waktu Eksekusi
~30-45 menit (GPU recommended)

### Requirements
- GPU recommended (bisa CPU, tapi lebih lambat)
- TensorFlow 2.10+

## 📋 TASK 3: Transfer Learning + ViT

### File
`TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb`

### Deskripsi
Implementasi Transfer Learning menggunakan pretrained models (DenseNet121) dan Vision Transformer (ViT) dengan berbagai konfigurasi.

### Sections

1. **Setup & Konfigurasi**
   - Import libraries
   - Set random seeds
   - Define paths

2. **DenseNet121 + LoRA**
   - Load pretrained DenseNet121
   - Freeze base model
   - Add LoRA head
   - Training dengan augmentation

3. **Vision Transformer (Manual)**
   - ViT implementation dari scratch
   - Patch embedding
   - Transformer blocks
   - Training

4. **HuggingFace ViT Pretrained**
   - Load pretrained ViT
   - Fine-tuning
   - Evaluation

5. **Evaluasi & Perbandingan**
   - Compare semua model
   - Benchmark visualizations
   - Per-class analysis

### Output

- **Models:** Saved di `artifacts/`
- **Training History:** Di notebook
- **Visualizations:** Di `output_images/`
- **Benchmark Comparisons:** Di `output_images/`

### Waktu Eksekusi
~1-3 jam (GPU required)

### Requirements
- GPU required
- TensorFlow 2.10+
- Transformers library (untuk HF ViT)

## 🔧 Tips Menggunakan Notebook

### 1. Jalankan Secara Berurutan

Selalu jalankan cells secara berurutan dari atas ke bawah untuk menghindari error.

### 2. Restart Kernel Jika Perlu

Jika ada error atau perubahan konfigurasi, restart kernel dan jalankan ulang.

### 3. Simpan Checkpoint

Gunakan ModelCheckpoint callback untuk menyimpan model terbaik.

### 4. Monitor Training

Gunakan progress bars dan print statements untuk monitor training.

### 5. Backup Hasil

Simpan hasil penting sebelum eksperimen besar.

## 📊 Struktur Output

Setelah menjalankan notebook, struktur folder akan seperti ini:

```
MACHINE_LEARNING/
├── artifacts/
│   ├── models/
│   ├── features/
│   └── metadata.csv
├── output_images/
│   ├── acc_loss_*.png
│   ├── conf_matrix_*.png
│   ├── *_5_predict_true_false.png
│   └── benchmark_*.png
└── notebooks/
    ├── TASK_1:*.ipynb
    ├── TASK_2:*.ipynb
    └── TASK_3:*.ipynb
```

## 🐛 Troubleshooting

### Kernel Died

```python
# Kurangi batch size
batch_size = 16  # atau lebih kecil

# Atau gunakan data generator
```

### Out of Memory

```python
# Clear session
import tensorflow as tf
tf.keras.backend.clear_session()

# Atau restart kernel
```

### Import Error

```bash
# Install missing packages
pip install package_name

# Atau install semua
pip install -r requirements.txt
```

## 📚 Referensi

- [TASK 1 Detail](../experiments/task1-classic-ml.md)
- [TASK 2 Detail](../experiments/task2-cnn-lora.md)
- [TASK 3 Detail](../experiments/task3-pretrained.md)

