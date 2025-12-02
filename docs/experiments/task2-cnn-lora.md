# TASK 2: Custom CNN + LoRA

## 📋 Deskripsi

Task ini mengimplementasikan Custom CNN dari scratch dengan arsitektur 4 blok konvolusi dan LoRA (Low-Rank Adaptation) untuk efisiensi parameter.

## 🏗️ Arsitektur

### Model Structure

```
Input (224×224×1)
    ↓
Conv Block 1 (32 filters, 3×3)
    ↓ BatchNorm → ReLU → MaxPool
Conv Block 2 (64 filters, 3×3)
    ↓ BatchNorm → ReLU → MaxPool
Conv Block 3 (128 filters, 3×3)
    ↓ BatchNorm → ReLU → MaxPool
Conv Block 4 (256 filters, 3×3)
    ↓ BatchNorm → ReLU → MaxPool
Global Average Pooling
    ↓
LoRA Dense (128 units, rank=4)
    ↓
LoRA Dense (3 units, rank=4) → Softmax
    ↓
Output (3 classes)
```

### Parameter

- **Total Parameters:** ~405,731
- **Trainable Parameters:** ~405,731
- **Model Size:** ~1.8 MB

## 🔬 Implementasi

### Model Definition

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn():
    model = models.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 1)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Conv Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # LoRA Dense Layers
        LoRADense(128, rank=4, alpha=32),
        LoRADense(3, rank=4, alpha=32, activation='softmax')
    ])
    
    return model
```

### Training Configuration

```python
# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Loss
loss = tf.keras.losses.CategoricalCrossentropy()

# Class Weights
class_weights = {
    0: 1.0,  # COVID-19
    1: 2.0,  # Non-COVID
    2: 2.0   # Normal
}

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

## 📊 Eksperimen

### Eksperimen 1: Tanpa Augmentation

**Konfigurasi:**
- Tidak menggunakan data augmentation
- Baseline untuk perbandingan

**Hasil:**
- **Akurasi:** 71.74%
- **Macro F1:** 0.6586
- **F1 COVID-19:** 0.8342
- **F1 Non-COVID:** 0.4788
- **F1 Normal:** 0.6627

**Analisis:**
- Overfitting mulai terjadi di Epoch 8
- Sensitivitas COVID-19 sangat tinggi
- Struggles dengan Non-COVID (F1 hanya 0.4788)

### Eksperimen 2: Dengan Augmentation

**Konfigurasi:**
- Data augmentation diterapkan:
  - Random Rotation (±15°)
  - Random Zoom (0.9-1.1)
  - Random Brightness/Contrast
  - Horizontal Flip (50%)

**Hasil:**
- **Akurasi:** 81.35%
- **Macro F1:** 0.7825
- **F1 COVID-19:** 0.8901
- **F1 Non-COVID:** 0.6601
- **F1 Normal:** 0.7972

**Peningkatan:**
- ✅ +9.61% akurasi
- ✅ +0.1239 Macro F1
- ✅ F1 Non-COVID meningkat 37.8% (0.4788 → 0.6601)

## 📈 Training Curves

### Tanpa Augmentation

- **Training Accuracy:** Mencapai ~85% di epoch awal
- **Validation Accuracy:** Plateau di ~72%
- **Overfitting:** Mulai di epoch 8

### Dengan Augmentation

- **Training Accuracy:** Lebih stabil, mencapai ~82%
- **Validation Accuracy:** Mengikuti training dengan baik
- **Overfitting:** Terkurangi signifikan

## 🔍 Analisis

### Kelebihan

1. **Sangat Lightweight:** Hanya ~1.8 MB
2. **Cepat Training:** Tidak perlu transfer learning
3. **Efisien Parameter:** LoRA mengurangi parameter
4. **Cocok untuk Mobile/Edge:** Ukuran kecil, performa baik

### Keterbatasan

1. **Perlu Augmentation:** Tanpa augmentation performa turun drastis
2. **Struggles dengan Fine-grained:** Kesulitan membedakan Non-COVID vs Normal
3. **Performa Lebih Rendah:** Dibandingkan transfer learning

## 💡 Kesimpulan

1. **Data Augmentation sangat penting** untuk model from scratch
2. **LoRA membantu efisiensi** tanpa mengorbankan performa
3. **Model cocok untuk deployment mobile/edge** karena ukurannya kecil
4. **Transfer learning masih lebih baik** untuk akurasi maksimal

## 📊 Visualisasi

Lihat visualisasi lengkap di:
- Training Curves: `output_images/acc_loss_custom_cnn_lora_*.png`
- Confusion Matrix: `output_images/conf_matrix_custom_cnn_lora_*.png`
- Predictions: `output_images/custom_cnn_lora_*_5_predict_true_false.png`

## 🔗 Referensi

- [Notebook: TASK_2:CNN+LoRA.ipynb](../../TASK_2:CNN+LoRA.ipynb)
- [Custom CNN Architecture](../methodology/custom-cnn.md)
- [LoRA Implementation](../methodology/lora.md)

