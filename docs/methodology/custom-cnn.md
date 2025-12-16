# Custom CNN Architecture

## 🏗️ Arsitektur Model

Custom CNN dari scratch dengan arsitektur 4 blok konvolusi dan LoRA untuk efisiensi parameter.

## 📐 Arsitektur Detail

### Struktur Model

```
Input (224×224×1)
    ↓
Conv Block 1 (32 filters)
    ↓
Conv Block 2 (64 filters)
    ↓
Conv Block 3 (128 filters)
    ↓
Conv Block 4 (256 filters)
    ↓
Global Average Pooling
    ↓
LoRA Dense Layer
    ↓
Output (3 classes)
```

### Conv Block Structure

Setiap blok konvolusi terdiri dari:

1. **Conv2D** - Convolutional layer
2. **BatchNorm** - Batch normalization
3. **ReLU** - Activation function
4. **MaxPool** - Max pooling

```python
def conv_block(filters, name_prefix):
    return tf.keras.Sequential([
        layers.Conv2D(
            filters, 
            (3, 3), 
            padding='same',
            name=f'{name_prefix}_conv'
        ),
        layers.BatchNormalization(name=f'{name_prefix}_bn'),
        layers.ReLU(name=f'{name_prefix}_relu'),
        layers.MaxPooling2D((2, 2), name=f'{name_prefix}_pool')
    ])
```

### Filter Progression

| Block | Filters | Output Size |
|:------|:-------:|:-----------:|
| Input | - | 224×224×1 |
| Block 1 | 32 | 112×112×32 |
| Block 2 | 64 | 56×56×64 |
| Block 3 | 128 | 28×28×128 |
| Block 4 | 256 | 14×14×256 |
| GAP | - | 256 |
| LoRA Dense | - | 3 |

## 🔧 Spesifikasi Model

### Parameter

- **Total Parameters:** ~405,731
- **Trainable Parameters:** ~405,731
- **Model Size:** ~1.8 MB (sangat lightweight)

### Input/Output

- **Input Shape:** (224, 224, 1) - Grayscale
- **Output Shape:** (3,) - 3 classes (COVID-19, Non-COVID, Normal)
- **Activation Output:** Softmax

## 🎯 Training Configuration

### Optimizer

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

### Loss Function

```python
# Class weights untuk handle imbalance
class_weights = {
    0: 1.0,  # COVID-19
    1: 2.0,  # Non-COVID (lebih sedikit)
    2: 2.0   # Normal (lebih sedikit)
}

loss = tf.keras.losses.CategoricalCrossentropy()
```

### Callbacks

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )
]
```

## 📊 Variasi Eksperimen

### 1. Pure Custom CNN (No LoRA, No Augmentation)

**Konfigurasi:**
- **Tanpa LoRA** pada layer Dense (menggunakan Dense standar)
- Tanpa data augmentation
- Baseline murni untuk perbandingan

**Hasil:**
- **Akurasi:** 70.37%
- **Macro F1:** 0.6347
- **F1 COVID-19:** 0.7907
- **F1 Non-COVID:** 0.4615
- **F1 Normal:** 0.6517

**Catatan:**
- Memberikan baseline murni tanpa optimasi
- Recall COVID-19 sangat tinggi (97.71%) namun precision rendah (66.41%)
- Menunjukkan pentingnya LoRA dan augmentasi

### 2. Custom CNN + LoRA (No Augmentation)

**Konfigurasi:**
- Dengan LoRA pada layer Dense
- Tanpa data augmentation
- Baseline untuk mengukur dampak augmentasi

**Hasil:**
- **Akurasi:** 71.74%
- **Macro F1:** 0.6586
- **F1 COVID-19:** 0.8342
- **F1 Non-COVID:** 0.4788
- **F1 Normal:** 0.6627

**Peningkatan dari Pure CNN:**
- ✅ +1.37% akurasi
- ✅ +0.0239 Macro F1
- ✅ Efisiensi parameter dengan LoRA

### 3. Custom CNN + LoRA (With Augmentation)

**Konfigurasi:**
- Dengan LoRA pada layer Dense
- Dengan data augmentation
- Random rotation, zoom, brightness, flip

**Hasil:**
- **Akurasi:** 81.35%
- **Macro F1:** 0.7825
- **F1 COVID-19:** 0.8901
- **F1 Non-COVID:** 0.6601
- **F1 Normal:** 0.7972

**Peningkatan dari LoRA No Aug:**
- ✅ +9.61% akurasi
- ✅ +0.1239 Macro F1
- ✅ F1 Non-COVID meningkat 37.8%

**Peningkatan Total dari Pure Baseline:**
- ✅ +10.98% akurasi
- ✅ +0.1478 Macro F1
- ✅ Kombinasi LoRA + Augmentasi sangat efektif

## 🔍 Analisis

### Kelebihan

- ✅ **Sangat lightweight** (~1.8 MB)
- ✅ **Cepat training** (tidak perlu transfer learning)
- ✅ **Efisien parameter** dengan LoRA
- ✅ **Cocok untuk deployment mobile/edge**

### Kekurangan

- ⚠️ **Perlu data augmentation** untuk performa optimal
- ⚠️ **Struggles dengan fine-grained features** (Non-COVID vs Normal)
- ⚠️ **Performa lebih rendah** dibandingkan transfer learning

### Dampak LoRA dan Data Augmentation

**Kontribusi LoRA:**

| Metrik | Pure CNN | CNN + LoRA | Improvement |
|:-------|:--------:|:----------:|:-----------:|
| **Akurasi** | 70.37% | 71.74% | +1.37% |
| **Macro F1** | 0.6347 | 0.6586 | +0.0239 |
| **Efisiensi** | Standar | Parameter efisien | LoRA advantage |

**Kontribusi Data Augmentation:**

| Metrik | No Aug (LoRA) | With Aug (LoRA) | Improvement |
|:-------|:-------------:|:---------------:|:-----------:|
| **Akurasi** | 71.74% | 81.35% | +9.61% |
| **Macro F1** | 0.6586 | 0.7825 | +0.1239 |
| **F1 Non-COVID** | 0.4788 | 0.6601 | +37.8% |

**Dampak Gabungan:**

| Metrik | Pure CNN | CNN + LoRA + Aug | Total Improvement |
|:-------|:--------:|:----------------:|:-----------------:|
| **Akurasi** | 70.37% | 81.35% | +10.98% |
| **Macro F1** | 0.6347 | 0.7825 | +0.1478 |

## 💡 Kesimpulan

1. **Pure CNN baseline menunjukkan keterbatasan model from scratch** tanpa optimasi
2. **LoRA memberikan efisiensi parameter** dengan peningkatan performa moderat (+1.37%)
3. **Data augmentation sangat penting** untuk model from scratch (+9.61% boost)
4. **Kombinasi LoRA + Augmentasi optimal** untuk deployment dengan peningkatan total +10.98%
5. **Custom CNN cocok untuk deployment mobile/edge** karena ukurannya yang kecil (~1.8 MB)
6. **Transfer learning masih lebih baik** untuk akurasi maksimal (82-91%)

[📖 Pelajari lebih lanjut tentang LoRA →](lora.md)

