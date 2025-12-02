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

### 1. Custom CNN (No Augmentation)

**Konfigurasi:**
- Tanpa data augmentation
- Baseline untuk perbandingan

**Hasil:**
- **Akurasi:** 71.74%
- **Macro F1:** 0.6586
- **F1 COVID-19:** 0.8342
- **F1 Non-COVID:** 0.4788
- **F1 Normal:** 0.6627

### 2. Custom CNN (With Augmentation)

**Konfigurasi:**
- Dengan data augmentation
- Random rotation, zoom, brightness, flip

**Hasil:**
- **Akurasi:** 81.35%
- **Macro F1:** 0.7825
- **F1 COVID-19:** 0.8901
- **F1 Non-COVID:** 0.6601
- **F1 Normal:** 0.7972

**Peningkatan:**
- ✅ +9.61% akurasi
- ✅ +0.1239 Macro F1
- ✅ F1 Non-COVID meningkat 37.8%

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

### Dampak Data Augmentation

Data augmentation memberikan peningkatan signifikan:

| Metrik | No Aug | With Aug | Improvement |
|:-------|:------:|:--------:|:-----------:|
| **Akurasi** | 71.74% | 81.35% | +9.61% |
| **Macro F1** | 0.6586 | 0.7825 | +0.1239 |
| **F1 Non-COVID** | 0.4788 | 0.6601 | +37.8% |

## 💡 Kesimpulan

1. **Custom CNN cocok untuk deployment mobile/edge** karena ukurannya yang kecil
2. **Data augmentation sangat penting** untuk model from scratch
3. **LoRA membantu efisiensi parameter** tanpa mengorbankan performa
4. **Transfer learning masih lebih baik** untuk akurasi maksimal

[📖 Pelajari lebih lanjut tentang LoRA →](lora.md)

